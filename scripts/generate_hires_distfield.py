import torch
import anndata
import numpy as np
import logging
import argparse
import os
import time
from scipy.spatial import cKDTree
from scipy.stats import mode
import pandas as pd
import warnings

from NTF.train import train
from NTF.sample import sample_points
from NTF.config import add_shared_args, process_args
import scanpy as sc
from NTF.helper_func import normalize_adata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sample_points_in_alpha_volume(raw_coords: np.ndarray, n_points: int, dist_tol_multiplier: float = 2.0) -> np.ndarray:
    logging.info(f"Sampling {n_points} points inside the complex tissue shape (KD-Tree distance tolerance)...")
    
    kdtree = cKDTree(raw_coords)
    distances, _ = kdtree.query(raw_coords, k=2)
    avg_dist = np.mean(distances[:, 1])
    
    threshold = avg_dist * dist_tol_multiplier
    logging.info(f"Average inter-cell distance: {avg_dist:.4f}, boundary tolerance threshold: {threshold:.4f}")
    
    min_bound = raw_coords.min(axis=0)
    max_bound = raw_coords.max(axis=0)
    
    bbox_volume = np.prod(max_bound - min_bound)
    if bbox_volume == 0:
        logging.error("Bounding box volume is zero; cannot sample.")
        return np.array([])

    sampled_points = []
    collected_count = 0
    chunk_size = n_points * 10
    max_iterations = 100
    iteration = 0
    
    while collected_count < n_points and iteration < max_iterations:
        candidates = np.random.uniform(low=min_bound, high=max_bound, size=(chunk_size, 3))
        dists, _ = kdtree.query(candidates)
        
        mask_inside = dists < threshold
        valid_points = candidates[mask_inside]
        
        sampled_points.append(valid_points)
        collected_count += len(valid_points)
        iteration += 1
        
    if collected_count == 0:
        logging.error("No valid points could be sampled. Try increasing the --dist_tol parameter.")
        return np.array([])
        
    sampled_points = np.vstack(sampled_points)
    
    if collected_count < n_points:
        warnings.warn(f"Only {collected_count} points sampled, fewer than the target {n_points}.")
    
    final_points = sampled_points[:n_points, :]
    logging.info(f"Distance-field sampling obtained {len(final_points)} points.")
    return final_points.astype(np.float32)

def get_args():
    shared_parser = add_shared_args()
    parser = argparse.ArgumentParser(
        description="Train model and generate high-resolution data using distance-field sampling.",
        parents=[shared_parser] 
    )
    
    task_group = parser.add_argument_group('Task Specific: Generation')
    task_group.add_argument('--n_generate', type=int, default=1000000, help='Number of new points to generate.')
    task_group.add_argument('--dist_tol', type=float, default=20, help='Tolerance multiplier for boundary determination.')
    
    task_group.add_argument('--expr_filter_ratio', type=float, default=0.03, 
                            help='Threshold ratio for filtering extremely low-expression points (default 0.03). '
                                 'Points whose total expression is below max_total_expression * this_ratio will be removed.')
    
    task_group.add_argument('--cat_keys', type=str, nargs='+', default=[], help='Categorical variables for KNN voting.')
    task_group.add_argument('--k_neighbors', type=int, default=5, help='Number of nearest neighbors for voting.')
    
    args = parser.parse_args()
    args = process_args(args)
    return args

def main():
    args = get_args()
    logging.info(f"Configuration loaded. Results will be saved to: {args.output_dir}")

    # --- 1. Load and preprocess data ---
    logging.info(f"Loading data from {args.input_data}...")
    adata = anndata.read_h5ad(args.input_data)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)
    
    raw_coords = adata.obsm['spatial'].copy()
    raw_bbox_size = (raw_coords.max(0) - raw_coords.min(0)).max()
    spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
    adata.obsm['spatial'] *= spatial_scaling
    logging.info(f"Spatial scaling factor: {spatial_scaling:.6f}")

    # --- 2. Train model (training time is logged automatically) ---
    trained_NTF = train(adata, args)
    trained_inr = trained_NTF.inr

    # --- 3. Generate new high-resolution spatial coordinates ---
    new_raw_coords = sample_points_in_alpha_volume(raw_coords, args.n_generate, dist_tol_multiplier=args.dist_tol)
    if new_raw_coords.shape[0] == 0:
        return

    scaled_new_coords = new_raw_coords * spatial_scaling
    new_coords_tensor = torch.from_numpy(scaled_new_coords).float().to(args.device)

    # --- 4. Predict expression at new coordinates ---
    logging.info(f"Predicting at {len(new_coords_tensor)} new coordinate points...")
    t_start = time.time()
    prediction_results = sample_points(trained_inr, new_coords_tensor)
    infer_time_sec = time.time() - t_start
    logging.info(f"Prediction complete. Time elapsed: {infer_time_sec:.2f} seconds.")

    predicted_expression = prediction_results["expression"].cpu().numpy()

    # --- 5. Expression-based background filtering ---
    if args.expr_filter_ratio > 0:
        logging.info("Identifying and removing near-empty regions based on total expression...")
        total_expr_per_spot = predicted_expression.sum(axis=1)
        
        # Threshold: fraction of the highest total expression
        expr_threshold = np.max(total_expr_per_spot) * args.expr_filter_ratio
        keep_mask = total_expr_per_spot > expr_threshold
        
        n_before = len(new_raw_coords)
        new_raw_coords = new_raw_coords[keep_mask]
        predicted_expression = predicted_expression[keep_mask]
        
        # Keep dropout probability matrix in sync
        if not args.no_dropout and 'dropout_prob' in prediction_results:
            dropout_prob = prediction_results['dropout_prob'].cpu().numpy()[keep_mask]
        
        logging.info(f"Expression filtering complete: removed {n_before - len(new_raw_coords)} low-expression points, "
                     f"retained {len(new_raw_coords)} high-confidence tissue points.")
        
        if len(new_raw_coords) == 0:
            logging.error("All points were removed by the expression threshold. Try lowering --expr_filter_ratio.")
            return
    else:
        if not args.no_dropout and 'dropout_prob' in prediction_results:
            dropout_prob = prediction_results['dropout_prob'].cpu().numpy()

    obs_df = pd.DataFrame(index=[f"new_cell_{i}" for i in range(len(new_raw_coords))])

    # --- 6. KNN-based majority voting for categorical variables ---
    if args.cat_keys:
        logging.info(f"Running KNN (k={args.k_neighbors}) majority voting for retained points...")
        tree = cKDTree(raw_coords)
        k = min(args.k_neighbors, len(raw_coords))
        distances, indices = tree.query(new_raw_coords, k=k)
        
        if k == 1:
            indices = indices.reshape(-1, 1)

        for key in args.cat_keys:
            if key not in adata.obs:
                continue
            cat_series = pd.Categorical(adata.obs[key])
            codes = cat_series.codes        
            categories = cat_series.categories 
            
            neighbor_codes = codes[indices]
            try:
                voted_modes, _ = mode(neighbor_codes, axis=1, keepdims=False)
            except TypeError:
                voted_modes, _ = mode(neighbor_codes, axis=1)
                
            obs_df[key] = categories[voted_modes.flatten()]
            obs_df[key] = obs_df[key].astype("category")

    # --- 7. Build and save AnnData object ---
    adata_hires = anndata.AnnData(X=predicted_expression, obs=obs_df, var=adata.var)
    adata_hires.obsm['spatial'] = new_raw_coords

    if not args.no_dropout and 'dropout_prob' in prediction_results:
        adata_hires.obsm['dropout_prob'] = dropout_prob

    adata_hires.uns['generation_info'] = {
        'source_file': args.input_data,
        'n_original_points': adata.n_obs,
        'n_generated_points': len(adata_hires),
        'expr_filter_ratio': args.expr_filter_ratio,
        'inference_time_seconds': infer_time_sec,
        'spatial_scaling_factor': spatial_scaling,
        'predicted_categorical_keys': args.cat_keys,
        'knn_neighbors': args.k_neighbors
    }
    
    adata_hires.layers['raw_reconstruction'] = adata_hires.X.copy()
    normalize_adata(adata_hires)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'{int(args.n_generate/10000)}w.h5ad')

    adata_hires.write_h5ad(output_path, compression='gzip')
    logging.info(f"High-resolution data successfully generated and saved to: {output_path}")

if __name__ == '__main__':
    main()
