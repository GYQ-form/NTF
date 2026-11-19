import argparse
import os
import logging
import numpy as np
import pandas as pd
import anndata
import torch
import matplotlib.pyplot as plt
from simu_expr_on_domain import simulate_gene_expression, assign_slices
# Assumes NTF training is accessible via NTF.train
from NTF.train import train 
from NTF.sample import sample_points
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from NTF.config import add_shared_args, process_args
from train_dropout_real import calculate_spatial_metrics
import scanpy as sc
from tqdm import tqdm

# This simulation script generates high-resolution spatial transcriptomics data
# based on a template dataset with domain annotations. It then trains a model
# using varying levels of data sparsity (by selecting slices at different intervals)
# and evaluates the reconstruction performance on the full dataset.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_args():
    # 1. Build shared parent parser
    shared_parser = add_shared_args()
    
    # 2. Build main parser, inheriting shared arguments
    parser = argparse.ArgumentParser(
        description="Train model and generate high-resolution data.",
        parents=[shared_parser] 
    )
    
    # 3. Add script-specific arguments
    task_group = parser.add_argument_group('Task Specific: Simulation on slice intervals')
    task_group.add_argument('--is_simulate', action='store_true', help='The input data is already simulated data')
    task_group.add_argument('--domain_key', type=str, default='domain')
    task_group.add_argument('--n_genes', type=int, default=100)
    task_group.add_argument('--total_slices', type=int, default=100)
    task_group.add_argument('--intervals', type=int, nargs='+', default=[2, 3, 4, 5, 6, 7, 8, 9, 10], help='Intervals k to test (e.g., 1 means use all, 2 means use every 2nd)')
    
    # 4. Parse arguments
    args = parser.parse_args()
    
    # 5. Post-process arguments
    args = process_args(args)
    
    return args

def get_reconstruction(model, adata_full, scale_factor, device):

    rec_adata = adata_full.copy()
    coords = torch.from_numpy(rec_adata.obsm['spatial']).float().to(device)
    scaled_coords = coords * scale_factor
    
    model.eval()
    with torch.no_grad():
        res = sample_points(model, scaled_coords, batch_size=8192) 
        
    pred_expr = res['expression'].cpu().numpy()
    rec_adata.layers['prediction'] = pred_expr

    return rec_adata

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.is_simulate:
        logging.info("Input data is already simulated. Skipping simulation step.")
        sim_adata = anndata.read_h5ad(args.input_data)
    else:
        # 1. Load template and generate ground truth
        logging.info("Generating Ground Truth Simulation Data...")
        template_adata = anndata.read_h5ad(args.input_data)

        # Simulate gene expression
        sim_adata = simulate_gene_expression(
            template_adata, 
            domain_key=args.domain_key, 
            n_genes=args.n_genes,
        )
        
        # Assign slice IDs
        sim_adata = assign_slices(sim_adata, n_slices=args.total_slices)
        
        # Normalise to 10,000 counts per cell
        sc.pp.normalize_total(sim_adata, target_sum=1e4)
        # Log1p transform
        sc.pp.log1p(sim_adata)
        # Save ground truth for reference
        sim_adata.write_h5ad(os.path.join(args.output_dir, "ground_truth_full.h5ad"),compression='gzip')

    # calculate spatial scaling factor
    raw_bbox_size = (sim_adata.obsm['spatial'].max(0) - sim_adata.obsm['spatial'].min(0)).max()
    spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
    logging.info(f"Scaling factor: {spatial_scaling:.6f}")
    
    results = []
    log_lines = []

    # 2. Experiment loop over slice intervals
    for k in args.intervals:
        logging.info(f"=== Starting Experiment with Interval k={k} ===")
        
        # 2.1 Subsample slices at interval k
        all_slice_ids = np.arange(args.total_slices)
        selected_slices = all_slice_ids[::k]  # keep every k-th slice
        selected_slices_str = selected_slices.astype(str)
        
        subset_adata = sim_adata[sim_adata.obs['slice_id'].isin(selected_slices_str)].copy()
        subset_adata.obsm['spatial'] *= spatial_scaling  # apply scaling
        n_train_cells = subset_adata.shape[0]
        n_selected_slices = len(selected_slices)
        
        logging.info(f"Selected {n_selected_slices}/{args.total_slices} slices. Training cells: {n_train_cells}")
        
        if n_train_cells < 100:
            logging.warning("Too few cells, skipping...")
            continue

        # 2.2 Train model on the selected subset of slices
        
        model_wrapper = train(subset_adata, args)
        trained_inr = model_wrapper.inr
        
        # 2.3 Evaluate reconstruction on the full ground truth
        rec_adata =  get_reconstruction(trained_inr, sim_adata, spatial_scaling, args.device)
        
        logging.info(f"--- Evaluation on k={k} ---")
        pearson_corrs = []
        spearman_corrs = []
        
        for i in tqdm(range(rec_adata.n_vars), desc="Calculating gene correlations"):
            p = rec_adata.layers['prediction'][:, i]
            t = rec_adata.X[:, i]
            pearson_corrs.append(pearsonr(t, p)[0])
            spearman_corrs.append(spearmanr(t, p)[0])

        # save correlations to a csv file
        calculate_spatial_metrics(rec_adata)
        rec_adata.var['pearson_corr'] = pearson_corrs
        rec_adata.var['spearman_corr'] = spearman_corrs
        rec_adata.write_h5ad(f'{args.output_dir}/k{k}.h5ad',compression='gzip')
        logging.info(f"Test results of k = {k} saved to {args.output_dir}/k{k}.h5ad")
        rec_adata.var.to_csv(f'{args.output_dir}/gene_metrics_k{k}.csv')

        avg_corr = np.nanmean(pearson_corrs[:90])
        avg_sp_corr = np.nanmean(spearman_corrs[:90])
        avg_rmse = np.nanmean(rec_adata.var['rmse'][:90])
        avg_ssim3d = np.nanmean(rec_adata.var['ssim_3d'][:90])
        log_lines.append("--- Summary ---\n")
        log_lines.append(f"Average Pearson correlation across all valid genes: {avg_corr:.4f}\n")
        log_lines.append(f"Average Spearman correlation across all valid genes: {avg_sp_corr:.4f}\n")
            
        results.append({
            'k': k,
            'n_slices_used': n_selected_slices,
            'n_train_cells': n_train_cells,
            'avg_pearson': avg_corr,
            'avg_spearman': avg_sp_corr,
            'avg_mse': avg_rmse,
            'avg_ssim_3d': avg_ssim3d
        })
        
        # Free GPU memory
        del model_wrapper
        del trained_inr
        torch.cuda.empty_cache()

    with open(f'{args.output_dir}/res.log', 'w') as f:
        f.writelines(log_lines)

    # 3. Save results
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(args.output_dir, 'interval_study_results.csv'), index=False)
    logging.info("Experiment finished.")

if __name__ == "__main__":
    main()