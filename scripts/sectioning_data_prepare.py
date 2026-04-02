import torch
import anndata
import numpy as np
import logging
import argparse
import os
import pickle
import pandas as pd

from NTF.train import train
from NTF.config import add_shared_args, process_args

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    shared_parser = add_shared_args()
    parser = argparse.ArgumentParser(
        description="Train model and prepare data for interactive in-silico sectioning.",
        parents=[shared_parser] 
    )
    parser.add_argument('--alpha_multiplier', type=float, default=14.0, 
                        help='Multiplier for average distance to determine Alpha Shape.')
    parser.add_argument('--cat_keys', type=str, nargs='+', default=[],
                        help='List of categorical variables in adata.obs to prepare for sectioning prediction.')
    
    args = parser.parse_args()
    return process_args(args)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. Load and preprocess data ---
    logging.info(f"Loading data from {args.input_data}...")
    adata = anndata.read_h5ad(args.input_data)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)

    raw_coords = adata.obsm['spatial'].copy()
    raw_bbox_size = (raw_coords.max(0) - raw_coords.min(0)).max()
    spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
    adata.obsm['spatial'] *= spatial_scaling
    
    logging.info(f"Spatial scaling factor: {spatial_scaling:.6f}")

    # --- Extract categorical variable data for KNN voting ---
    obs_categories = {}
    if args.cat_keys:
        for key in args.cat_keys:
            if key in adata.obs:
                cat_series = pd.Categorical(adata.obs[key])
                obs_categories[key] = {
                    'codes': cat_series.codes,
                    'categories': cat_series.categories.tolist()
                }
                logging.info(f"Extracted categorical variable '{key}' ({len(cat_series.categories)} classes)")
            else:
                logging.warning(f"Variable '{key}' not found in adata.obs; skipping.")

    # --- 2. Train model (training time is logged automatically) ---
    logging.info("Training model...")
    trained_NTF = train(adata, args)
    trained_inr = trained_NTF.inr

    # --- 3. Build non-convex mesh surface (Alpha Shape) ---
    logging.info("Computing non-convex Alpha Shape for 3D mesh representation...")
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(raw_coords)
        
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        
        alpha_val = avg_dist * args.alpha_multiplier 
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_val)
        
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        retry_count = 0
        while (len(faces) < 100 or len(vertices) < 10) and retry_count < 5:
            alpha_val *= 1.5
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha_val)
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            retry_count += 1
            
    except ImportError:
        logging.error("Open3D is not installed.")
        return

    # --- 4. Save all required assets ---
    model_path = os.path.join(args.output_dir, 'trained_inr.pt')
    torch.save(trained_inr, model_path)
    
    metadata = {
        'vertices': vertices,
        'faces': faces,
        'raw_coords': raw_coords,      
        'avg_dist': avg_dist,          
        'spatial_scaling': spatial_scaling,
        'gene_names': adata.var.gene_symbol.tolist() if 'gene_symbol' in adata.var.columns else adata.var_names.tolist(),
        'obs_categories': obs_categories,
        'raw_coords_min': raw_coords.min(axis=0),
        'raw_coords_max': raw_coords.max(axis=0)
    }
    
    meta_path = os.path.join(args.output_dir, 'metadata.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
        
    logging.info(f"Assets successfully saved to {args.output_dir}")

if __name__ == '__main__':
    main()
