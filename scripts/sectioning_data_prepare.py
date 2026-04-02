import torch
import anndata
import numpy as np
import logging
import argparse
import os
import pickle
import pandas as pd

# 导入你的包
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
    # 新增参数：保存定性变量供后续 KNN 投票
    parser.add_argument('--cat_keys', type=str, nargs='+', default=[],
                        help='List of categorical variables in adata.obs to prepare for sectioning prediction.')
    
    args = parser.parse_args()
    return process_args(args)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. 加载并预处理数据 ---
    logging.info(f"Loading data from {args.input_data}...")
    adata = anndata.read_h5ad(args.input_data)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)

    raw_coords = adata.obsm['spatial'].copy()
    raw_bbox_size = (raw_coords.max(0) - raw_coords.min(0)).max()
    spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
    adata.obsm['spatial'] *= spatial_scaling
    
    logging.info(f"Spatial scaling factor: {spatial_scaling:.6f}")

    # ================= 提取定性变量数据 =================
    obs_categories = {}
    if args.cat_keys:
        for key in args.cat_keys:
            if key in adata.obs:
                cat_series = pd.Categorical(adata.obs[key])
                obs_categories[key] = {
                    'codes': cat_series.codes,                      # 整数数组
                    'categories': cat_series.categories.tolist()    # 字符串列表
                }
                logging.info(f"提取定性变量 '{key}' (共 {len(cat_series.categories)} 类)")
            else:
                logging.warning(f"变量 '{key}' 不在 adata.obs 中，跳过。")
    # ====================================================

    # --- 2. 训练模型 ---
    logging.info("Training model...")
    trained_NTF = train(adata, args)
    trained_inr = trained_NTF.inr

    # --- 3. 生成非凸 Mesh 表面数据 (Alpha Shape) ---
    logging.info("Computing Non-convex Alpha Shape for 3D mesh representation...")
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

    # --- 4. 保存所有需要的资产 ---
    model_path = os.path.join(args.output_dir, 'trained_inr.pt')
    torch.save(trained_inr, model_path)
    
    metadata = {
        'vertices': vertices,
        'faces': faces,
        'raw_coords': raw_coords,      
        'avg_dist': avg_dist,          
        'spatial_scaling': spatial_scaling,
        'gene_names': adata.var.gene_symbol.tolist() if 'gene_symbol' in adata.var.columns else adata.var_names.tolist(),
        'obs_categories': obs_categories, # 存入定性变量用于 KNN
        'raw_coords_min': raw_coords.min(axis=0),
        'raw_coords_max': raw_coords.max(axis=0)
    }
    
    meta_path = os.path.join(args.output_dir, 'metadata.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
        
    logging.info(f"Assets successfully saved to {args.output_dir}")

if __name__ == '__main__':
    main()