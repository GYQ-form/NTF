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

# 导入我们创建的包
from NTF.train import train
from NTF.sample import sample_points
from NTF.config import add_shared_args, process_args
import scanpy as sc
from NTF.helper_func import normalize_adata

# --- 全局设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sample_points_in_alpha_volume(raw_coords: np.ndarray, n_points: int, dist_tol_multiplier: float = 2.0) -> np.ndarray:
    logging.info(f"开始在复杂组织轮廓内采样 {n_points} 个点 (基于 KD-Tree 距离容差)...")
    
    kdtree = cKDTree(raw_coords)
    distances, _ = kdtree.query(raw_coords, k=2)
    avg_dist = np.mean(distances[:, 1])
    
    threshold = avg_dist * dist_tol_multiplier
    logging.info(f"组织内平均细胞间距: {avg_dist:.4f}, 边界容差阈值: {threshold:.4f}")
    
    min_bound = raw_coords.min(axis=0)
    max_bound = raw_coords.max(axis=0)
    
    bbox_volume = np.prod(max_bound - min_bound)
    if bbox_volume == 0:
        logging.error("边界框体积为0，无法进行采样。")
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
        logging.error("未能采样到任何有效的点，请尝试调大 --dist_tol 参数。")
        return np.array([])
        
    sampled_points = np.vstack(sampled_points)
    
    if collected_count < n_points:
        warnings.warn(f"仅采样到 {collected_count} 个点，少于目标数量 {n_points}。")
    
    final_points = sampled_points[:n_points, :]
    logging.info(f"初步空间距离采样获得 {len(final_points)} 个点。")
    return final_points.astype(np.float32)

def get_args():
    shared_parser = add_shared_args()
    parser = argparse.ArgumentParser(
        description="Train model and generate high-resolution data.",
        parents=[shared_parser] 
    )
    
    task_group = parser.add_argument_group('Task Specific: Generation')
    task_group.add_argument('--n_generate', type=int, default=1000000, help='Number of new points to generate.')
    task_group.add_argument('--dist_tol', type=float, default=20, help='Tolerance multiplier for boundary determination.')
    
    # --- 新增：表达量过滤参数 ---
    task_group.add_argument('--expr_filter_ratio', type=float, default=0.03, 
                            help='筛选极低表达点的阈值比例(默认0.03)。总表达量低于 最高总表达量*此比例 的点将被删除。')
    
    task_group.add_argument('--cat_keys', type=str, nargs='+', default=[], help='Categorical variables for KNN voting.')
    task_group.add_argument('--k_neighbors', type=int, default=5, help='Number of nearest neighbors for voting.')
    
    args = parser.parse_args()
    args = process_args(args)
    return args

def main():
    args = get_args()
    logging.info(f"配置加载完成。结果将保存在: {args.output_dir}")

    # --- 1. 加载并预处理数据 ---
    logging.info(f"从 {args.input_data} 加载数据...")
    adata = anndata.read_h5ad(args.input_data)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)
    
    raw_coords = adata.obsm['spatial'].copy()
    raw_bbox_size = (raw_coords.max(0) - raw_coords.min(0)).max()
    spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
    adata.obsm['spatial'] *= spatial_scaling
    logging.info(f"空间缩放因子: {spatial_scaling:.6f}")

    # --- 2. 训练模型 ---
    t_start = time.time()
    trained_NTF = train(adata, args)
    trained_inr = trained_NTF.inr
    train_time_sec = time.time() - t_start
    logging.info(f"模型训练完成。耗时: {train_time_sec:.2f} 秒。")

    # --- 3. 生成新的高分辨率空间坐标 ---
    new_raw_coords = sample_points_in_alpha_volume(raw_coords, args.n_generate, dist_tol_multiplier=args.dist_tol)
    if new_raw_coords.shape[0] == 0:
        return

    scaled_new_coords = new_raw_coords * spatial_scaling
    new_coords_tensor = torch.from_numpy(scaled_new_coords).float().to(args.device)

    # --- 4. 使用模型进行表达量预测 ---
    logging.info(f"在 {len(new_coords_tensor)} 个新坐标点上进行预测...")
    t_start = time.time()
    prediction_results = sample_points(trained_inr, new_coords_tensor)
    infer_time_sec = time.time() - t_start
    logging.info(f"预测完成。耗时: {infer_time_sec:.2f} 秒。")

    predicted_expression = prediction_results["expression"].cpu().numpy()

    # ================= 5. 新增：基于总表达量的背景过滤 =================
    if args.expr_filter_ratio > 0:
        logging.info("基于总体表达量识别并剔除极低表达的无细胞区域...")
        total_expr_per_spot = predicted_expression.sum(axis=1)
        
        # 计算阈值：最高总表达量的 x%
        expr_threshold = np.max(total_expr_per_spot) * args.expr_filter_ratio
        keep_mask = total_expr_per_spot > expr_threshold
        
        n_before = len(new_raw_coords)
        new_raw_coords = new_raw_coords[keep_mask]
        predicted_expression = predicted_expression[keep_mask]
        
        # 同步截断 dropout 概率矩阵
        if not args.no_dropout and 'dropout_prob' in prediction_results:
            dropout_prob = prediction_results['dropout_prob'].cpu().numpy()[keep_mask]
        
        logging.info(f"表达量过滤完成: 剔除了 {n_before - len(new_raw_coords)} 个极低表达点，保留了 {len(new_raw_coords)} 个高置信度组织点。")
        
        if len(new_raw_coords) == 0:
            logging.error("警告：所有点均被表达量阈值筛选掉！请尝试调低 --expr_filter_ratio 参数。")
            return
    else:
        if not args.no_dropout and 'dropout_prob' in prediction_results:
            dropout_prob = prediction_results['dropout_prob'].cpu().numpy()
    # ===================================================================

    # 创建保留点的索引 DataFrame
    obs_df = pd.DataFrame(index=[f"new_cell_{i}" for i in range(len(new_raw_coords))])

    # --- 6. 基于 KNN 的分类变量投票 (仅对过滤后的高置信度点进行，大幅加速) ---
    if args.cat_keys:
        logging.info(f"使用 KNN (k={args.k_neighbors}) 为保留点进行分类变量多数投票...")
        tree = cKDTree(raw_coords)
        k = min(args.k_neighbors, len(raw_coords))
        distances, indices = tree.query(new_raw_coords, k=k)
        
        if k == 1: indices = indices.reshape(-1, 1)

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

    # --- 7. 创建并保存 AnnData 对象 ---
    adata_hires = anndata.AnnData(X=predicted_expression, obs=obs_df, var=adata.var)
    adata_hires.obsm['spatial'] = new_raw_coords

    if not args.no_dropout and 'dropout_prob' in prediction_results:
        adata_hires.obsm['dropout_prob'] = dropout_prob

    adata_hires.uns['generation_info'] = {
        'source_file': args.input_data,
        'n_original_points': adata.n_obs,
        'n_generated_points': len(adata_hires),
        'expr_filter_ratio': args.expr_filter_ratio,
        'training_time_seconds': train_time_sec,
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
    logging.info(f"成功生成高分辨率数据并保存至: {output_path}")

if __name__ == '__main__':
    main()