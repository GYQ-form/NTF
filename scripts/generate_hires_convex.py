import torch
import anndata
import numpy as np
import logging
import argparse
import os
import time
from scipy.spatial import ConvexHull, Delaunay, cKDTree
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

def sample_points_in_hull(hull: ConvexHull, n_points: int) -> np.ndarray:
    logging.info(f"开始在组织轮廓（凸包）内采样 {n_points} 个点...")
    tesselation = Delaunay(hull.points)
    min_bound = hull.points.min(axis=0)
    max_bound = hull.points.max(axis=0)
    
    hull_volume = hull.volume
    bbox_volume = np.prod(max_bound - min_bound)
    acceptance_ratio = hull_volume / bbox_volume if bbox_volume > 0 else 0
    
    if acceptance_ratio == 0:
        logging.error("凸包体积为0，无法进行采样。请检查输入坐标是否共面或共线。")
        return np.array([])

    n_candidates_to_generate = int(n_points / acceptance_ratio * 1.2)
    logging.info(f"凸包体积/边界框体积 = {acceptance_ratio:.3f}。将生成约 {n_candidates_to_generate} 个候选点。")

    candidates = np.random.uniform(low=min_bound, high=max_bound, size=(n_candidates_to_generate, 3))
    mask_inside = tesselation.find_simplex(candidates) >= 0
    points_inside = candidates[mask_inside]
    
    if len(points_inside) >= n_points:
        sampled_points = points_inside[:n_points, :]
    else:
        warnings.warn(
            f"采样到的点数 ({len(points_inside)}) 少于目标数量 ({n_points})。"
            "可能是因为组织形状过于复杂或稀疏。将返回所有采样到的内部点。",
            UserWarning
        )
        sampled_points = points_inside
        
    logging.info(f"成功采样 {len(sampled_points)} 个点。")
    return sampled_points.astype(np.float32)

def get_args():
    shared_parser = add_shared_args()
    parser = argparse.ArgumentParser(
        description="Train model and generate high-resolution data.",
        parents=[shared_parser] 
    )
    
    task_group = parser.add_argument_group('Task Specific: Generation')
    task_group.add_argument('--n_generate', type=int, default=1000000, help='Number of new points to generate.')
    
    # --- 新增: KNN 投票参数 ---
    task_group.add_argument('--cat_keys', type=str, nargs='+', default=[],
                            help='List of categorical variables in adata.obs to predict via KNN voting (e.g., cell_type).')
    task_group.add_argument('--k_neighbors', type=int, default=5,
                            help='Number of nearest neighbors to use for majority voting of categorical variables.')
    
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
    logging.info(f"使用所有 {adata.n_obs} 个细胞进行训练。空间缩放因子: {spatial_scaling:.6f}")

    # --- 2. 训练模型 ---
    t_start = time.time()
    trained_NTF = train(adata, args)
    trained_inr = trained_NTF.inr
    train_time_sec = time.time() - t_start
    logging.info(f"模型训练完成。耗时: {train_time_sec:.2f} 秒。")

    # --- 3. 生成新的高分辨率空间坐标 ---
    logging.info("根据原始数据轮廓计算三维凸包...")
    try:
        hull = ConvexHull(raw_coords)
    except Exception as e:
        logging.error(f"计算凸包失败: {e}")
        return

    new_raw_coords = sample_points_in_hull(hull, args.n_generate)
    if new_raw_coords.shape[0] == 0:
        logging.error("未能生成任何新坐标点，程序终止。")
        return

    scaled_new_coords = new_raw_coords * spatial_scaling
    new_coords_tensor = torch.from_numpy(scaled_new_coords).float().to(args.device)

    # --- 4. 使用模型进行预测 ---
    logging.info(f"在 {len(new_coords_tensor)} 个新坐标点上进行预测...")
    t_start = time.time()
    prediction_results = sample_points(trained_inr, new_coords_tensor)
    infer_time_sec = time.time() - t_start
    logging.info(f"预测完成。耗时: {infer_time_sec:.2f} 秒。")

    predicted_expression = prediction_results["expression"].cpu().numpy()
    obs_df = pd.DataFrame(index=[f"new_cell_{i}" for i in range(len(new_raw_coords))])

    # ================= 5. 新增：基于 KNN 的分类变量投票 =================
    if args.cat_keys:
        logging.info(f"使用 KNN (k={args.k_neighbors}) 为新生成点进行分类变量多数投票...")
        tree = cKDTree(raw_coords)
        k = min(args.k_neighbors, len(raw_coords))
        distances, indices = tree.query(new_raw_coords, k=k)
        
        if k == 1: indices = indices.reshape(-1, 1)

        for key in args.cat_keys:
            if key not in adata.obs:
                logging.warning(f"变量 '{key}' 不在原始数据 adata.obs 中，跳过。")
                continue
            cat_series = pd.Categorical(adata.obs[key])
            codes = cat_series.codes        
            categories = cat_series.categories 
            
            neighbor_codes = codes[indices]
            try:
                voted_modes, _ = mode(neighbor_codes, axis=1, keepdims=False)
            except TypeError:
                voted_modes, _ = mode(neighbor_codes, axis=1)
                
            voted_modes = voted_modes.flatten()
            obs_df[key] = categories[voted_modes]
            obs_df[key] = obs_df[key].astype("category")
        logging.info(f"KNN 分类变量投票完成: {args.cat_keys}")
    # ==========================================================

    adata_hires = anndata.AnnData(X=predicted_expression, obs=obs_df, var=adata.var)
    adata_hires.obsm['spatial'] = new_raw_coords

    if not args.no_dropout and 'dropout_prob' in prediction_results:
        adata_hires.obsm['dropout_prob'] = prediction_results['dropout_prob'].cpu().numpy()

    adata_hires.uns['generation_info'] = {
        'source_file': args.input_data,
        'n_original_points': adata.n_obs,
        'n_generated_points': len(adata_hires),
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