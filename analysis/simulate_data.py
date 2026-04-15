import anndata
import numpy as np
import pandas as pd
import logging
import argparse
import os
import sys

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_mock_anndata(
    n_obs=500000, 
    n_slices=5, 
    n_clusters=4, 
    genes_per_pattern=3,
    batch_effect_std=0.5,
    noise_std=0.1
):
    """
    创建一个模拟的三维空间转录组AnnData对象。
    修改点：批次效应和噪声仅添加到有真实表达值的地方。
    """
    
    # --- 0. 定义空间表达模式 ---
    patterns = [
        "global_x_gradient",
        "global_y_gradient",
        "global_center_radial_out",
        "cluster_opposing_x_gradient",
        "cluster_opposing_y_gradient",
        "cluster_opposing_z_gradient",
        "cluster_opposing_radial",
    ]
    # 确保 patterns 数量足够
    if len(patterns) < 5:
         used_patterns = np.random.choice(patterns, 5, replace=True)
    else:
         used_patterns = np.choose(np.arange(5), patterns) # 简单取前5个或按逻辑取
         
    n_genes = len(used_patterns) * genes_per_pattern
    
    logging.info(
        f"创建模拟AnnData: {n_obs}个细胞, {n_genes}个基因, {n_slices}个切片, "
        f"{n_clusters}个细胞团。"
    )

    # --- 1. 创建细胞三维坐标和细胞团分配 ---
    logging.info("步骤1: 生成细胞坐标和细胞团...")
    cluster_centers = np.random.uniform(low=0, high=80, size=(n_clusters, 3))
    obs_per_cluster = np.full(n_clusters, n_obs // n_clusters)
    obs_per_cluster[0] += n_obs % n_clusters
    
    coords_list = []
    cluster_ids_list = []
    for i in range(n_clusters):
        cluster_size = np.random.uniform(low=4, high=8)
        # 生成簇内坐标
        c_coords = np.random.randn(obs_per_cluster[i], 3) * cluster_size + cluster_centers[i]
        coords_list.append(c_coords)
        cluster_ids_list.extend([f'cluster_{i}'] * obs_per_cluster[i])

    coords = np.vstack(coords_list)
    shuffle_idx = np.random.permutation(n_obs)
    coords = coords[shuffle_idx]
    cluster_ids = np.array(cluster_ids_list)[shuffle_idx]

    # --- 2. 基于空间模式创建真实基因表达 ---
    logging.info("步骤2: 生成具有高Dropout特性的真实基因表达...")
    X_truth = np.zeros((n_obs, n_genes), dtype=np.float32)
    var_pattern_names = []
    gene_idx = 0

    # 归一化坐标用于计算模式
    coords_norm = (coords - coords.min(0)) / (coords.max(0) - coords.min(0) + 1e-6)
    dist_from_global_center = np.linalg.norm(coords - coords.mean(0), axis=1)
    dist_norm_global = (dist_from_global_center / (dist_from_global_center.max() + 1e-6))

    for pattern_name in used_patterns:
        for _ in range(genes_per_pattern):
            if gene_idx >= n_genes: break
            
            # 随机选择表达该基因的细胞团子集
            n_expressing_clusters = np.random.randint(1, max(2, n_clusters // 2 + 1))
            expressing_cluster_indices = np.random.choice(
                np.arange(n_clusters), n_expressing_clusters, replace=False
            )
            
            expressing_cell_mask = np.isin(
                cluster_ids, [f'cluster_{i}' for i in expressing_cluster_indices]
            )

            # --- 应用模式逻辑 (与之前相同) ---
            if pattern_name.startswith("global"):
                if pattern_name == "global_x_gradient":
                    X_truth[expressing_cell_mask, gene_idx] = coords_norm[expressing_cell_mask, 0] * 2.0
                elif pattern_name == "global_y_gradient":
                    X_truth[expressing_cell_mask, gene_idx] = coords_norm[expressing_cell_mask, 1] * 2.0
                elif pattern_name == "global_center_radial_out":
                    X_truth[expressing_cell_mask, gene_idx] = dist_norm_global[expressing_cell_mask] * 2.0
            
            elif pattern_name.startswith("cluster"):
                permuted_selection = np.random.permutation(expressing_cluster_indices)
                n_positive = len(permuted_selection) // 2
                positive_clusters = permuted_selection[:n_positive]
                negative_clusters = permuted_selection[n_positive:]

                if pattern_name == "cluster_opposing_x_gradient":
                    for i in positive_clusters:
                        mask = (cluster_ids == f'cluster_{i}')
                        X_truth[mask, gene_idx] = coords_norm[mask, 0] * 2.0
                    for i in negative_clusters:
                        mask = (cluster_ids == f'cluster_{i}')
                        X_truth[mask, gene_idx] = (1 - coords_norm[mask, 0]) * 2.0
                
                if pattern_name == "cluster_opposing_y_gradient":
                    for i in positive_clusters:
                        mask = (cluster_ids == f'cluster_{i}')
                        X_truth[mask, gene_idx] = coords_norm[mask, 1] * 2.0
                    for i in negative_clusters:
                        mask = (cluster_ids == f'cluster_{i}')
                        X_truth[mask, gene_idx] = (1 - coords_norm[mask, 1]) * 2.0

                if pattern_name == "cluster_opposing_z_gradient":
                    for i in positive_clusters:
                        mask = (cluster_ids == f'cluster_{i}')
                        X_truth[mask, gene_idx] = coords_norm[mask, 2] * 2.0
                    for i in negative_clusters:
                        mask = (cluster_ids == f'cluster_{i}')
                        X_truth[mask, gene_idx] = (1 - coords_norm[mask, 2]) * 2.0
                
                elif pattern_name == "cluster_opposing_radial":
                    for i in positive_clusters:
                        mask = (cluster_ids == f'cluster_{i}')
                        dist_from_center = np.linalg.norm(coords[mask] - cluster_centers[i], axis=1)
                        if dist_from_center.max() > 0:
                            X_truth[mask, gene_idx] = (dist_from_center / dist_from_center.max()) * 2.0
                    for i in negative_clusters:
                        mask = (cluster_ids == f'cluster_{i}')
                        dist_from_center = np.linalg.norm(coords[mask] - cluster_centers[i], axis=1)
                        if dist_from_center.max() > 0:
                            X_truth[mask, gene_idx] = (1 - dist_from_center / dist_from_center.max()) * 2.0

            elif pattern_name == "random_noise":
                X_truth[expressing_cell_mask, gene_idx] = np.random.rand(np.sum(expressing_cell_mask)) * 1.5

            var_pattern_names.append(pattern_name)
            gene_idx += 1

    # --- 3. 创建切片ID ---
    logging.info("步骤3: 根据Z轴坐标分配切片ID...")
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    slice_bins = np.linspace(z_min, z_max, n_slices + 1)
    slice_bins[0] -= 1e-6; slice_bins[-1] += 1e-6
    slice_indices = np.digitize(coords[:, 2], bins=slice_bins)
    slice_ids = np.array([f'slice_{i}' for i in slice_indices]) # 转为numpy array方便索引

    # --- 4. 模拟批次效应并创建观测值 (关键修改) ---
    logging.info("步骤4: 模拟切片批次效应和噪声 (仅在有表达处添加)...")
    X_observed = np.copy(X_truth)
    
    unique_slice_ids = np.unique(slice_ids)
    
    # 4.1 添加批次效应 (仅在 X_truth > 0 的地方)
    for s_id in unique_slice_ids:
        # 获取当前切片的掩码
        slice_mask = (slice_ids == s_id)
        
        # 为该切片生成每个基因的固定偏移量 (batch effect)
        # shape: (n_genes,)
        slice_gene_effect = np.random.normal(0, batch_effect_std, n_genes)
        
        # 找到当前切片中，有真实表达值的位置 (non-zero elements)
        # 我们只在这些位置加上 batch effect
        # 使用 where 获取切片内的非零索引
        # 注意：这里需要操作的是 X_observed[slice_mask, :] 这个子矩阵
        
        # 为了高效，我们可以先切出子矩阵
        sub_matrix = X_observed[slice_mask, :]
        
        # 创建一个与子矩阵同形状的 effect 矩阵
        effect_matrix = np.tile(slice_gene_effect, (sub_matrix.shape[0], 1))
        
        # 核心逻辑：只在 sub_matrix > 0 的位置加上 effect
        nonzero_mask = sub_matrix > 0
        sub_matrix[nonzero_mask] += effect_matrix[nonzero_mask]
        
        # 将修改后的子矩阵放回原数组
        X_observed[slice_mask, :] = sub_matrix

    # 4.2 添加随机高斯噪声 (仅在 X_truth > 0 的地方)
    # 生成全量噪声矩阵
    noise_matrix = np.random.randn(n_obs, n_genes) * noise_std
    
    # 获取全非零掩码
    global_nonzero_mask = X_observed > 0
    
    # 仅在非零位置叠加噪声
    X_observed[global_nonzero_mask] += noise_matrix[global_nonzero_mask]
    
    # 再次确保非负
    X_observed[X_observed < 0] = 0

    # --- 5. 划分训练集和测试集 (按切片划分) ---
    logging.info("步骤5: 随机划分训练集/测试集切片并组装AnnData...")
    
    # 随机打乱切片ID列表
    shuffled_slice_ids = np.random.permutation(unique_slice_ids)
    split_idx = len(shuffled_slice_ids) // 2
    
    test_slices = set(shuffled_slice_ids[:split_idx])
    train_slices = set(shuffled_slice_ids[split_idx:])
    
    # 创建 dataset 标签列
    dataset_labels = []
    for s_id in slice_ids:
        if s_id in train_slices:
            dataset_labels.append('train')
        else:
            dataset_labels.append('test')
            
    logging.info(f"训练集切片数: {len(train_slices)}, 测试集切片数: {len(test_slices)}")

    # --- 6. 组装AnnData对象 ---
    obs_df = pd.DataFrame({
        'slice_id': pd.Categorical(slice_ids),
        'cluster_id': pd.Categorical(cluster_ids),
        'group': pd.Categorical(dataset_labels) # 新增 dataset 列
    }, index=[f'cell_{i}' for i in range(n_obs)])
    
    var_df = pd.DataFrame({
        'pattern_name': var_pattern_names
    }, index=[f'gene_{i}' for i in range(n_genes)])

    adata = anndata.AnnData(
        X=X_observed, 
        obs=obs_df,
        var=var_df,
        obsm={'spatial': coords},
        layers={'truth': X_truth}
    )
    
    logging.info("模拟数据创建完成。")
    return adata

def parse_args():
    parser = argparse.ArgumentParser(description="生成三维空间转录组模拟数据")

     # 可选参数
    parser.add_argument("--n_slices", type=int, default=100, help="Z轴切片数量 (默认: 100)")
    parser.add_argument("--n_clusters", type=int, default=10, help="细胞团数量 (默认: 10)")
    parser.add_argument("--genes_per_pattern", type=int, default=20, help="每种模式重复的基因数 (默认: 20)")
    parser.add_argument("--batch_effect", type=float, default=0.2, help="切片批次效应标准差 (默认: 0.2)")
    parser.add_argument("--noise", type=float, default=0.1, help="高斯噪声标准差 (默认: 0.1)")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # 确保输出目录存在
    args.output = f'/home/gongyuqiao/ur_annotation/NTF/mytrain/data/simulation/scale_test/{args.n_slices}w.h5ad'
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logging.info(f"创建输出目录: {out_dir}")

    # 调用生成函数
    adata = create_mock_anndata(
        n_obs=args.n_slices*10000,
        n_slices=args.n_slices,
        n_clusters=args.n_clusters,
        genes_per_pattern=args.genes_per_pattern,
        batch_effect_std=args.batch_effect,
        noise_std=args.noise
    )
    
    # 打印一些统计信息
    logging.info("数据生成完毕，统计信息如下:")
    print(adata)
    print("\nObs 分布情况:")
    print(adata.obs['group'].value_counts())
    
    # 保存
    logging.info(f"正在保存至: {args.output}")
    adata.write_h5ad(args.output, compression='gzip')
    logging.info("完成！")