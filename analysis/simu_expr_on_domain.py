import numpy as np
import pandas as pd
import anndata
from scipy.stats import nbinom
from typing import Literal, Dict, Optional, List

def simulate_gene_expression(
    adata: anndata.AnnData,
    domain_key: str,
    n_genes: int = 50,
    distribution: Literal['NB', 'Poisson'] = 'NB',
    diff_genes_ratio: float = 0.9,
    base_mean: float = 10.0,
    dispersion: float = 1.0, # 仅用于 NB, 对应 n (size) 参数
    pattern_strength: float = 1.0, # 控制空间梯度的强度 (0表示无梯度, 1表示从0到2倍均值变化)
    active_domain_prob: float = 0.5, # 差异基因在多少比例的 domain 中是活跃的
    seed: int = 42
) -> anndata.AnnData:
    """
    生成具有复杂空间 Pattern 的模拟基因表达数据。
    
    特性:
    1. 非管家基因只在部分 Domain 表达，其余为 0。
    2. 同一个基因在不同活跃 Domain 可以拥有完全不同的空间 Pattern (线性XYZ, 径向等)。
    """
    np.random.seed(seed)
    n_cells = adata.shape[0]
    domains = adata.obs[domain_key].unique()
    n_domains = len(domains)
    
    # 1. 初始化表达矩阵 (全 0)
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    
    # 2. 区分管家基因和差异基因
    n_diff = int(n_genes * diff_genes_ratio)
    # 索引 0 ~ n_diff-1 是差异基因，剩下的管家基因
    diff_gene_indices = np.arange(n_diff)
    hk_gene_indices = np.arange(n_diff, n_genes)
    
    gene_names = ([f'Gene_{i}_Diff' for i in range(n_diff)] + 
                  [f'Gene_{i}_HK' for i in range(n_diff, n_genes)])
    
    # 用于记录 Ground Truth 信息 (哪个基因在哪个 domain 是什么 pattern)
    # 结构: domain -> gene_idx -> pattern_name
    gt_patterns = {d: {} for d in domains}
    
    # 定义可用的 Pattern 类型
    # constant: 均匀分布
    # linear_x/y/z_inc: 沿轴递增
    # linear_x/y/z_dec: 沿轴递减
    # radial_out: 中心低四周高
    # radial_in: 中心高四周低
    pattern_types = [
        'constant', 
        'linear_x_inc', 'linear_y_inc', 'linear_z_inc',
        'linear_x_dec', 'linear_y_dec', 'linear_z_dec',
        'radial_out', 'radial_in'
    ]
    
    # 3. 逐个 Domain 进行处理 (这样方便计算该 Domain 的局部坐标)
    obs_domains = adata.obs[domain_key].values
    coords_full = adata.obsm['spatial']
    
    for d in domains:
        # 获取当前 Domain 的细胞
        mask = (obs_domains == d)
        if np.sum(mask) == 0: continue
        
        domain_coords = coords_full[mask]
        n_domain_cells = domain_coords.shape[0]
        
        # --- A. 计算局部归一化坐标 (0~1) ---
        c_min = domain_coords.min(axis=0)
        c_max = domain_coords.max(axis=0)
        c_range = c_max - c_min
        c_range[c_range == 0] = 1.0 # 防止除以0
        
        # 归一化 XYZ
        norm_xyz = (domain_coords - c_min) / c_range # shape (N, 3)
        norm_x, norm_y, norm_z = norm_xyz[:, 0], norm_xyz[:, 1], norm_xyz[:, 2]
        
        # 归一化径向距离 (Radial)
        center = domain_coords.mean(axis=0)
        dists = np.linalg.norm(domain_coords - center, axis=1)
        max_dist = dists.max() if dists.max() > 0 else 1.0
        norm_r = dists / max_dist
        
        # --- B. 确定本 Domain 中活跃的基因 ---
        # 1. 管家基因：全部活跃
        # 2. 差异基因：随机活跃
        is_active_diff = np.random.rand(n_diff) < active_domain_prob
        active_diff_indices = diff_gene_indices[is_active_diff]
        
        active_genes_in_this_domain = np.concatenate([active_diff_indices, hk_gene_indices])
        
        # --- C. 为活跃基因分配 Pattern ---
        # 创建一个 multiplier 矩阵，初始为 1.0
        # shape: (n_domain_cells, n_total_genes) - 但我们只更新活跃的列
        domain_mu = np.zeros((n_domain_cells, n_genes), dtype=np.float32)
        
        # 对于管家基因，主要使用 constant，偶尔加一点点微弱 pattern 也可以，这里设为 constant
        # 对于差异基因，随机选择 pattern
        
        # 批量分配 pattern 以提高效率
        # 为每个活跃的 diff 基因随机选一个 pattern ID
        diff_patterns_ids = np.random.choice(len(pattern_types), size=len(active_diff_indices))
        
        # 构建 Pattern 向量字典
        # 将 Pattern 映射到具体的 multiplier 向量 (shape: n_domain_cells)
        # 基础 multiplier 范围: [1 - strength, 1 + strength]
        # 例如 strength=0.5, 范围 [0.5, 1.5]
        low = max(0.0, 1.0 - pattern_strength)
        high = 1.0 + pattern_strength
        
        vectors = {}
        vectors['constant'] = np.ones(n_domain_cells)
        # Linear Increasing: 0 -> 1 映射到 low -> high
        vectors['linear_x_inc'] = low + (high - low) * norm_x
        vectors['linear_y_inc'] = low + (high - low) * norm_y
        vectors['linear_z_inc'] = low + (high - low) * norm_z
        # Linear Decreasing: 0 -> 1 映射到 high -> low
        vectors['linear_x_dec'] = high - (high - low) * norm_x
        vectors['linear_y_dec'] = high - (high - low) * norm_y
        vectors['linear_z_dec'] = high - (high - low) * norm_z
        # Radial
        vectors['radial_out'] = low + (high - low) * norm_r     # 中心低，四周高
        vectors['radial_in']  = high - (high - low) * norm_r    # 中心高，四周低
        
        # --- D. 填充均值矩阵 ---
        
        # 1. 处理差异基因 (Diff)
        for idx, pattern_id in zip(active_diff_indices, diff_patterns_ids):
            pat_name = pattern_types[pattern_id]
            domain_mu[:, idx] = base_mean * vectors[pat_name]
            gt_patterns[d][gene_names[idx]] = pat_name # 记录 GT
            
        # 2. 处理管家基因 (HK) - 默认为 Constant
        # 也可以给管家基因加一点随机波动，这里保持简单 constant
        domain_mu[:, hk_gene_indices] = base_mean * vectors['constant'].reshape(-1, 1)
        for idx in hk_gene_indices:
            gt_patterns[d][gene_names[idx]] = 'constant'

        # --- E. 采样生成 Counts ---
        # 仅对活跃基因进行采样，非活跃基因保持为 0
        active_mask_cols = active_genes_in_this_domain
        
        if len(active_mask_cols) > 0:
            mus = domain_mu[:, active_mask_cols]
            
            if distribution == 'Poisson':
                # Poisson 只有 mu 参数
                counts = np.random.poisson(mus)
            else: # Negative Binomial
                # Scipy nbinom parametrization:
                # mean = n * (1-p) / p  =>  p = n / (n + mean)
                # n = 1 / dispersion (or just dispersion param if strictly defined)
                # 这里我们假设 dispersion 参数直接等于 n (size)
                n = 1.0 / dispersion 
                p = n / (n + mus + 1e-9) # 加 epsilon 防止除0
                counts = nbinom.rvs(n=n, p=p)
            
            X[np.ix_(mask, active_mask_cols)] = counts

    # 4. 创建 AnnData
    sim_adata = anndata.AnnData(X=X, obs=adata.obs.copy(), obsm=adata.obsm.copy())
    sim_adata.var_names = gene_names
    
    # 记录详细的 Ground Truth
    sim_adata.uns['simulation_params'] = {
        'distribution': distribution,
        'active_domain_prob': active_domain_prob,
        'pattern_strength': pattern_strength,
        'gene_patterns': gt_patterns # 这是一个复杂的嵌套字典
    }
    
    print(f"Simulation done. Active Diff Genes per domain (avg): {active_domain_prob*100:.1f}%")
    
    return sim_adata


def assign_slices(adata: anndata.AnnData, z_col_idx: int = 2, n_slices: int = 100) -> anndata.AnnData:
    """
    将Z轴坐标离散化为切片ID。
    """
    z_coords = adata.obsm['spatial'][:, z_col_idx]
    z_min, z_max = z_coords.min(), z_coords.max()
    
    # 简单的线性分箱
    bins = np.linspace(z_min, z_max, n_slices + 1)
    # digitize 返回 1 到 n_slices
    slice_ids = np.digitize(z_coords, bins) - 1 
    # 修正边界值
    slice_ids = np.clip(slice_ids, 0, n_slices - 1)
    
    adata.obs['slice_id'] = slice_ids.astype(str) # NTF通常要求slice_id为字符串或分类
    adata.obs['z_numeric'] = z_coords # 保留原始Z坐标
    return adata