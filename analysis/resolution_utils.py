import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from scipy.spatial import cKDTree

def calculate_avg_cell_distance(coords: np.ndarray) -> float:
    """计算平均细胞间距（基于最近邻）"""
    if len(coords) < 2:
        return 1.0
    tree = cKDTree(coords)
    # 查找每个点最近的2个邻居（第1个是自己，第2个是最近邻）
    dists, _ = tree.query(coords, k=2) 
    return np.mean(dists[:, 1])

def degrade_resolution(
    adata_slice: anndata.AnnData, 
    bin_factor: float = 2.0
) -> anndata.AnnData:
    """
    将单个切片的分辨率降低。
    
    Args:
        adata_slice: 单个切片的 AnnData
        bin_factor: 网格大小倍数。1.0 表示网格大小约等于细胞平均间距。
                    2.0 表示网格边长是平均间距的2倍（面积约4倍）。
    
    Returns:
        AnnData: 降维后的数据 (obs数减少)
    """
    coords = adata_slice.obsm['spatial']
    
    if len(coords) == 0:
        return adata_slice
        
    # 1. 确定网格大小
    avg_dist = calculate_avg_cell_distance(coords)
    grid_size = avg_dist * bin_factor
    
    # 2. 建立网格索引
    # 将坐标除以 grid_size 并取整，得到 grid ID (ix, iy)
    # 注意：这里我们只关心 x, y 平面上的 binning，z 轴因为是切片内，通常变化极小或是常数
    # 但为了通用性，我们对三维都做处理，或者假设 z 是常数
    grid_indices = np.floor(coords / grid_size).astype(int)
    
    # 3. 聚合数据
    # 使用 pandas 的 groupby 功能快速聚合
    df_coords = pd.DataFrame(grid_indices, columns=['gx', 'gy', 'gz'])
    df_coords['original_index'] = np.arange(len(coords))
    
    # 组合成唯一的 grid key
    # 简单的做法是 group by ['gx', 'gy'] (假设 z 是一样的)
    # 如果 z 有微小抖动，也应该被归为一个 bin
    grouped = df_coords.groupby(['gx', 'gy', 'gz'])
    
    new_coords = []
    new_expr = []
    new_slice_ids = []
    
    # 原始表达矩阵 (假设是 dense numpy array，如果是 sparse 需要转换)
    raw_X = adata_slice.X
    if hasattr(raw_X, 'toarray'):
        raw_X = raw_X.toarray()
        
    raw_slice_ids = adata_slice.obs['slice_id'].values
    
    # 遍历每个 Grid
    # 注意：这种循环在 Python 中如果不优化可能会慢。
    # 对于数千个点还可以，如果非常多建议用 scipy.stats.binned_statistic_2d 或 torch_scatter
    
    # 优化方案：利用 unique grid indices 向量化操作
    # 这里为了代码清晰度，且切片内细胞数通常有限（几千到几万），先用 Pandas
    
    # 计算每个 grid 包含的原始 indices
    indices_groups = grouped.indices # dict: key -> array of indices
    
    for grid_key, indices in indices_groups.items():
        # 1. 坐标：取几何中心
        cell_coords = coords[indices]
        centroid = np.mean(cell_coords, axis=0)
        new_coords.append(centroid)
        
        # 2. 表达：平均值 (Normalize: 模拟测序深度归一化后的效果)
        cell_expr = raw_X[indices]
        avg_expr = np.mean(cell_expr, axis=0)
        new_expr.append(avg_expr)
        
        # 3. Slice ID: 保持不变
        new_slice_ids.append(raw_slice_ids[indices[0]])
        
    if len(new_coords) == 0:
        return adata_slice[[]] # empty
        
    new_coords = np.array(new_coords)
    new_expr = np.array(new_expr)
    new_slice_ids = np.array(new_slice_ids)
    
    # 4. 构建新的 AnnData
    new_adata = anndata.AnnData(X=new_expr)
    new_adata.obsm['spatial'] = new_coords
    new_adata.obs['slice_id'] = new_slice_ids
    new_adata.var_names = adata_slice.var_names
    # 标记它是低分辨率
    new_adata.obs['resolution_type'] = 'low_res'
    new_adata.obs['bin_factor'] = bin_factor
    
    return new_adata

def create_mixed_resolution_dataset(
    adata_full: anndata.AnnData,
    slice_interval_k: int = 5,
    high_res_interval_m: int = 2,
    bin_factor: float = 2.0
) -> anndata.AnnData:
    """
    创建混合分辨率数据集。
    
    Args:
        slice_interval_k: 切片采样间隔 (每 k 张取一张)
        high_res_interval_m: 在选中的切片中，每 m 张保留一张高分辨率
                             例如 m=1: 全部高分; m=2: 高, 低, 高, 低...
        bin_factor: 低分辨率的聚合因子
    """
    # 1. 筛选切片
    all_slices = np.sort(adata_full.obs['slice_id'].unique().astype(int))
    selected_slice_ids = all_slices[::slice_interval_k].astype(str)
    
    subset_adata = adata_full[adata_full.obs['slice_id'].isin(selected_slice_ids)].copy()
    
    # 2. 处理每个切片
    processed_adatas = []
    
    # 按 slice_id 排序处理
    # 注意：slice_id 是字符串，要确保排序正确
    sorted_ids = sorted(selected_slice_ids, key=lambda x: int(x))
    
    for idx, sid in enumerate(sorted_ids):
        # 提取当前切片数据
        slice_data = subset_adata[subset_adata.obs['slice_id'] == sid].copy()
        
        # 判断是否保留高分辨率
        # idx 是选中切片的序号 (0, 1, 2...)
        if ((idx+1) % high_res_interval_m) == 0:
            # High Resolution
            slice_data.obs['resolution_type'] = 'high_res'
            slice_data.obs['bin_factor'] = 1.0
            processed_adatas.append(slice_data)
        else:
            # Low Resolution
            # logging.info(f"Degrading slice {sid} (idx {idx}) with factor {bin_factor}")
            low_res_data = degrade_resolution(slice_data, bin_factor=bin_factor)
            processed_adatas.append(low_res_data)
            
    # 3. 合并
    if not processed_adatas:
        return subset_adata # fallback
        
    mixed_adata = anndata.concat(processed_adatas, join='outer')
    
    # 恢复 var 信息 (concat 可能会丢失 var)
    mixed_adata.var = adata_full.var.copy()
    
    return mixed_adata