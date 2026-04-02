import anndata
import scipy.sparse as sp
import numpy as np
import scanpy as sc
import warnings
from skimage.metrics import structural_similarity as ssim

def normalize_adata(adata: anndata.AnnData, layer: str = None):
    """
    将 AnnData 中的基因表达量进行非零区域的 Min-Max 标准化。
    0 值保持为 0，非零值被线性映射到 0.1 ~ 1.0 之间。
    
    参数:
    - adata: AnnData 对象
    - layer: 指定要处理的 layer 名称，如果为 None，则默认处理 adata.X
    
    返回:
    - 处理后的 AnnData 对象 (就地修改)
    """
    # 确定要操作的矩阵
    matrix = adata.layers[layer] if layer else adata.X
    
    # 检查是否为稀疏矩阵
    is_sparse = sp.issparse(matrix)
    
    if is_sparse:
        # 为了高效地按列（基因）进行操作，将其转换为 CSC 格式
        # 操作 .data 属性可以完美避开所有的 0 值，极致节省内存
        mat_csc = matrix.tocsc()
        
        for i in range(mat_csc.shape[1]):
            # 获取第 i 列所有的非零数据
            start_idx = mat_csc.indptr[i]
            end_idx = mat_csc.indptr[i+1]
            col_data = mat_csc.data[start_idx:end_idx]
            
            if len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()
                range_val = max_val - min_val
                
                if range_val > 0:
                    # 线性映射到 0.1 ~ 1.0
                    mat_csc.data[start_idx:end_idx] = 0.1 + 0.9 * ((col_data - min_val) / range_val)
                else:
                    # 如果该基因所有非零细胞的表达量都完全一样，直接将其设为 1.0
                    mat_csc.data[start_idx:end_idx] = 1.0
                    
        # 存回原来的位置（通常 scanpy 习惯使用 CSR 格式，所以转回 CSR）
        result_matrix = mat_csc.tocsr()
        
    else:
        # 密集矩阵 (Dense np.ndarray) 处理逻辑
        eps = 1e-6
        is_nonzero = matrix > eps
        
        # 找到非零的最小值
        masked_expr = np.where(is_nonzero, matrix, np.inf)
        min_vals_nonzero = masked_expr.min(axis=0)
        min_vals_nonzero[np.isinf(min_vals_nonzero)] = 0.0  # 处理全零基因
        
        # 找到最大值并计算极差
        max_vals = matrix.max(axis=0)
        range_vals = max_vals - min_vals_nonzero
        range_vals[range_vals <= 0] = 1.0  # 防止除以 0
        
        result_matrix = np.zeros_like(matrix)
        
        # 计算映射
        scaled_expr = 0.1 + 0.9 * ((matrix - min_vals_nonzero) / range_vals)
        
        # 仅替换非零位置
        result_matrix[is_nonzero] = scaled_expr[is_nonzero]
        result_matrix = np.clip(result_matrix, 0.0, 1.0)
        
    # 将处理后的结果写回 adata
    if layer:
        adata.layers[layer] = result_matrix
    else:
        adata.X = result_matrix



def calculate_spatial_metrics(adata: sc.AnnData, prediction_layer: str = 'prediction', spatial_key: str = 'spatial'):
    """
    计算每个基因的RMSE和三维SSIM，并动态处理SSIM的窗口大小。

    参数:
    - adata: AnnData 对象。
    - prediction_layer: 存储预测值的层的名称。
    - spatial_key: 存储空间坐标的obsm键名。

    返回:
    - AnnData: 更新后的AnnData对象，在 .var 中添加了 'rmse' 和 'ssim_3d' 列。
    """
    # 确保 AnnData 对象包含所需数据
    if prediction_layer not in adata.layers:
        raise ValueError(f"错误: 在 .layers 中未找到预测层 '{prediction_layer}'。")
    if spatial_key not in adata.obsm:
        raise ValueError(f"错误: 在 .obsm 中未找到空间坐标 '{spatial_key}'。")
    if adata.obsm[spatial_key].shape[1] != 3:
        raise ValueError(f"错误: obsm['{spatial_key}'] 中的坐标应为三维。")

    # --- 1. 计算每个基因的 RMSE ---
    true_expr = adata.X
    pred_expr = adata.layers[prediction_layer]
    
    if not isinstance(true_expr, np.ndarray):
        true_expr = true_expr.toarray()
    if not isinstance(pred_expr, np.ndarray):
        pred_expr = pred_expr.toarray()

    rmse_per_gene = np.sqrt(np.mean((true_expr - pred_expr)**2, axis=0))
    adata.var['rmse'] = rmse_per_gene
    print("已计算所有基因的RMSE。")

    # --- 2. 计算每个基因的三维 SSIM ---
    coords = adata.obsm[spatial_key].astype(int)
    grid_dims = coords.max(axis=0) + 1
    
    # --- 新增：动态确定SSIM的win_size ---
    min_dim = min(grid_dims)
    
    # win_size必须是奇数且小于等于最小维度
    if min_dim % 2 == 0:
        win_size = min_dim - 1
    else:
        win_size = min_dim
        
    # 如果win_size太小，SSIM没有意义，无法计算
    if win_size < 3:
        warnings.warn(
            f"空间网格的最小维度是 {min_dim}，太小而无法计算有意义的SSIM（需要至少为3）。"
            f"将跳过所有基因的SSIM计算，并将结果设置为 NaN。",
            UserWarning
        )
        adata.var['ssim_3d'] = np.nan
        return
    
    print(f"空间网格维度为: {grid_dims}。将使用 win_size={win_size} 进行SSIM计算。")
    
    ssim_scores = []
    
    for i, gene_name in enumerate(adata.var_names):
        true_volume = np.zeros(grid_dims)
        pred_volume = np.zeros(grid_dims)

        gene_true_expr = true_expr[:, i]
        gene_pred_expr = pred_expr[:, i]

        x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]
        true_volume[x_coords, y_coords, z_coords] = gene_true_expr
        pred_volume[x_coords, y_coords, z_coords] = gene_pred_expr
        
        data_range = max(true_volume.max(), pred_volume.max()) - min(true_volume.min(), pred_volume.min())
        
        if data_range == 0:
            current_ssim = 1.0
        else:
            # 使用动态计算的win_size
            current_ssim = ssim(true_volume, pred_volume, data_range=data_range, win_size=win_size)
        
        ssim_scores.append(current_ssim)
        
        if (i + 1) % 10 == 0 or (i + 1) == adata.n_vars:
             print(f"已处理 {i + 1}/{adata.n_vars} 个基因的SSIM计算...")

    adata.var['ssim_3d'] = ssim_scores
    print("已计算所有基因的三维SSIM。")