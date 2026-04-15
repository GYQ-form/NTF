import anndata
import scipy.sparse as sp
import numpy as np
import scanpy as sc
from skimage.metrics import structural_similarity as ssim

def normalize_adata(adata: anndata.AnnData, layer: str = None):
    """
    Perform non-zero Min-Max normalisation on gene expression in an AnnData object.
    Zero values are kept as 0; non-zero values are linearly mapped to the range 0.1–1.0.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to normalise.
    layer : str, optional
        Name of the layer to process. If None, adata.X is used.

    Returns
    -------
    AnnData
        The AnnData object modified in-place.
    """
    # Determine the matrix to operate on
    matrix = adata.layers[layer] if layer else adata.X
    
    # Check whether the matrix is sparse
    is_sparse = sp.issparse(matrix)
    
    if is_sparse:
        # Convert to CSC for efficient column-wise operations;
        # operating on .data avoids touching stored zeros and minimises memory usage
        mat_csc = matrix.tocsc()
        
        for i in range(mat_csc.shape[1]):
            # Retrieve all non-zero values in column i
            start_idx = mat_csc.indptr[i]
            end_idx = mat_csc.indptr[i+1]
            col_data = mat_csc.data[start_idx:end_idx]
            
            if len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()
                range_val = max_val - min_val
                
                if range_val > 0:
                    # Linearly map to 0.1–1.0
                    mat_csc.data[start_idx:end_idx] = 0.1 + 0.9 * ((col_data - min_val) / range_val)
                else:
                    # All non-zero cells have identical expression; set to 1.0
                    mat_csc.data[start_idx:end_idx] = 1.0
                    
        # Write back; scanpy typically expects CSR format
        result_matrix = mat_csc.tocsr()
        
    else:
        # Dense (np.ndarray) processing
        eps = 1e-6
        is_nonzero = matrix > eps
        
        # Find the non-zero minimum per gene
        masked_expr = np.where(is_nonzero, matrix, np.inf)
        min_vals_nonzero = masked_expr.min(axis=0)
        min_vals_nonzero[np.isinf(min_vals_nonzero)] = 0.0  # Handle all-zero genes
        
        # Find the maximum per gene and compute the range
        max_vals = matrix.max(axis=0)
        
        # 先记录下哪些基因的 max 和 min 是相等的
        range_vals = max_vals - min_vals_nonzero
        is_identical_expr = range_vals <= 0
        
        range_vals[is_identical_expr] = 1.0  # Avoid division by zero
        
        result_matrix = np.zeros_like(matrix)
        
        # Compute the linear mapping
        scaled_expr = 0.1 + 0.9 * ((matrix - min_vals_nonzero) / range_vals)
        
        # 对于 max==min 的基因，将其映射值强制设为 1.0（与 Sparse 逻辑保持一致）
        scaled_expr[:, is_identical_expr] = 1.0
        
        # Replace only the non-zero positions
        result_matrix[is_nonzero] = scaled_expr[is_nonzero]
        result_matrix = np.clip(result_matrix, 0.0, 1.0)
        
    # Write the processed result back to adata
    if layer:
        adata.layers[layer] = result_matrix
    else:
        adata.X = result_matrix



def calculate_spatial_metrics(adata: sc.AnnData, prediction_layer: str = 'prediction', spatial_key: str = 'spatial'):
    """
    Compute per-gene RMSE and 3D SSIM, dynamically adapting the SSIM window size.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing ground-truth expression in adata.X and predictions
        in adata.layers[prediction_layer].
    prediction_layer : str
        Name of the layer holding predicted expression values.
    spatial_key : str
        Key in adata.obsm holding 3D spatial coordinates.

    Returns
    -------
    AnnData
        Updated AnnData with 'rmse' and 'ssim_3d' columns added to adata.var.
    """
    # Validate inputs
    if prediction_layer not in adata.layers:
        raise ValueError(f"Prediction layer '{prediction_layer}' not found in .layers.")
    if spatial_key not in adata.obsm:
        raise ValueError(f"Spatial key '{spatial_key}' not found in .obsm.")
    if adata.obsm[spatial_key].shape[1] != 3:
        raise ValueError(f"obsm['{spatial_key}'] should contain 3D coordinates.")

    # --- 1. Compute per-gene RMSE ---
    true_expr = adata.X
    pred_expr = adata.layers[prediction_layer]
    
    if not isinstance(true_expr, np.ndarray):
        true_expr = true_expr.toarray()
    if not isinstance(pred_expr, np.ndarray):
        pred_expr = pred_expr.toarray()

    rmse_per_gene = np.sqrt(np.mean((true_expr - pred_expr)**2, axis=0))
    adata.var['rmse'] = rmse_per_gene
    print("RMSE computed for all genes.")

    # --- 2. Compute per-gene 3D SSIM ---
    coords_original = adata.obsm[spatial_key]

    # Scale coordinates to fit within a reasonable range for SSIM computation
    # Ensures min dimension >= target_min_dim and max dimension <= target_max_dim
    target_min_dim = 10  # Minimum grid dimension for meaningful SSIM
    target_max_dim = 64  # Maximum grid dimension to avoid excessive computation time
    coords_max = coords_original.max(axis=0)
    coords_min = coords_original.min(axis=0)
    original_dims = coords_max - coords_min + 1
    max_dim = original_dims.max()
    min_dim = original_dims.min()

    # Determine appropriate scaling factor
    if max_dim > target_max_dim:
        # Scale down to fit within max target
        scale_factor = target_max_dim / max_dim
    elif min_dim < target_min_dim:
        # Scale up to meet minimum dimension requirement
        scale_factor = target_min_dim / min_dim
    else:
        scale_factor = 1.0

    coords_scaled = (coords_original - coords_min) * scale_factor
    coords = coords_scaled.astype(int)
    grid_dims = coords.max(axis=0) + 1

    # Re-check and adjust if any dimension became too small due to integer rounding
    actual_min_dim = grid_dims.min()
    if actual_min_dim < 3:
        # Force scale up to ensure at least 3 in minimum dimension
        scale_factor = 3.0 / min_dim
        coords_scaled = (coords_original - coords_min) * scale_factor
        coords = coords_scaled.astype(int)
        grid_dims = coords.max(axis=0) + 1

    # Dynamically determine SSIM window size (must be odd and <= smallest grid dimension)
    min_dim = grid_dims.min()
    if min_dim % 2 == 0:
        win_size = min_dim - 1
    else:
        win_size = min_dim

    # Final safety check
    if win_size < 3:
        win_size = 3

    print(f"Original spatial dimensions: {original_dims}. "
          f"Scaled spatial dimensions: {grid_dims} (scale_factor={scale_factor:.3f}). "
          f"Using win_size={win_size} for SSIM.")

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
            current_ssim = ssim(true_volume, pred_volume, data_range=data_range, win_size=win_size)

        ssim_scores.append(current_ssim)

        if (i + 1) % 10 == 0 or (i + 1) == adata.n_vars:
             print(f"SSIM computed for {i + 1}/{adata.n_vars} genes...")

    adata.var['ssim_3d'] = ssim_scores
    print("3D SSIM computed for all genes.")
