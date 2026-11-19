import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from scipy.spatial import cKDTree

def calculate_avg_cell_distance(coords: np.ndarray) -> float:
    """Compute the average nearest-neighbour distance between cells."""
    if len(coords) < 2:
        return 1.0
    tree = cKDTree(coords)
    # Query k=2 neighbours; the first is the point itself, the second is the nearest neighbour
    dists, _ = tree.query(coords, k=2) 
    return np.mean(dists[:, 1])

def degrade_resolution(
    adata_slice: anndata.AnnData, 
    bin_factor: float = 2.0
) -> anndata.AnnData:
    """
    Reduce the spatial resolution of a single tissue slice.
    
    Args:
        adata_slice: AnnData object for a single slice.
        bin_factor: Grid cell size multiplier. 1.0 sets the grid size to approximately
                    the average inter-cell distance. 2.0 uses a grid whose edge length
                    is twice the average distance (roughly 4x area per bin).
    
    Returns:
        AnnData: Down-sampled data with fewer observations.
    """
    coords = adata_slice.obsm['spatial']
    
    if len(coords) == 0:
        return adata_slice
        
    # 1. Determine grid cell size
    avg_dist = calculate_avg_cell_distance(coords)
    grid_size = avg_dist * bin_factor
    
    # 2. Compute grid indices
    # Divide coordinates by grid_size and floor to obtain (ix, iy, iz) grid IDs.
    # Although slices are approximately 2D, we bin all three axes for generality.
    grid_indices = np.floor(coords / grid_size).astype(int)
    
    # 3. Aggregate cells within each grid bin
    df_coords = pd.DataFrame(grid_indices, columns=['gx', 'gy', 'gz'])
    df_coords['original_index'] = np.arange(len(coords))
    
    grouped = df_coords.groupby(['gx', 'gy', 'gz'])
    
    new_coords = []
    new_expr = []
    new_slice_ids = []
    
    raw_X = adata_slice.X
    if hasattr(raw_X, 'toarray'):
        raw_X = raw_X.toarray()
        
    raw_slice_ids = adata_slice.obs['slice_id'].values
    
    indices_groups = grouped.indices  # dict: grid_key -> array of original indices
    
    for grid_key, indices in indices_groups.items():
        # Coordinate: geometric centroid of cells in this bin
        cell_coords = coords[indices]
        centroid = np.mean(cell_coords, axis=0)
        new_coords.append(centroid)
        
        # Expression: mean across cells in the bin
        cell_expr = raw_X[indices]
        avg_expr = np.mean(cell_expr, axis=0)
        new_expr.append(avg_expr)
        
        # Slice ID: inherit from the first cell in the bin
        new_slice_ids.append(raw_slice_ids[indices[0]])
        
    if len(new_coords) == 0:
        return adata_slice[[]]  # return empty AnnData
        
    new_coords = np.array(new_coords)
    new_expr = np.array(new_expr)
    new_slice_ids = np.array(new_slice_ids)
    
    # 4. Build output AnnData
    new_adata = anndata.AnnData(X=new_expr)
    new_adata.obsm['spatial'] = new_coords
    new_adata.obs['slice_id'] = new_slice_ids
    new_adata.var_names = adata_slice.var_names
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
    Assemble a mixed-resolution dataset.
    
    Args:
        adata_full: Full AnnData object containing all slices.
        slice_interval_k: Slice sampling interval (keep 1 slice every k slices).
        high_res_interval_m: Among the selected slices, retain every m-th slice at
                             full resolution; the rest are degraded.
                             Example: m=1 means all selected slices are high-res;
                                      m=2 alternates high / low.
        bin_factor: Resolution degradation factor for low-res slices.
    """
    # 1. Select slices
    all_slices = np.sort(adata_full.obs['slice_id'].unique().astype(int))
    selected_slice_ids = all_slices[::slice_interval_k].astype(str)
    
    subset_adata = adata_full[adata_full.obs['slice_id'].isin(selected_slice_ids)].copy()
    
    # 2. Process each selected slice
    processed_adatas = []
    
    sorted_ids = sorted(selected_slice_ids, key=lambda x: int(x))
    
    for idx, sid in enumerate(sorted_ids):
        slice_data = subset_adata[subset_adata.obs['slice_id'] == sid].copy()
        
        if ((idx + 1) % high_res_interval_m) == 0:
            # Keep at full resolution
            slice_data.obs['resolution_type'] = 'high_res'
            slice_data.obs['bin_factor'] = 1.0
            processed_adatas.append(slice_data)
        else:
            # Degrade to low resolution
            low_res_data = degrade_resolution(slice_data, bin_factor=bin_factor)
            processed_adatas.append(low_res_data)
            
    # 3. Concatenate all processed slices
    if not processed_adatas:
        return subset_adata  # fallback
        
    mixed_adata = anndata.concat(processed_adatas, join='outer')
    
    # Restore var metadata (may be lost during concat)
    mixed_adata.var = adata_full.var.copy()
    
    return mixed_adata
