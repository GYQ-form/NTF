import numpy as np
import pandas as pd
import anndata
from scipy.stats import nbinom
from typing import Literal, Dict, Optional, List

# Mouse brain data used to generate the simulated data can be found at Zenodo: https://doi.org/10.5281/zenodo.19590994

def simulate_gene_expression(
    adata: anndata.AnnData,
    domain_key: str,
    n_genes: int = 50,
    distribution: Literal['NB', 'Poisson'] = 'NB',
    diff_genes_ratio: float = 0.9,
    base_mean: float = 10.0,
    dispersion: float = 1.0,  # used for NB only; corresponds to the n (size) parameter
    pattern_strength: float = 1.0,  # controls spatial gradient strength (0 = no gradient, 1 = 0 to 2x mean)
    active_domain_prob: float = 0.5,  # fraction of domains in which differential genes are active
    seed: int = 42
) -> anndata.AnnData:
    """
    Generate simulated gene expression with complex spatial patterns.
    
    Features:
    1. Non-housekeeping genes are expressed only in a subset of domains; zero elsewhere.
    2. The same gene can have different spatial patterns across different active domains.
    """
    np.random.seed(seed)
    n_cells = adata.shape[0]
    domains = adata.obs[domain_key].unique()
    n_domains = len(domains)
    
    # 1. Initialise expression matrix (all zeros)
    X = np.zeros((n_cells, n_genes), dtype=np.float32)
    
    # 2. Separate housekeeping genes from differential genes
    n_diff = int(n_genes * diff_genes_ratio)
    # Indices 0 ~ n_diff-1 are differential genes; the rest are housekeeping
    diff_gene_indices = np.arange(n_diff)
    hk_gene_indices = np.arange(n_diff, n_genes)
    
    gene_names = ([f'Gene_{i}_Diff' for i in range(n_diff)] + 
                  [f'Gene_{i}_HK' for i in range(n_diff, n_genes)])
    
    # Ground-truth record: domain -> gene_idx -> pattern_name
    gt_patterns = {d: {} for d in domains}
    
    # Available spatial pattern types:
    # constant: uniform expression
    # linear_x/y/z_inc: monotonically increasing along axis
    # linear_x/y/z_dec: monotonically decreasing along axis
    # radial_out: low at centre, high at periphery
    # radial_in: high at centre, low at periphery
    pattern_types = [
        'constant', 
        'linear_x_inc', 'linear_y_inc', 'linear_z_inc',
        'linear_x_dec', 'linear_y_dec', 'linear_z_dec',
        'radial_out', 'radial_in'
    ]
    
    # 3. Process each domain independently (local coordinates per domain)
    obs_domains = adata.obs[domain_key].values
    coords_full = adata.obsm['spatial']
    
    for d in domains:
        mask = (obs_domains == d)
        if np.sum(mask) == 0: continue
        
        domain_coords = coords_full[mask]
        n_domain_cells = domain_coords.shape[0]
        
        # --- A. Compute locally normalised coordinates (0~1) ---
        c_min = domain_coords.min(axis=0)
        c_max = domain_coords.max(axis=0)
        c_range = c_max - c_min
        c_range[c_range == 0] = 1.0  # avoid division by zero
        
        norm_xyz = (domain_coords - c_min) / c_range  # shape (N, 3)
        norm_x, norm_y, norm_z = norm_xyz[:, 0], norm_xyz[:, 1], norm_xyz[:, 2]
        
        # Normalised radial distance
        center = domain_coords.mean(axis=0)
        dists = np.linalg.norm(domain_coords - center, axis=1)
        max_dist = dists.max() if dists.max() > 0 else 1.0
        norm_r = dists / max_dist
        
        # --- B. Determine active genes for this domain ---
        # Housekeeping genes: always active
        # Differential genes: randomly active with probability active_domain_prob
        is_active_diff = np.random.rand(n_diff) < active_domain_prob
        active_diff_indices = diff_gene_indices[is_active_diff]
        
        active_genes_in_this_domain = np.concatenate([active_diff_indices, hk_gene_indices])
        
        # --- C. Assign spatial patterns to active genes ---
        domain_mu = np.zeros((n_domain_cells, n_genes), dtype=np.float32)
        
        # Randomly assign a pattern to each active differential gene
        diff_patterns_ids = np.random.choice(len(pattern_types), size=len(active_diff_indices))
        
        # Build pattern multiplier vectors; range: [1 - strength, 1 + strength]
        low = max(0.0, 1.0 - pattern_strength)
        high = 1.0 + pattern_strength
        
        vectors = {}
        vectors['constant'] = np.ones(n_domain_cells)
        vectors['linear_x_inc'] = low + (high - low) * norm_x
        vectors['linear_y_inc'] = low + (high - low) * norm_y
        vectors['linear_z_inc'] = low + (high - low) * norm_z
        vectors['linear_x_dec'] = high - (high - low) * norm_x
        vectors['linear_y_dec'] = high - (high - low) * norm_y
        vectors['linear_z_dec'] = high - (high - low) * norm_z
        vectors['radial_out'] = low + (high - low) * norm_r   # low at centre, high at periphery
        vectors['radial_in']  = high - (high - low) * norm_r  # high at centre, low at periphery
        
        # --- D. Fill mean expression matrix ---
        
        # 1. Differential genes
        for idx, pattern_id in zip(active_diff_indices, diff_patterns_ids):
            pat_name = pattern_types[pattern_id]
            domain_mu[:, idx] = base_mean * vectors[pat_name]
            gt_patterns[d][gene_names[idx]] = pat_name
            
        # 2. Housekeeping genes — constant expression
        domain_mu[:, hk_gene_indices] = base_mean * vectors['constant'].reshape(-1, 1)
        for idx in hk_gene_indices:
            gt_patterns[d][gene_names[idx]] = 'constant'

        # --- E. Sample counts for active genes only ---
        active_mask_cols = active_genes_in_this_domain
        
        if len(active_mask_cols) > 0:
            mus = domain_mu[:, active_mask_cols]
            
            if distribution == 'Poisson':
                counts = np.random.poisson(mus)
            else:  # Negative Binomial
                # Scipy nbinom parametrisation:
                # mean = n * (1-p) / p  =>  p = n / (n + mean)
                n = 1.0 / dispersion 
                p = n / (n + mus + 1e-9)
                counts = nbinom.rvs(n=n, p=p)
            
            X[np.ix_(mask, active_mask_cols)] = counts

    # 4. Build output AnnData
    sim_adata = anndata.AnnData(X=X, obs=adata.obs.copy(), obsm=adata.obsm.copy())
    sim_adata.var_names = gene_names
    
    sim_adata.uns['simulation_params'] = {
        'distribution': distribution,
        'active_domain_prob': active_domain_prob,
        'pattern_strength': pattern_strength,
        'gene_patterns': gt_patterns
    }
    
    print(f"Simulation done. Active diff genes per domain (avg): {active_domain_prob*100:.1f}%")
    
    return sim_adata


def assign_slices(adata: anndata.AnnData, z_col_idx: int = 2, n_slices: int = 100) -> anndata.AnnData:
    """
    Discretise Z-axis coordinates into slice IDs.
    """
    z_coords = adata.obsm['spatial'][:, z_col_idx]
    z_min, z_max = z_coords.min(), z_coords.max()
    
    bins = np.linspace(z_min, z_max, n_slices + 1)
    slice_ids = np.digitize(z_coords, bins) - 1 
    slice_ids = np.clip(slice_ids, 0, n_slices - 1)
    
    adata.obs['slice_id'] = slice_ids.astype(str)
    adata.obs['z_numeric'] = z_coords
    return adata
