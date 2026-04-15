import anndata
import numpy as np
import pandas as pd
import logging
import argparse
import os
import sys

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
    Create a simulated 3D spatial transcriptomics AnnData object.
    Batch effects and noise are only added where true expression values are non-zero.
    """
    
    # --- 0. Define spatial expression patterns ---
    patterns = [
        "global_x_gradient",
        "global_y_gradient",
        "global_center_radial_out",
        "cluster_opposing_x_gradient",
        "cluster_opposing_y_gradient",
        "cluster_opposing_z_gradient",
        "cluster_opposing_radial",
    ]
    # Ensure enough patterns are available
    if len(patterns) < 5:
         used_patterns = np.random.choice(patterns, 5, replace=True)
    else:
         used_patterns = np.choose(np.arange(5), patterns)
         
    n_genes = len(used_patterns) * genes_per_pattern
    
    logging.info(
        f"Creating simulated AnnData: {n_obs} cells, {n_genes} genes, {n_slices} slices, "
        f"{n_clusters} cell clusters."
    )

    # --- 1. Generate 3D cell coordinates and cluster assignments ---
    logging.info("Step 1: Generating cell coordinates and clusters...")
    cluster_centers = np.random.uniform(low=0, high=80, size=(n_clusters, 3))
    obs_per_cluster = np.full(n_clusters, n_obs // n_clusters)
    obs_per_cluster[0] += n_obs % n_clusters
    
    coords_list = []
    cluster_ids_list = []
    for i in range(n_clusters):
        cluster_size = np.random.uniform(low=4, high=8)
        c_coords = np.random.randn(obs_per_cluster[i], 3) * cluster_size + cluster_centers[i]
        coords_list.append(c_coords)
        cluster_ids_list.extend([f'cluster_{i}'] * obs_per_cluster[i])

    coords = np.vstack(coords_list)
    shuffle_idx = np.random.permutation(n_obs)
    coords = coords[shuffle_idx]
    cluster_ids = np.array(cluster_ids_list)[shuffle_idx]

    # --- 2. Create ground-truth gene expression from spatial patterns ---
    logging.info("Step 2: Generating ground-truth gene expression with high dropout characteristics...")
    X_truth = np.zeros((n_obs, n_genes), dtype=np.float32)
    var_pattern_names = []
    gene_idx = 0

    # Normalised coordinates for pattern computation
    coords_norm = (coords - coords.min(0)) / (coords.max(0) - coords.min(0) + 1e-6)
    dist_from_global_center = np.linalg.norm(coords - coords.mean(0), axis=1)
    dist_norm_global = (dist_from_global_center / (dist_from_global_center.max() + 1e-6))

    for pattern_name in used_patterns:
        for _ in range(genes_per_pattern):
            if gene_idx >= n_genes: break
            
            # Randomly select a subset of clusters that express this gene
            n_expressing_clusters = np.random.randint(1, max(2, n_clusters // 2 + 1))
            expressing_cluster_indices = np.random.choice(
                np.arange(n_clusters), n_expressing_clusters, replace=False
            )
            
            expressing_cell_mask = np.isin(
                cluster_ids, [f'cluster_{i}' for i in expressing_cluster_indices]
            )

            # Apply pattern logic
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

    # --- 3. Assign slice IDs based on Z-axis coordinate ---
    logging.info("Step 3: Assigning slice IDs from Z-axis coordinates...")
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    slice_bins = np.linspace(z_min, z_max, n_slices + 1)
    slice_bins[0] -= 1e-6; slice_bins[-1] += 1e-6
    slice_indices = np.digitize(coords[:, 2], bins=slice_bins)
    slice_ids = np.array([f'slice_{i}' for i in slice_indices])

    # --- 4. Simulate batch effects and noise (only at non-zero positions) ---
    logging.info("Step 4: Simulating per-slice batch effects and noise (non-zero positions only)...")
    X_observed = np.copy(X_truth)
    
    unique_slice_ids = np.unique(slice_ids)
    
    # 4.1 Add per-slice batch effects (only where X_truth > 0)
    for s_id in unique_slice_ids:
        slice_mask = (slice_ids == s_id)
        
        # Fixed per-gene offset for this slice (batch effect)
        slice_gene_effect = np.random.normal(0, batch_effect_std, n_genes)
        
        sub_matrix = X_observed[slice_mask, :]
        effect_matrix = np.tile(slice_gene_effect, (sub_matrix.shape[0], 1))
        
        # Only add effect at non-zero positions
        nonzero_mask = sub_matrix > 0
        sub_matrix[nonzero_mask] += effect_matrix[nonzero_mask]
        
        X_observed[slice_mask, :] = sub_matrix

    # 4.2 Add Gaussian noise (only where expression is non-zero)
    noise_matrix = np.random.randn(n_obs, n_genes) * noise_std
    global_nonzero_mask = X_observed > 0
    X_observed[global_nonzero_mask] += noise_matrix[global_nonzero_mask]
    
    # Ensure non-negative values
    X_observed[X_observed < 0] = 0

    # --- 5. Split into train/test sets by slice ---
    logging.info("Step 5: Randomly splitting train/test slices and assembling AnnData...")
    
    shuffled_slice_ids = np.random.permutation(unique_slice_ids)
    split_idx = len(shuffled_slice_ids) // 2
    
    test_slices = set(shuffled_slice_ids[:split_idx])
    train_slices = set(shuffled_slice_ids[split_idx:])
    
    dataset_labels = []
    for s_id in slice_ids:
        if s_id in train_slices:
            dataset_labels.append('train')
        else:
            dataset_labels.append('test')
            
    logging.info(f"Train slices: {len(train_slices)}, Test slices: {len(test_slices)}")

    # --- 6. Assemble AnnData object ---
    obs_df = pd.DataFrame({
        'slice_id': pd.Categorical(slice_ids),
        'cluster_id': pd.Categorical(cluster_ids),
        'group': pd.Categorical(dataset_labels)
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
    
    logging.info("Simulation complete.")
    return adata

def parse_args():
    parser = argparse.ArgumentParser(description="Generate simulated 3D spatial transcriptomics data")

    parser.add_argument("--n_slices", type=int, default=100, help="Number of Z-axis slices (default: 100)")
    parser.add_argument("--n_clusters", type=int, default=10, help="Number of cell clusters (default: 10)")
    parser.add_argument("--genes_per_pattern", type=int, default=20, help="Number of genes per expression pattern (default: 20)")
    parser.add_argument("--batch_effect", type=float, default=0.2, help="Standard deviation of per-slice batch effects (default: 0.2)")
    parser.add_argument("--noise", type=float, default=0.1, help="Standard deviation of Gaussian noise (default: 0.1)")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    args.output = f'/home/gongyuqiao/ur_annotation/NTF/mytrain/data/simulation/scale_test/{args.n_slices}w.h5ad'
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logging.info(f"Created output directory: {out_dir}")

    adata = create_mock_anndata(
        n_obs=args.n_slices*10000,
        n_slices=args.n_slices,
        n_clusters=args.n_clusters,
        genes_per_pattern=args.genes_per_pattern,
        batch_effect_std=args.batch_effect,
        noise_std=args.noise
    )
    
    logging.info("Data generation complete. Summary:")
    print(adata)
    print("\nObs group distribution:")
    print(adata.obs['group'].value_counts())
    
    logging.info(f"Saving to: {args.output}")
    adata.write_h5ad(args.output, compression='gzip')
    logging.info("Done!")
