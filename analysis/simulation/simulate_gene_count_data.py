"""Generate simple-pattern simulation data with a parameterized gene count.

Same generator logic as analysis/simulate_data.py (the scale_test generator), with
three differences:
  1. --n_genes sets the gene count directly; the 5 patterns are assigned cyclically
     (gene i -> patterns[i % 5]) instead of genes_per_pattern blocks.
  2. --output sets the output path (no hardcoded directory).
  3. Generation is chunked over genes (--chunk_genes) so wide matrices do not
     blow up peak memory; per-stage RNG streams are derived from SeedSequence
     spawning, so results do not depend on the chunk size.

Batch effects and noise are only added where true expression values are non-zero,
identical to the original generator.
"""
import argparse
import json
import logging
import os
import time

import anndata
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PATTERNS = [
    "global_x_gradient",
    "global_y_gradient",
    "global_center_radial_out",
    "cluster_opposing_x_gradient",
    "cluster_opposing_y_gradient",
]


def generate_coords(rng, n_obs, n_clusters):
    cluster_centers = rng.uniform(low=0, high=80, size=(n_clusters, 3))
    obs_per_cluster = np.full(n_clusters, n_obs // n_clusters)
    obs_per_cluster[0] += n_obs % n_clusters

    coords_list, cluster_ids_list = [], []
    for i in range(n_clusters):
        cluster_size = rng.uniform(low=4, high=8)
        coords_list.append(rng.standard_normal((obs_per_cluster[i], 3)) * cluster_size + cluster_centers[i])
        cluster_ids_list.extend([f'cluster_{i}'] * obs_per_cluster[i])

    coords = np.vstack(coords_list)
    cluster_ids = np.array(cluster_ids_list)
    shuffle_idx = rng.permutation(n_obs)
    return coords[shuffle_idx], cluster_ids[shuffle_idx], cluster_centers


def truth_column(rng, pattern_name, cluster_ids, coords, coords_norm, dist_norm_global, cluster_centers, n_clusters):
    n_obs = coords.shape[0]
    col = np.zeros(n_obs, dtype=np.float32)

    n_expressing = rng.integers(1, max(2, n_clusters // 2 + 1))
    expressing = rng.choice(np.arange(n_clusters), size=n_expressing, replace=False)
    expressing_mask = np.isin(cluster_ids, [f'cluster_{i}' for i in expressing])

    if pattern_name == "global_x_gradient":
        col[expressing_mask] = coords_norm[expressing_mask, 0] * 2.0
    elif pattern_name == "global_y_gradient":
        col[expressing_mask] = coords_norm[expressing_mask, 1] * 2.0
    elif pattern_name == "global_center_radial_out":
        col[expressing_mask] = dist_norm_global[expressing_mask] * 2.0
    elif pattern_name.startswith("cluster"):
        permuted = rng.permutation(expressing)
        n_positive = len(permuted) // 2
        positive, negative = permuted[:n_positive], permuted[n_positive:]
        axis = {"cluster_opposing_x_gradient": 0,
                "cluster_opposing_y_gradient": 1,
                "cluster_opposing_z_gradient": 2}.get(pattern_name)
        for i in positive:
            mask = cluster_ids == f'cluster_{i}'
            if axis is not None:
                col[mask] = coords_norm[mask, axis] * 2.0
            else:
                d = np.linalg.norm(coords[mask] - cluster_centers[i], axis=1)
                if d.max() > 0:
                    col[mask] = (d / d.max()) * 2.0
        for i in negative:
            mask = cluster_ids == f'cluster_{i}'
            if axis is not None:
                col[mask] = (1 - coords_norm[mask, axis]) * 2.0
            else:
                d = np.linalg.norm(coords[mask] - cluster_centers[i], axis=1)
                if d.max() > 0:
                    col[mask] = (1 - d / d.max()) * 2.0
    else:
        raise ValueError(f"unknown pattern: {pattern_name}")
    return col


def main():
    parser = argparse.ArgumentParser(description="Generate simple-pattern simulation data with arbitrary gene count")
    parser.add_argument("--n_slices", type=int, default=100)
    parser.add_argument("--n_genes", type=int, required=True)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--batch_effect", type=float, default=0.2)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_genes", type=int, default=500)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t0 = time.time()
    n_obs = args.n_slices * 10000
    n_genes = args.n_genes
    ss = np.random.SeedSequence(args.seed)
    ss_coords, ss_split, ss_gene, ss_batch, ss_noise = ss.spawn(5)

    logging.info(f"Generating {n_obs} cells x {n_genes} genes, {args.n_slices} slices, seed={args.seed}")

    coords, cluster_ids, cluster_centers = generate_coords(np.random.default_rng(ss_coords), n_obs, args.n_clusters)
    coords_norm = (coords - coords.min(0)) / (coords.max(0) - coords.min(0) + 1e-6)
    dist_global = np.linalg.norm(coords - coords.mean(0), axis=1)
    dist_norm_global = dist_global / (dist_global.max() + 1e-6)

    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    slice_bins = np.linspace(z_min, z_max, args.n_slices + 1)
    slice_bins[0] -= 1e-6
    slice_bins[-1] += 1e-6
    slice_ids = np.array([f'slice_{i}' for i in np.digitize(coords[:, 2], bins=slice_bins)])
    unique_slice_ids = np.unique(slice_ids)

    shuffled = np.random.default_rng(ss_split).permutation(unique_slice_ids)
    test_slices = set(shuffled[: len(shuffled) // 2])
    dataset_labels = np.where(np.isin(slice_ids, list(test_slices)), 'test', 'train')

    pattern_names = [PATTERNS[i % len(PATTERNS)] for i in range(n_genes)]
    gene_rngs = [np.random.default_rng(s) for s in ss_gene.spawn(n_genes)]

    X_truth = np.zeros((n_obs, n_genes), dtype=np.float32)
    X_observed = np.zeros((n_obs, n_genes), dtype=np.float32)
    chunk_starts = list(range(0, n_genes, args.chunk_genes))
    batch_rngs = [np.random.default_rng(s) for s in ss_batch.spawn(len(unique_slice_ids) * len(chunk_starts))]
    noise_rngs = [np.random.default_rng(s) for s in ss_noise.spawn(len(chunk_starts))]

    for ci, g0 in enumerate(chunk_starts):
        g1 = min(g0 + args.chunk_genes, n_genes)
        width = g1 - g0
        logging.info(f"gene chunk [{g0}, {g1})")
        for g in range(g0, g1):
            X_truth[:, g] = truth_column(
                gene_rngs[g], pattern_names[g], cluster_ids, coords, coords_norm,
                dist_norm_global, cluster_centers, args.n_clusters,
            )

        truth_chunk = X_truth[:, g0:g1]
        obs_chunk = truth_chunk.copy()
        for si, s_id in enumerate(unique_slice_ids):
            rng = batch_rngs[si * len(chunk_starts) + ci]
            effect = rng.standard_normal(width).astype(np.float32) * args.batch_effect
            mask = slice_ids == s_id
            sub = obs_chunk[mask, :]
            nz = sub > 0
            sub[nz] += np.broadcast_to(effect, sub.shape)[nz]
            obs_chunk[mask, :] = sub

        noise = noise_rngs[ci].standard_normal((n_obs, width)).astype(np.float32) * args.noise
        nz = obs_chunk > 0
        obs_chunk[nz] += noise[nz]
        obs_chunk[obs_chunk < 0] = 0
        X_observed[:, g0:g1] = obs_chunk

    zero_frac_truth = float((X_truth == 0).mean())
    zero_frac_obs = float((X_observed == 0).mean())
    logging.info(f"zero fraction: truth={zero_frac_truth:.4f}, observed={zero_frac_obs:.4f}")

    obs_df = pd.DataFrame({
        'slice_id': pd.Categorical(slice_ids),
        'cluster_id': pd.Categorical(cluster_ids),
        'group': pd.Categorical(dataset_labels),
    }, index=[f'cell_{i}' for i in range(n_obs)])
    var_df = pd.DataFrame({'pattern_name': pattern_names}, index=[f'gene_{i}' for i in range(n_genes)])

    adata = anndata.AnnData(X=X_observed, obs=obs_df, var=var_df,
                            obsm={'spatial': coords}, layers={'truth': X_truth})

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"writing {args.output} (gzip)")
    adata.write_h5ad(args.output, compression='gzip', compression_opts=4)

    meta = {
        'script': os.path.abspath(__file__),
        'n_obs': int(n_obs), 'n_genes': int(n_genes), 'n_slices': int(args.n_slices),
        'n_clusters': int(args.n_clusters), 'batch_effect': args.batch_effect,
        'noise': args.noise, 'seed': args.seed, 'chunk_genes': args.chunk_genes,
        'patterns': PATTERNS,
        'pattern_assignment': 'gene i -> patterns[i % 5]',
        'zero_fraction_truth': zero_frac_truth,
        'zero_fraction_observed': zero_frac_obs,
        'n_train_slices': int(len(unique_slice_ids) - len(test_slices)),
        'n_test_slices': int(len(test_slices)),
        'output': os.path.abspath(args.output),
        'output_bytes': os.path.getsize(args.output),
        'elapsed_seconds': time.time() - t0,
    }
    meta_path = args.output + '.genmeta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logging.info(f"done in {meta['elapsed_seconds']:.1f}s, metadata -> {meta_path}")


if __name__ == '__main__':
    main()
