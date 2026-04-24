import torch
import anndata
import numpy as np
import logging
import argparse
import os
import time
from scipy.spatial import cKDTree
from scipy.stats import mode
import pandas as pd
import warnings
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import tempfile
import h5py

from NTF.train import train
from NTF.sample import sample_points
from NTF.config import add_shared_args, process_args
import scanpy as sc
from NTF.helper_func import normalize_adata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _thread_sample_chunk(kdtree, min_bound, max_bound, chunk_size, threshold, seed):
    """Thread-safe worker for distance-field sampling.

    cKDTree.query() is a C-level call that releases the GIL,
    so threads achieve true parallelism without the overhead of fork().
    """
    rng = np.random.RandomState(seed)
    candidates = rng.uniform(low=min_bound, high=max_bound, size=(chunk_size, 3)).astype(np.float32)
    dists, _ = kdtree.query(candidates)
    return candidates[dists < threshold]


def sample_points_in_alpha_volume_parallel(raw_coords: np.ndarray, n_points: int,
                                          dist_tol_multiplier: float = 2.0,
                                          n_workers: int = 8,
                                          chunk_size: int = 10_000_000,
                                          low_memory: bool = False) -> np.ndarray:
    """Sample points inside tissue boundary using KD-Tree distance field (parallel version).

    Uses ThreadPoolExecutor instead of multiprocessing to avoid fork/memory issues.
    cKDTree.query() releases the GIL so threads achieve real parallelism.

    Parameters
    ----------
    raw_coords : np.ndarray
        Original spatial coordinates (n_cells, 3)
    n_points : int
        Target number of points to sample
    dist_tol_multiplier : float
        Multiplier for average inter-cell distance to define boundary (default 2.0)
    n_workers : int
        Number of parallel threads (default 8)
    chunk_size : int
        Number of candidates per chunk (default 10M)
    low_memory : bool
        If True, use sequential processing with smaller chunks

    Returns
    -------
    np.ndarray
        Array of sampled points (n_points, 3)
    """
    logging.info(f"Sampling {n_points:,} points using distance-field method ({n_workers} threads)...")

    # Build KD-Tree and compute threshold
    kdtree = cKDTree(raw_coords)
    distances, _ = kdtree.query(raw_coords, k=2)
    avg_dist = np.mean(distances[:, 1])
    threshold = avg_dist * dist_tol_multiplier

    logging.info(f"Average inter-cell distance: {avg_dist:.4f}, boundary threshold: {threshold:.4f}")

    min_bound = raw_coords.min(axis=0)
    max_bound = raw_coords.max(axis=0)

    bbox_volume = np.prod(max_bound - min_bound)
    if bbox_volume == 0:
        logging.error("Bounding box volume is zero; cannot sample.")
        return np.array([])

    # Estimate acceptance ratio by sampling
    test_candidates = np.random.uniform(low=min_bound, high=max_bound, size=(10000, 3)).astype(np.float32)
    test_dists, _ = kdtree.query(test_candidates)
    acceptance_ratio = np.mean(test_dists < threshold)

    if acceptance_ratio == 0:
        logging.error("No points fall within distance threshold. Try increasing --dist_tol.")
        return np.array([])

    logging.info(f"Estimated acceptance ratio: {acceptance_ratio:.3f}")

    if low_memory:
        logging.info("Low-memory mode: using sequential processing")
        return _sample_distfield_sequential(kdtree, n_points, threshold, chunk_size, min_bound, max_bound)

    # Estimate total candidates needed (with 30% buffer)
    total_candidates_needed = int(n_points / acceptance_ratio * 1.3)
    n_chunks = max(1, (total_candidates_needed + chunk_size - 1) // chunk_size)

    logging.info(f"Estimated {total_candidates_needed:,} candidates across {n_chunks} chunks")

    # Thread-parallel sampling — shared KD-Tree, no fork, no memory duplication
    sampled_points = []
    collected_count = 0

    with tqdm(total=n_points, desc="Sampling points", unit="pts", unit_scale=True) as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _thread_sample_chunk, kdtree, min_bound, max_bound, chunk_size, threshold, i * 12345
                ): i for i in range(n_chunks)
            }

            for future in as_completed(futures):
                result = future.result()
                if len(result) > 0:
                    sampled_points.append(result)
                    collected_count += len(result)
                    pbar.update(min(len(result), n_points - (collected_count - len(result))))

                if collected_count >= n_points:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

    if collected_count == 0:
        logging.error("No valid points sampled. Try increasing --dist_tol parameter.")
        return np.array([])

    all_points = np.vstack(sampled_points)

    if collected_count < n_points:
        warnings.warn(f"Only {collected_count:,} points sampled, fewer than target {n_points:,}.")

    final_points = all_points[:n_points]
    logging.info(f"Distance-field sampling obtained {len(final_points):,} points")
    return final_points


def _sample_distfield_sequential(kdtree, n_points, threshold, chunk_size, min_bound, max_bound):
    """Sequential fallback for low-memory mode."""
    sampled_points = []
    collected_count = 0
    max_iterations = 1000
    iteration = 0

    with tqdm(total=n_points, desc="Sampling (sequential)", unit="pts", unit_scale=True) as pbar:
        while collected_count < n_points and iteration < max_iterations:
            candidates = np.random.uniform(low=min_bound, high=max_bound, size=(chunk_size, 3)).astype(np.float32)
            dists, _ = kdtree.query(candidates)

            mask_inside = dists < threshold
            valid_points = candidates[mask_inside]

            if len(valid_points) > 0:
                sampled_points.append(valid_points)
                collected_count += len(valid_points)
                pbar.update(len(valid_points))

            iteration += 1

    if collected_count == 0:
        return np.array([])

    return np.vstack(sampled_points)[:n_points]


def get_args():
    shared_parser = add_shared_args()
    parser = argparse.ArgumentParser(
        description="Train model and generate high-resolution data using optimized distance-field sampling.",
        parents=[shared_parser]
    )

    task_group = parser.add_argument_group('Task Specific: Generation')
    task_group.add_argument('--n_generate', type=int, default=1000000, help='Number of new points to generate.')
    task_group.add_argument('--dist_tol', type=float, default=20.0,
                           help='Tolerance multiplier for boundary determination (default 20.0).')

    task_group.add_argument('--expr_filter_ratio', type=float, default=0.03,
                           help='Threshold ratio for filtering low-expression points (default 0.03). '
                                'Points with total expression < max_total_expression * ratio will be removed.')

    task_group.add_argument('--cat_keys', type=str, nargs='+', default=[],
                           help='Categorical variables for KNN voting.')
    task_group.add_argument('--k_neighbors', type=int, default=5,
                           help='Number of nearest neighbors for voting.')

    opt_group = parser.add_argument_group('Optimization Parameters')
    opt_group.add_argument('--n_workers', type=int, default=8,
                          help='Number of parallel workers for candidate generation (default 8).')
    opt_group.add_argument('--candidate_chunk_size', type=int, default=5_000_000,
                          help='Candidates per chunk (default 5M). Larger = faster but more memory.')
    opt_group.add_argument('--inference_chunk_size', type=int, default=4194304,
                          help='Number of points per chunk for the outer inference loop (default 4194304). '
                               'Controls how many points are sent to sample_points() at a time.')
    opt_group.add_argument('--inference_batch_size', type=int, default=16384,
                          help='Batch size used inside sample_points() for GPU forward passes (default 16384).')
    opt_group.add_argument('--low_memory', action='store_true',
                          help='Use sequential processing with smaller chunks (slower but lower memory).')

    args = parser.parse_args()
    args = process_args(args)
    return args


def main():
    args = get_args()
    logging.info(f"Configuration loaded. Results will be saved to: {args.output_dir}")
    logging.info(f"Optimization: {args.n_workers} workers, {args.candidate_chunk_size:,} candidates/chunk, "
                f"inference chunk size {args.inference_chunk_size}, batch size {args.inference_batch_size}")

    # --- 1. Load and preprocess data ---
    logging.info(f"Loading data from {args.input_data}...")
    adata = anndata.read_h5ad(args.input_data)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)

    raw_coords = adata.obsm['spatial'].copy()
    raw_bbox_size = (raw_coords.max(0) - raw_coords.min(0)).max()
    spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
    adata.obsm['spatial'] *= spatial_scaling
    logging.info(f"Training on {adata.n_obs:,} cells. Spatial scaling factor: {spatial_scaling:.6f}")

    # --- 2. Sample coordinates BEFORE training (fork-safe: low memory footprint) ---
    t_start_sampling = time.time()
    new_raw_coords = sample_points_in_alpha_volume_parallel(
        raw_coords, args.n_generate,
        dist_tol_multiplier=args.dist_tol,
        n_workers=args.n_workers,
        chunk_size=args.candidate_chunk_size,
        low_memory=args.low_memory
    )
    sampling_time = time.time() - t_start_sampling

    if new_raw_coords.shape[0] == 0:
        return

    logging.info(f"Sampling completed in {sampling_time:.1f} seconds ({len(new_raw_coords)/sampling_time:.0f} points/sec)")

    # --- 3. Train model ---
    trained_NTF = train(adata, args)
    trained_inr = trained_NTF.inr

    # --- 4. Predict expression at new coordinates (memory-mapped) ---
    scaled_new_coords = new_raw_coords * spatial_scaling
    new_coords_tensor = torch.from_numpy(scaled_new_coords).float().to(args.device)

    n_points = len(new_coords_tensor)
    n_genes = len(adata.var_names)
    logging.info(f"Predicting at {n_points:,} points × {n_genes} genes "
                f"(chunk size {args.inference_chunk_size}, batch size {args.inference_batch_size})")
    logging.info(f"Dense matrix would be {n_points * n_genes * 4 / 1e9:.1f} GB — using memory-mapped file")

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = args.output_dir

    # Create memory-mapped arrays for expression (and optionally dropout)
    expr_mmap_path = os.path.join(tmp_dir, '_tmp_expr.mmap')
    expr_mmap = np.memmap(expr_mmap_path, dtype=np.float32, mode='w+', shape=(n_points, n_genes))

    has_dropout = not args.no_dropout
    dropout_mmap = None
    dropout_mmap_path = None
    if has_dropout:
        dropout_mmap_path = os.path.join(tmp_dir, '_tmp_dropout.mmap')
        dropout_mmap = np.memmap(dropout_mmap_path, dtype=np.float32, mode='w+', shape=(n_points, n_genes))

    # Also compute per-point total expression on-the-fly for later filtering
    total_expr = np.zeros(n_points, dtype=np.float32)

    t_start = time.time()
    infer_model_sec = 0.0
    infer_io_sec = 0.0

    with tqdm(total=n_points, desc="Inference", unit="pts", unit_scale=True) as pbar:
        for i in range(0, n_points, args.inference_chunk_size):
            end = min(i + args.inference_chunk_size, n_points)
            batch = new_coords_tensor[i:end]

            t_model = time.time()
            batch_results = sample_points(trained_inr, batch, batch_size=args.inference_batch_size)
            infer_model_sec += time.time() - t_model

            t_io = time.time()
            expr_np = batch_results["expression"].cpu().numpy()
            expr_mmap[i:end] = expr_np
            total_expr[i:end] = expr_np.sum(axis=1)

            if has_dropout and "dropout_prob" in batch_results:
                dropout_mmap[i:end] = batch_results["dropout_prob"].cpu().numpy()
            infer_io_sec += time.time() - t_io

            pbar.update(end - i)

    # Flush to disk
    t_io = time.time()
    expr_mmap.flush()
    if dropout_mmap is not None:
        dropout_mmap.flush()
    infer_io_sec += time.time() - t_io

    infer_total_sec = time.time() - t_start
    logging.info(f"Inference complete in {infer_total_sec:.1f}s "
                f"(model: {infer_model_sec:.1f}s, I/O: {infer_io_sec:.1f}s, "
                f"{n_points/infer_total_sec:.0f} pts/sec)")

    # Free GPU memory — model is no longer needed
    del trained_NTF, trained_inr, new_coords_tensor
    torch.cuda.empty_cache()

    # --- 5. Expression-based background filtering ---
    if args.expr_filter_ratio > 0:
        logging.info("Filtering low-expression background regions...")
        expr_threshold = np.max(total_expr) * args.expr_filter_ratio
        keep_mask = total_expr > expr_threshold
        keep_indices = np.where(keep_mask)[0]

        n_before = n_points
        n_kept = len(keep_indices)
        logging.info(f"Expression filtering: removed {n_before - n_kept:,} low-expression points, "
                    f"retained {n_kept:,} high-confidence tissue points")

        if n_kept == 0:
            logging.error("All points removed by expression threshold. Try lowering --expr_filter_ratio.")
            _cleanup_mmap(expr_mmap_path, dropout_mmap_path)
            return

        new_raw_coords = new_raw_coords[keep_indices]
    else:
        keep_indices = np.arange(n_points)
        n_kept = n_points

    del total_expr

    obs_df = pd.DataFrame(index=[f"new_cell_{i}" for i in range(n_kept)])

    # --- 6. KNN-based majority voting ---
    if args.cat_keys:
        n_retained_points = n_kept
        logging.info(f"Running KNN (k={args.k_neighbors}) majority voting for {n_retained_points:,} points...")

        tree = cKDTree(raw_coords)
        k = min(args.k_neighbors, len(raw_coords))
        knn_chunk_size = 2000000  # process 2M points per chunk

        voted_results = {key: np.zeros(n_retained_points, dtype=np.int32) for key in args.cat_keys}

        with tqdm(total=n_retained_points, desc="KNN voting", unit="pts", unit_scale=True) as pbar:
            for start_idx in range(0, n_retained_points, knn_chunk_size):
                end_idx = min(start_idx + knn_chunk_size, n_retained_points)
                chunk_coords = new_raw_coords[start_idx:end_idx]

                distances, indices = tree.query(chunk_coords, k=k)
                if k == 1:
                    indices = indices.reshape(-1, 1)

                for key in args.cat_keys:
                    if key not in adata.obs:
                        continue
                    cat_series = pd.Categorical(adata.obs[key])
                    codes = cat_series.codes
                    neighbor_codes = codes[indices]

                    try:
                        voted_modes, _ = mode(neighbor_codes, axis=1, keepdims=False)
                    except TypeError:
                        voted_modes, _ = mode(neighbor_codes, axis=1)

                    voted_results[key][start_idx:end_idx] = voted_modes.flatten()

                pbar.update(end_idx - start_idx)

        for key in args.cat_keys:
            if key not in adata.obs: continue
            cat_series = pd.Categorical(adata.obs[key])
            categories = cat_series.categories
            obs_df[key] = categories[voted_results[key]]
            obs_df[key] = obs_df[key].astype("category")

        logging.info(f"KNN categorical voting complete: {args.cat_keys}")

    # --- 7. Stream CSR directly to h5ad via h5py (single-pass, resizable datasets) ---
    output_path = os.path.join(args.output_dir, f'{int(args.n_generate/10000)}w_distfield_optimized.h5ad')
    logging.info("Streaming sparse CSR matrix to h5ad via h5py...")

    write_chunk_size = 2_000_000  # rows per chunk

    # Re-open memmap as read-only
    del expr_mmap
    expr_mmap = np.memmap(expr_mmap_path, dtype=np.float32, mode='r', shape=(n_points, n_genes))

    # Single-pass: use resizable h5py datasets to avoid a counting pass
    INIT_NNZ = 1_000_000_000  # initial allocation, will resize at the end

    with h5py.File(output_path, 'w') as f:
        x_group = f.create_group('X')
        x_group.attrs['encoding-type'] = 'csr_matrix'
        x_group.attrs['encoding-version'] = '0.1.0'
        x_group.attrs['shape'] = np.array([n_kept, n_genes], dtype=np.int64)

        ds_data = x_group.create_dataset('data', shape=(INIT_NNZ,), maxshape=(None,),
                                         dtype=np.float32, chunks=(min(INIT_NNZ, 1_000_000),),
                                         compression='lzf')
        ds_indices = x_group.create_dataset('indices', shape=(INIT_NNZ,), maxshape=(None,),
                                            dtype=np.int32, chunks=(min(INIT_NNZ, 1_000_000),),
                                            compression='lzf')
        ds_indptr = x_group.create_dataset('indptr', shape=(n_kept + 1,),
                                           dtype=np.int64, chunks=(min(n_kept + 1, 1_000_000),),
                                           compression='lzf')

        data_offset = 0
        current_capacity = INIT_NNZ
        ds_indptr[0] = 0

        with tqdm(total=n_kept, desc="Writing CSR", unit="rows", unit_scale=True) as pbar:
            for chunk_start in range(0, n_kept, write_chunk_size):
                chunk_end = min(chunk_start + write_chunk_size, n_kept)
                chunk_indices_arr = keep_indices[chunk_start:chunk_end]
                chunk_data = np.array(expr_mmap[chunk_indices_arr])

                chunk_csr = sp.csr_matrix(chunk_data)
                chunk_nnz = chunk_csr.nnz

                # Grow datasets if needed
                needed = data_offset + chunk_nnz
                if needed > current_capacity:
                    new_cap = max(needed, int(current_capacity * 1.5))
                    ds_data.resize((new_cap,))
                    ds_indices.resize((new_cap,))
                    current_capacity = new_cap

                if chunk_nnz > 0:
                    ds_data[data_offset:data_offset + chunk_nnz] = chunk_csr.data
                    ds_indices[data_offset:data_offset + chunk_nnz] = chunk_csr.indices

                chunk_indptr = chunk_csr.indptr[1:].astype(np.int64) + data_offset
                ds_indptr[chunk_start + 1:chunk_end + 1] = chunk_indptr

                data_offset += chunk_nnz
                del chunk_csr, chunk_data
                pbar.update(chunk_end - chunk_start)

        # Trim datasets to actual size
        total_nnz = data_offset
        ds_data.resize((total_nnz,))
        ds_indices.resize((total_nnz,))

        sparsity = 1.0 - total_nnz / (n_kept * n_genes)
        logging.info(f"Expression matrix: {n_kept:,} × {n_genes}, nnz: {total_nnz:,}, sparsity: {sparsity:.1%}")

        # Write obs (DataFrame)
        obs_group = f.create_group('obs')
        obs_group.attrs['encoding-type'] = 'dataframe'
        obs_group.attrs['encoding-version'] = '0.2.0'
        obs_group.attrs['_index'] = '_index'
        obs_index = np.array(obs_df.index, dtype='S')
        obs_group.create_dataset('_index', data=obs_index, compression='gzip')
        column_order = []
        for col in obs_df.columns:
            column_order.append(col)
            if obs_df[col].dtype.name == 'category':
                cat_group = obs_group.create_group(col)
                cat_group.attrs['encoding-type'] = 'categorical'
                cat_group.attrs['encoding-version'] = '0.2.0'
                cat_group.attrs['ordered'] = False
                categories = obs_df[col].cat.categories.values
                cat_group.create_dataset('categories', data=np.array(categories, dtype='S'),
                                         compression='gzip')
                cat_group.create_dataset('codes', data=obs_df[col].cat.codes.values,
                                         compression='gzip')
            else:
                obs_group.create_dataset(col, data=obs_df[col].values, compression='gzip')
        obs_group.attrs['column-order'] = column_order

        # Write var (from original adata)
        var_group = f.create_group('var')
        var_group.attrs['encoding-type'] = 'dataframe'
        var_group.attrs['encoding-version'] = '0.2.0'
        var_group.attrs['_index'] = '_index'
        var_index = np.array(adata.var.index, dtype='S')
        var_group.create_dataset('_index', data=var_index, compression='gzip')
        var_col_order = []
        for col in adata.var.columns:
            var_col_order.append(col)
            var_group.create_dataset(col, data=np.array(adata.var[col].values, dtype='S'),
                                     compression='gzip')
        var_group.attrs['column-order'] = var_col_order

        # Write obsm/spatial
        obsm_group = f.create_group('obsm')
        obsm_group.create_dataset('spatial', data=new_raw_coords, compression='gzip')

        # Write uns/generation_info
        uns_group = f.create_group('uns')
        gen_group = uns_group.create_group('generation_info')
        gen_info = {
            'source_file': args.input_data,
            'n_original_points': adata.n_obs,
            'n_generated_points': n_kept,
            'expr_filter_ratio': args.expr_filter_ratio,
            'sampling_time_seconds': sampling_time,
            'inference_total_seconds': infer_total_sec,
            'inference_model_seconds': infer_model_sec,
            'inference_io_seconds': infer_io_sec,
            'spatial_scaling_factor': spatial_scaling,
            'dist_tol_multiplier': args.dist_tol,
            'knn_neighbors': args.k_neighbors,
        }
        for k, v in gen_info.items():
            if isinstance(v, str):
                gen_group.attrs[k] = v
            else:
                gen_group.attrs[k] = v
        opt_group_uns = gen_group.create_group('optimization')
        opt_group_uns.attrs['n_workers'] = args.n_workers
        opt_group_uns.attrs['candidate_chunk_size'] = args.candidate_chunk_size
        opt_group_uns.attrs['inference_chunk_size'] = args.inference_chunk_size
        opt_group_uns.attrs['inference_batch_size'] = args.inference_batch_size
        opt_group_uns.attrs['low_memory_mode'] = args.low_memory

        if args.cat_keys:
            gen_group.attrs['predicted_categorical_keys'] = args.cat_keys

        # Encode file format version for anndata compatibility
        f.attrs['encoding-type'] = 'anndata'
        f.attrs['encoding-version'] = '0.1.0'

    logging.info(f"High-resolution data successfully saved to: {output_path}")

    # Cleanup temp files
    del expr_mmap
    _cleanup_mmap(expr_mmap_path, dropout_mmap_path)


def _cleanup_mmap(*paths):
    """Remove temporary memory-mapped files."""
    for p in paths:
        if p is not None and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == '__main__':
    main()
