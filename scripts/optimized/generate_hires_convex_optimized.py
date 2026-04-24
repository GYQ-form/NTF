import torch
import anndata
import numpy as np
import logging
import argparse
import os
import time
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.stats import mode
import pandas as pd
import warnings
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import h5py

from NTF.train import train
from NTF.sample import sample_points
from NTF.config import add_shared_args, process_args
import scanpy as sc
from NTF.helper_func import normalize_adata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _thread_sample_hull_chunk(tesselation, min_bound, max_bound, chunk_size, seed):
    """Thread-safe worker for convex hull sampling.

    Delaunay.find_simplex() is a C-level call that releases the GIL,
    so threads achieve true parallelism without the overhead of fork().
    """
    rng = np.random.RandomState(seed)
    candidates = rng.uniform(low=min_bound, high=max_bound, size=(chunk_size, 3)).astype(np.float32)
    mask_inside = tesselation.find_simplex(candidates) >= 0
    return candidates[mask_inside]


def sample_points_in_hull_parallel(hull: ConvexHull, n_points: int, n_workers: int = 8,
                                   chunk_size: int = 10_000_000, low_memory: bool = False) -> np.ndarray:
    """Sample points inside convex hull using thread-parallel processing.

    Uses ThreadPoolExecutor instead of multiprocessing to avoid fork/memory issues.
    Delaunay.find_simplex() releases the GIL so threads achieve real parallelism.

    Parameters
    ----------
    hull : ConvexHull
        Convex hull object from scipy.spatial
    n_points : int
        Target number of points to sample
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
    logging.info(f"Sampling {n_points:,} points inside convex hull ({n_workers} threads)...")

    tesselation = Delaunay(hull.points)
    min_bound = hull.points.min(axis=0)
    max_bound = hull.points.max(axis=0)

    hull_volume = hull.volume
    bbox_volume = np.prod(max_bound - min_bound)
    acceptance_ratio = hull_volume / bbox_volume if bbox_volume > 0 else 0

    if acceptance_ratio == 0:
        logging.error("Convex hull volume is zero; cannot sample.")
        return np.array([])

    logging.info(f"Hull volume / bounding-box volume = {acceptance_ratio:.3f}")

    if low_memory:
        logging.info("Low-memory mode: using sequential processing")
        return _sample_hull_sequential(tesselation, n_points, chunk_size, min_bound, max_bound)

    # Estimate total candidates needed (with 30% buffer)
    total_candidates_needed = int(n_points / acceptance_ratio * 1.3)
    n_chunks = max(1, (total_candidates_needed + chunk_size - 1) // chunk_size)

    logging.info(f"Estimated {total_candidates_needed:,} candidates across {n_chunks} chunks")

    # Thread-parallel sampling — shared Delaunay, no fork, no memory duplication
    sampled_points = []
    collected_count = 0

    with tqdm(total=n_points, desc="Sampling points", unit="pts", unit_scale=True) as pbar:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _thread_sample_hull_chunk, tesselation, min_bound, max_bound, chunk_size, i * 12345
                ): i for i in range(n_chunks)
            }

            for future in as_completed(futures):
                result = future.result()
                if len(result) > 0:
                    sampled_points.append(result)
                    collected_count += len(result)
                    pbar.update(min(len(result), n_points - (collected_count - len(result))))

                if collected_count >= n_points:
                    for f in futures:
                        f.cancel()
                    break

    if collected_count == 0:
        logging.error("No valid points sampled. Check hull geometry.")
        return np.array([])

    all_points = np.vstack(sampled_points)

    if collected_count < n_points:
        warnings.warn(f"Only {collected_count:,} points sampled, fewer than target {n_points:,}.")

    final_points = all_points[:n_points]
    logging.info(f"Successfully sampled {len(final_points):,} points")
    return final_points


def _sample_hull_sequential(tesselation, n_points, chunk_size, min_bound, max_bound):
    """Sequential fallback for low-memory mode."""
    sampled_points = []
    collected_count = 0
    max_iterations = 1000
    iteration = 0

    with tqdm(total=n_points, desc="Sampling (sequential)", unit="pts", unit_scale=True) as pbar:
        while collected_count < n_points and iteration < max_iterations:
            candidates = np.random.uniform(low=min_bound, high=max_bound, size=(chunk_size, 3)).astype(np.float32)
            mask_inside = tesselation.find_simplex(candidates) >= 0
            points_inside = candidates[mask_inside]

            if len(points_inside) > 0:
                sampled_points.append(points_inside)
                collected_count += len(points_inside)
                pbar.update(len(points_inside))

            iteration += 1

    if collected_count == 0:
        return np.array([])

    return np.vstack(sampled_points)[:n_points]


def get_args():
    shared_parser = add_shared_args()
    parser = argparse.ArgumentParser(
        description="Train model and generate high-resolution data using optimized convex hull sampling.",
        parents=[shared_parser]
    )

    task_group = parser.add_argument_group('Task Specific: Generation')
    task_group.add_argument('--n_generate', type=int, default=1000000, help='Number of new points to generate.')

    task_group.add_argument('--cat_keys', type=str, nargs='+', default=[],
                            help='List of categorical variables in adata.obs to predict via KNN voting.')
    task_group.add_argument('--k_neighbors', type=int, default=5,
                            help='Number of nearest neighbors for majority voting.')

    opt_group = parser.add_argument_group('Optimization Parameters')
    opt_group.add_argument('--n_workers', type=int, default=8,
                          help='Number of parallel threads for candidate generation (default 8).')
    opt_group.add_argument('--candidate_chunk_size', type=int, default=5_000_000,
                          help='Candidates per chunk (default 5M). Larger = faster but more memory.')
    opt_group.add_argument('--inference_chunk_size', type=int, default=4_000_000,
                          help='Points per chunk for the outer inference loop (default 4M).')
    opt_group.add_argument('--inference_batch_size', type=int, default=16384,
                          help='Batch size inside sample_points() for GPU forward passes (default 16384).')
    opt_group.add_argument('--low_memory', action='store_true',
                          help='Use sequential processing with smaller chunks (slower but lower memory).')

    args = parser.parse_args()
    args = process_args(args)
    return args


def main():
    args = get_args()
    logging.info(f"Configuration loaded. Results will be saved to: {args.output_dir}")
    logging.info(f"Optimization: {args.n_workers} threads, {args.candidate_chunk_size:,} candidates/chunk, "
                f"inference chunk {args.inference_chunk_size:,}, batch {args.inference_batch_size:,}")

    # --- 1. Load and preprocess data ---
    logging.info(f"Loading data from {args.input_data}...")
    adata = anndata.read_h5ad(args.input_data)
    adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)

    raw_coords = adata.obsm['spatial'].copy()
    raw_bbox_size = (raw_coords.max(0) - raw_coords.min(0)).max()
    spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
    adata.obsm['spatial'] *= spatial_scaling
    logging.info(f"Training on {adata.n_obs:,} cells. Spatial scaling factor: {spatial_scaling:.6f}")

    # --- 2. Sample coordinates BEFORE training (thread-safe, low memory footprint) ---
    logging.info("Computing 3D convex hull from original data...")
    try:
        hull = ConvexHull(raw_coords)
    except Exception as e:
        logging.error(f"Failed to compute convex hull: {e}")
        return

    t_start_sampling = time.time()
    new_raw_coords = sample_points_in_hull_parallel(
        hull, args.n_generate,
        n_workers=args.n_workers,
        chunk_size=args.candidate_chunk_size,
        low_memory=args.low_memory
    )
    sampling_time = time.time() - t_start_sampling

    if new_raw_coords.shape[0] == 0:
        logging.error("No new coordinates generated; aborting.")
        return

    logging.info(f"Sampling completed in {sampling_time:.1f}s ({args.n_generate/sampling_time:.0f} pts/sec)")

    # --- 3. Train model ---
    t_start_train = time.time()
    trained_NTF = train(adata, args)
    trained_inr = trained_NTF.inr
    train_time_sec = time.time() - t_start_train
    logging.info(f"Training complete in {train_time_sec:.1f} seconds")

    # --- 4. Predict expression at new coordinates (memory-mapped) ---
    scaled_new_coords = new_raw_coords * spatial_scaling
    new_coords_tensor = torch.from_numpy(scaled_new_coords).float().to(args.device)

    n_points = len(new_coords_tensor)
    n_genes = len(adata.var_names)
    logging.info(f"Predicting at {n_points:,} points × {n_genes} genes "
                f"(chunk {args.inference_chunk_size:,}, batch {args.inference_batch_size:,})")
    logging.info(f"Dense matrix would be {n_points * n_genes * 4 / 1e9:.1f} GB — using memory-mapped file")

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_dir = args.output_dir

    expr_mmap_path = os.path.join(tmp_dir, '_tmp_expr.mmap')
    expr_mmap = np.memmap(expr_mmap_path, dtype=np.float32, mode='w+', shape=(n_points, n_genes))

    has_dropout = not args.no_dropout
    dropout_mmap = None
    dropout_mmap_path = None
    if has_dropout:
        dropout_mmap_path = os.path.join(tmp_dir, '_tmp_dropout.mmap')
        dropout_mmap = np.memmap(dropout_mmap_path, dtype=np.float32, mode='w+', shape=(n_points, n_genes))

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

            if has_dropout and "dropout_prob" in batch_results:
                dropout_mmap[i:end] = batch_results["dropout_prob"].cpu().numpy()
            infer_io_sec += time.time() - t_io

            pbar.update(end - i)

    t_io = time.time()
    expr_mmap.flush()
    if dropout_mmap is not None:
        dropout_mmap.flush()
    infer_io_sec += time.time() - t_io

    infer_total_sec = time.time() - t_start
    logging.info(f"Inference complete in {infer_total_sec:.1f}s "
                f"(model: {infer_model_sec:.1f}s, I/O: {infer_io_sec:.1f}s, "
                f"{n_points/infer_total_sec:.0f} pts/sec)")

    del trained_NTF, trained_inr, new_coords_tensor
    torch.cuda.empty_cache()

    # --- 5. KNN-based majority voting ---
    obs_df = pd.DataFrame(index=[f"new_cell_{i}" for i in range(n_points)])

    if args.cat_keys:
        logging.info(f"Running KNN (k={args.k_neighbors}) majority voting for {n_points:,} points...")
        tree = cKDTree(raw_coords)
        k = min(args.k_neighbors, len(raw_coords))
        knn_chunk_size = 2_000_000

        voted_results = {key: np.zeros(n_points, dtype=np.int32) for key in args.cat_keys}

        with tqdm(total=n_points, desc="KNN voting", unit="pts", unit_scale=True) as pbar:
            for start_idx in range(0, n_points, knn_chunk_size):
                end_idx = min(start_idx + knn_chunk_size, n_points)
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

    # --- 6. Stream CSR directly to h5ad via h5py (single-pass, resizable datasets) ---
    output_path = os.path.join(args.output_dir, f'{int(args.n_generate/10000)}w_convex_optimized.h5ad')
    logging.info("Streaming sparse CSR matrix to h5ad via h5py...")

    write_chunk_size = 2_000_000
    n_kept = n_points
    keep_indices = np.arange(n_points)

    del expr_mmap
    expr_mmap = np.memmap(expr_mmap_path, dtype=np.float32, mode='r', shape=(n_points, n_genes))

    INIT_NNZ = 1_000_000_000

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
                chunk_idx = keep_indices[chunk_start:chunk_end]
                chunk_data = np.array(expr_mmap[chunk_idx])

                chunk_csr = sp.csr_matrix(chunk_data)
                chunk_nnz = chunk_csr.nnz

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

        total_nnz = data_offset
        ds_data.resize((total_nnz,))
        ds_indices.resize((total_nnz,))

        sparsity = 1.0 - total_nnz / (n_kept * n_genes)
        logging.info(f"Expression matrix: {n_kept:,} × {n_genes}, nnz: {total_nnz:,}, sparsity: {sparsity:.1%}")

        # Write obs
        obs_group = f.create_group('obs')
        obs_group.attrs['encoding-type'] = 'dataframe'
        obs_group.attrs['encoding-version'] = '0.2.0'
        obs_group.attrs['_index'] = '_index'
        obs_group.create_dataset('_index', data=np.array(obs_df.index, dtype='S'), compression='gzip')
        column_order = []
        for col in obs_df.columns:
            column_order.append(col)
            if obs_df[col].dtype.name == 'category':
                cat_group = obs_group.create_group(col)
                cat_group.attrs['encoding-type'] = 'categorical'
                cat_group.attrs['encoding-version'] = '0.2.0'
                cat_group.attrs['ordered'] = False
                categories = obs_df[col].cat.categories.values
                cat_group.create_dataset('categories', data=np.array(categories, dtype='S'), compression='gzip')
                cat_group.create_dataset('codes', data=obs_df[col].cat.codes.values, compression='gzip')
            else:
                obs_group.create_dataset(col, data=obs_df[col].values, compression='gzip')
        obs_group.attrs['column-order'] = column_order

        # Write var
        var_group = f.create_group('var')
        var_group.attrs['encoding-type'] = 'dataframe'
        var_group.attrs['encoding-version'] = '0.2.0'
        var_group.attrs['_index'] = '_index'
        var_group.create_dataset('_index', data=np.array(adata.var.index, dtype='S'), compression='gzip')
        var_col_order = []
        for col in adata.var.columns:
            var_col_order.append(col)
            var_group.create_dataset(col, data=np.array(adata.var[col].values, dtype='S'), compression='gzip')
        var_group.attrs['column-order'] = var_col_order

        # Write obsm/spatial
        obsm_group = f.create_group('obsm')
        obsm_group.create_dataset('spatial', data=new_raw_coords, compression='gzip')

        # Write uns
        uns_group = f.create_group('uns')
        gen_group = uns_group.create_group('generation_info')
        for k, v in {
            'source_file': args.input_data,
            'n_original_points': adata.n_obs,
            'n_generated_points': n_kept,
            'sampling_time_seconds': sampling_time,
            'training_time_seconds': train_time_sec,
            'inference_total_seconds': infer_total_sec,
            'inference_model_seconds': infer_model_sec,
            'inference_io_seconds': infer_io_sec,
            'spatial_scaling_factor': spatial_scaling,
            'knn_neighbors': args.k_neighbors,
        }.items():
            gen_group.attrs[k] = v
        opt_uns = gen_group.create_group('optimization')
        opt_uns.attrs['n_workers'] = args.n_workers
        opt_uns.attrs['candidate_chunk_size'] = args.candidate_chunk_size
        opt_uns.attrs['inference_chunk_size'] = args.inference_chunk_size
        opt_uns.attrs['inference_batch_size'] = args.inference_batch_size
        opt_uns.attrs['low_memory_mode'] = args.low_memory
        if args.cat_keys:
            gen_group.attrs['predicted_categorical_keys'] = args.cat_keys

        f.attrs['encoding-type'] = 'anndata'
        f.attrs['encoding-version'] = '0.1.0'

    logging.info(f"High-resolution data successfully saved to: {output_path}")

    # Cleanup temp files
    del expr_mmap
    _cleanup_mmap(expr_mmap_path, dropout_mmap_path)

    # Performance summary
    total_time = sampling_time + train_time_sec + infer_total_sec
    logging.info(f"\n{'='*60}")
    logging.info(f"PERFORMANCE SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Sampling:        {sampling_time:8.1f}s")
    logging.info(f"Training:        {train_time_sec:8.1f}s")
    logging.info(f"Inference total: {infer_total_sec:8.1f}s")
    logging.info(f"  ├─ Model:      {infer_model_sec:8.1f}s")
    logging.info(f"  └─ I/O:        {infer_io_sec:8.1f}s")
    logging.info(f"{'─'*60}")
    logging.info(f"Total:           {total_time:8.1f}s")
    logging.info(f"{'='*60}\n")


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
