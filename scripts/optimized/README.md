# Optimized High-Resolution Generation Scripts

These scripts are optimized versions of `generate_hires_convex.py` and `generate_hires_distfield.py` for generating **tens of millions or more** points under limited memory.

> **When to use these**: The standard scripts in `scripts/` work well for up to ~10 million points. When generating 10M+ points (especially 100M+), you should use the optimized versions here, since the dense expression matrix alone would exceed available RAM (e.g. 100M points × 1000 genes × 4 bytes ≈ 400 GB).

## What's Different

**Model training and inference are extremely fast** — a typical run trains in ~12 minutes and infers 100M points in ~1 minutes on a single GPU. **The dominant cost at ultra-large scale is data I/O**: writing hundreds of GB of predictions to disk and converting them into sparse format for the final `.h5ad` file.

The optimized scripts address this with the following changes:

| Feature | Standard scripts | Optimized scripts |
|---|---|---|
| Parallelism | Sequential sampling | `ThreadPoolExecutor` (GIL-free C calls) |
| Inference output | Accumulate in RAM | Stream to `np.memmap` on disk |
| h5ad writing | `adata.write_h5ad()` (needs full matrix in RAM) | Chunk-wise CSR streaming via `h5py` |
| Compression | gzip (slow) | lzf (5–10× faster writes) |
| int overflow | int32 indptr | int64 indptr (supports nnz > 2 billion) |
| Timing | Total only | Breakdown: sampling / training / model inference / I/O |

## Scripts

### `generate_hires_distfield_optimized.py`

Distance-field sampling (KD-Tree boundary). Best for **non-convex** or complex tissue shapes.

```bash
python scripts/optimized/generate_hires_distfield_optimized.py \
    -i data.h5ad -o output/ --slice_id slice \
    --n_generate 100000000 \
    --dist_tol 20 \
    --expr_filter_ratio 0.03
```

### `generate_hires_convex_optimized.py`

Convex hull sampling (Delaunay). Best for **roughly convex** tissues.

```bash
python scripts/optimized/generate_hires_convex_optimized.py \
    -i data.h5ad -o output/ --slice_id slice \
    --n_generate 100000000
```

## Additional Arguments

Both scripts accept all [shared NTF arguments](../README.md) plus:

| Argument | Default | Description |
|---|---|---|
| `--n_generate` | 1000000 | Number of new points to generate |
| `--cat_keys` | `[]` | Categorical variables for KNN voting |
| `--k_neighbors` | 5 | Nearest neighbors for KNN voting |
| `--n_workers` | 8 | Parallel threads for candidate sampling |
| `--candidate_chunk_size` | 5000000 | Candidates per sampling chunk |
| `--inference_chunk_size` | 4000000 | Points per outer inference loop iteration |
| `--inference_batch_size` | 16384 | GPU batch size inside `sample_points()` |
| `--low_memory` | off | Fall back to sequential sampling |

The distfield script additionally accepts:

| Argument | Default | Description |
|---|---|---|
| `--dist_tol` | 20.0 | Boundary tolerance multiplier |
| `--expr_filter_ratio` | 0.03 | Remove points with total expression below `max × ratio` |



## Output

Each script produces a single `.h5ad` file with suffix `_convex_optimized` or `_distfield_optimized`. The file is in anndata-compatible format with a CSR sparse `X` matrix. Temporary memmap files are automatically cleaned up on success.

