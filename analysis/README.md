# NTF Analysis Scripts

This directory contains supplementary analysis scripts for reproducing the paper's results. These scripts cover two main use cases: **simulated data generation** and **reconstruction experiments** on sparse-slice input and mixed-resolution (including low-resolution) 3D data.

---

## Overview

| File | Type | Description |
|---|---|---|
| `simulate_data.py` | Standalone script | Generate synthetic 3D spatial transcriptomics datasets of varying scale |
| `simu_sparse_interval.py` | Standalone script | Evaluate NTF reconstruction across different slice-sampling intervals (sparse input) |
| `simu_mixed_resolution.py` | Standalone script | Evaluate NTF reconstruction with mixed high/low-resolution slice inputs |
| `simu_expr_on_domain.py` | Helper module | Simulate spatially-patterned gene expression on domain-annotated tissue |
| `resolution_utils.py` | Helper module | Utilities for degrading slice resolution and assembling mixed-resolution datasets |

---

## Scripts

### `simulate_data.py`

Generates a synthetic AnnData object with realistic 3D spatial structure, including multiple cell clusters, spatially-patterned gene expression, per-slice batch effects, and a train/test split column.

#### Arguments

| Argument | Default | Description |
|---|---|---|
| `--n_slices` | 100 | Number of Z-axis slices (total cells = `n_slices × 10000`) |
| `--n_clusters` | 10 | Number of spatial cell clusters |
| `--genes_per_pattern` | 20 | Number of genes to repeat per expression pattern |
| `--batch_effect` | 0.2 | Standard deviation of per-slice batch effects |
| `--noise` | 0.1 | Standard deviation of Gaussian noise added to expression |

The output path is hard-coded in the `__main__` block and should be edited before running.

#### Example

```bash
python analysis/simulate_data.py \
    --n_slices 50 \
    --n_clusters 8 \
    --genes_per_pattern 10
```

#### Output

An `.h5ad` file containing:
- `adata.X`: Observed gene expression (with batch effects and noise)
- `adata.layers['truth']`: Ground-truth expression (before noise/batch effects)
- `adata.obsm['spatial']`: 3D coordinates
- `adata.obs['slice_id']`: Slice assignment based on Z coordinate
- `adata.obs['cluster_id']`: Cell cluster assignment
- `adata.obs['group']`: `'train'` or `'test'` split label

---

### `simu_sparse_interval.py`

Trains an NTF model using subsets of slices selected at different intervals and evaluates the reconstruction on the full ground-truth dataset. This reproduces the **sparse slice input** experiment.

Optionally generates ground-truth data from a template with domain annotations (via `simu_expr_on_domain.py`), or loads a pre-simulated `.h5ad` file directly.

#### Shared Arguments

Inherits all shared NTF arguments from `NTF.config.add_shared_args` (see [`scripts/README.md`](../scripts/README.md)).

#### Task-Specific Arguments

| Argument | Default | Description |
|---|---|---|
| `--is_simulate` | False | If set, treat `--input_data` as an already-simulated dataset |
| `--domain_key` | `domain` | `adata.obs` column containing domain/region annotations |
| `--n_genes` | 100 | Number of genes to simulate (when generating from template) |
| `--total_slices` | 100 | Total number of slices to assign when simulating |
| `--intervals` | `2 3 4 5 6 7 8 9 10` | Slice sampling intervals to test (k=1 uses all slices, k=2 uses every 2nd, etc.) |

#### Example

```bash
# Using a pre-simulated dataset
python analysis/simu_sparse_interval.py \
    -i data/simulated.h5ad \
    -o results/sparse_interval/ \
    --is_simulate \
    --intervals 2 4 6 8 10

# Generating from a domain-annotated template
python analysis/simu_sparse_interval.py \
    -i data/template.h5ad \
    -o results/sparse_interval/ \
    --domain_key domain \
    --n_genes 100 \
    --total_slices 100 \
    --intervals 2 5 10
```

#### Outputs

For each interval `k`:
- `results/sparse_interval/k{k}.h5ad` – Reconstructed AnnData with `layers['prediction']`
- `results/sparse_interval/gene_metrics_k{k}.csv` – Per-gene Pearson and Spearman correlations
- `results/sparse_interval/interval_study_results.csv` – Summary metrics across all intervals
- `results/sparse_interval/res.log` – Average correlation summary

---

### `simu_mixed_resolution.py`

Trains NTF on datasets with a mix of high-resolution and low-resolution slices, sweeping over both the low-resolution degradation factor and the frequency of high-resolution slices. This reproduces the **mixed-resolution** experiment.

#### Shared Arguments

Inherits all shared NTF arguments from `NTF.config.add_shared_args`.

#### Task-Specific Arguments

| Argument | Default | Description |
|---|---|---|
| `--slice_interval_k` | 5 | Base slice sampling interval (every k-th slice is used) |
| `--high_res_intervals_m` | `1 2 3 4 5 6 10 20 30` | List of high-resolution frequency values to test. `m=1` means all selected slices are high-res; larger `m` means fewer high-res slices |
| `--bin_factors` | `2.0 3.0 4.0 5.0 6.0 7.0 8.0` | List of resolution degradation factors for low-res slices |

#### Example

```bash
python analysis/simu_mixed_resolution.py \
    -i data/simulated.h5ad \
    -o results/mixed_resolution/ \
    --slice_interval_k 5 \
    --high_res_intervals_m 1 2 5 10 \
    --bin_factors 2.0 4.0 6.0
```

#### Outputs

For each `(m, bin_factor)` combination:
- `results/mixed_resolution/rec_m{m}_bf{bf}.h5ad` – Reconstructed AnnData
- `results/mixed_resolution/train_m{m}_bf{bf}.h5ad` – Training data used
- `results/mixed_resolution/m{m}_bf{bf}.csv` – Per-gene metrics
- `results/mixed_resolution/mixed_res_results.csv` – Summary table
- `results/mixed_resolution/mixed_res_plot.png` – Line plot of Pearson correlation vs. high-res interval

---

## Helper Modules

### `simu_expr_on_domain.py`

Provides two functions used internally by `simu_sparse_interval.py`:

- **`simulate_gene_expression(adata, domain_key, ...)`** – Generates spatially-patterned gene expression using Negative Binomial or Poisson sampling. Each gene is independently active in a random subset of domains, with a randomly selected spatial pattern (linear gradients, radial, constant). Mouse brain data used to generate the simulated data can be found at [Zenodo](https://doi.org/10.5281/zenodo.19590994).
- **`assign_slices(adata, n_slices, ...)`** – Discretises the Z-axis into `n_slices` equally-spaced bins and adds a `slice_id` column to `adata.obs`.

### `resolution_utils.py`

Provides two functions used internally by `simu_mixed_resolution.py`:

- **`degrade_resolution(adata_slice, bin_factor)`** – Aggregates cells within a 2D spatial grid (grid size = `avg_cell_distance × bin_factor`) to simulate lower-resolution measurements.
- **`create_mixed_resolution_dataset(adata_full, slice_interval_k, high_res_interval_m, bin_factor)`** – Assembles a mixed-resolution AnnData by combining degraded and original slices according to the specified intervals.
