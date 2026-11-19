# NTF Analysis Scripts

This directory contains supplementary analysis scripts for reproducing the paper's results. These scripts cover two main use cases: **simulated data generation** and **reconstruction experiments** on sparse-slice input and mixed-resolution (including low-resolution) 3D data.

---

## Overview

| File                                             | Type              | Description                                                                                                                                                                                                                                                                         |
| ------------------------------------------------ | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `simulate_data.py`                             | Standalone script | Generate synthetic 3D spatial transcriptomics datasets of varying scale                                                                                                                                                                                                             |
| `simu_sparse_interval.py`                      | Standalone script | Evaluate NTF reconstruction across different slice-sampling intervals (sparse input)                                                                                                                                                                                                |
| `simu_mixed_resolution.py`                     | Standalone script | Evaluate NTF reconstruction with mixed high/low-resolution slice inputs                                                                                                                                                                                                             |
| `tutorial_simulation_reconstruction.py`        | Notebook source   | Public Tutorial 1: HK-free starting data, optional mixed-resolution construction/training/full-coordinate reconstruction, complete cached results, `Blues` 2D comparisons and three-view 3D input/truth/prediction visualisation                                                  |
| `tutorial_simulation_mousebrain_full_rerun.py` | Notebook source   | Public Tutorial 4: unconditional from-scratch 100-gene mixed-resolution construction/training/all-coordinate reconstruction; all per-gene metrics with 90-Diff overall means, HK noise controls, raw-expression maps and transparent 3D points; no cached results or point thinning |
| `simu_expr_on_domain.py`                       | Helper module     | Simulate spatially-patterned gene expression on domain-annotated tissue                                                                                                                                                                                                             |
| `resolution_utils.py`                          | Helper module     | Utilities for degrading slice resolution and assembling mixed-resolution datasets                                                                                                                                                                                                   |
| `simulate_gene_count_data.py`                  | Standalone script | Generate simple-pattern simulation data with an arbitrary gene count (R2M4 revision experiment)                                                                                                                                                                                     |
| `central_eval.py`                              | Standalone script | Re-evaluate any method's `test_results.h5ad` against the real simulation truth (`layers['truth']`) with vectorized metrics                                                                                                                                                      |
| `eval_svg_conservation_sim.py`                 | Standalone script | SVG/spatial-pattern conservation evaluation (Moran's I, Top-K SVG overlap/Jaccard) on simulation results                                                                                                                                                                            |
| `run_svg_benchmark.py`                         | Standalone script | Run the three-method SVG conservation benchmark on existing predictions                                                                                                                                                                                                             |
| `run_svg_benchmark_scales.py`                  | Standalone script | All-scale SVG benchmark (19 simulated scales) with 3D and per-2D-slice Moran's I corr / SVG Jaccard; Excel workbook split by method                                                                                                                                                 |
| `run_svg_benchmark_scales_unist.py`            | Standalone script | R2.t5 patch: same all-scale SVG protocol applied to UniST-fast (19 scales) / UniST-inr (6 scales), with a tolerance-0 paired check against the three-method results                                                                                                                 |
| `build_summary_with_unist.py`                  | Standalone script | Merge the UniST SVG scale rows into a five-method Excel workbook (`svg_benchmark_allscales_summary_with_unist.xlsx`)                                                                                                                                                              |
| `plot_svg_scale_five_methods.R`                | Standalone script | Five-method 2×2 metric-vs-scale figure (ggplot classic, svg + png)                                                                                                                                                                                                                 |
| `simulation_allmetrics_bm.R`                   | Standalone script | Refined all-scale benchmark figure: per-metric line plots (RMSE/SSIM/Spearman + 3D/2D Moran's I corr & SVG Jaccard) and log10 time-cost bar plot from `simulation_scale_allmetrics.xlsx` (NTF/SpatialZ/FEAST/UniST)                                                               |
| `plot_simulation_mousebrain.R`                 | Standalone script | Mouse-brain simulation sparse-slice benchmark figure: avg Spearman/RMSE/3D SSIM vs n_slices_used from `simulation_mousebrain.xlsx` (NTF/FEAST/SpatialZ/UniST, 17 slice settings each)                                                                                             |
| `summarize_gene_count_benchmark.py`            | Standalone script | Collect gene-count benchmark results into summary CSVs, an Excel workbook and figures                                                                                                                                                                                               |
| `run_with_monitor.sh`                          | Shell wrapper     | Run a command in a conda env on a chosen GPU while recording wall time, peak GPU memory and peak RSS                                                                                                                                                                                |
| `run_ablation_sim.sh`                          | Shell script      | R1 Major 7: run the 10 model-component ablation configs (M0–M9) on the 1M-cell 100-gene simulation                                                                                                                                                                                 |
| `run_central_eval_sim.sh`                      | Shell script      | R1 Major 7: centrally re-evaluate all ablation runs against `layers['truth']` via `central_eval.py`                                                                                                                                                                             |
| `summarize_ablation.py`                        | Standalone script | R1 Major 7: collect ablation results (simulation + mouseVC) into summary CSVs and an Excel workbook                                                                                                                                                                                 |
| `plot_ablation.py`                             | Standalone script | R1 Major 7: ablation figures (plotnine, theme_classic, SVG output)                                                                                                                                                                                                                  |
| `plot_refine_ablation.py`                      | Standalone script | R1 Major 7 refine: per-dataset ablation bar charts (simulation / mouseVC, one figure each) from `r1m7_model_ablation/refine/ablation_summary.xlsx`                                                                                                                                |

---

## Revision experiments

The scripts above the line reproduce the paper's original simulation experiments. The scripts below were added during revision (see `mytrain/revision/plan/` and `mytrain/revision/experiment/` for plans, full records and results).

### SVG conservation (`eval_svg_conservation_sim.py`, `run_svg_benchmark.py`)

`eval_svg_conservation_sim.py` evaluates spatial-pattern conservation of an NTF result h5ad (`--dataset scale|domain`): stratified sampling, a shared 3D kNN graph, gene-wise Moran's I, and Top-K SVG overlap/Jaccard, using chunked h5py reads so 10M-point files never load fully into memory. `run_svg_benchmark.py` applies the same evaluation uniformly to NTF/FEAST/SpatialZ predictions.

```bash
python analysis/simulation/eval_svg_conservation_sim.py \
    --dataset scale \
    --result mytrain/res/simulation/scale_test/100w/test_results.h5ad \
    --output_dir out/scale_100w \
    --sample_size 100000 --k 15 --seed 42 --topk 5 10 20 50
```

Results and full record: `mytrain/revision/experiment/svg_benchmark/`.

### all-scale SVG benchmark (`run_svg_benchmark_scales.py`)

`run_svg_benchmark_scales.py` reuses the readers, stratified sampler, kNN graph and Moran's I implementation of `run_svg_benchmark.py` (imported, must sit next to it) and extends the benchmark to every simulated scale of the simple simulation data — 10w–100w by 10w and 200w–1000w by 100w (19 scales). Besides the whole-volume 3D Moran's I correlation and Top-20 SVG Jaccard, it computes both metrics separately on each 2D slice (2D kNN graph on xy coordinates; slices with fewer than `--min_slice_points` sampled points are skipped and marked NA) and summarizes the per-slice mean/std. The Excel workbook is split into one sheet per method (rows = scale, columns = metrics), plus `Summary`, `Method_Status` and `Metadata` sheets; a 2×2 line-plot figure shows metric-vs-scale curves per method. SpatialZ has no compatible results for 60w, 90w and ≥200w; those stay `NA`.

```bash
python analysis/simulation/run_svg_benchmark_scales.py \
    --output_dir mytrain/revision/experiment/svg_benchmark_allscales \
    --sample_size 100000 --graph_k 15 --seed 42 --top_k 20 --min_slice_points 30
```

Results and full record: `mytrain/revision/experiment/svg_benchmark_allscales/`.

### gene-count benchmark (`simulate_gene_count_data.py`, `central_eval.py`, `summarize_gene_count_benchmark.py`, `run_with_monitor.sh`)

`simulate_gene_count_data.py` is the scale_test generator parameterized by `--n_genes` (patterns assigned cyclically), `--output` and `--seed`; generation is chunked over genes so 1M × 10,000 datasets fit in memory. `central_eval.py` re-evaluates any method's `test_results.h5ad` against the real `layers['truth']` of the input data (the historical pipelines evaluate against different targets) and writes a standardized `eval_input.h5ad` for the SVG evaluation above. `summarize_gene_count_benchmark.py` aggregates everything into `benchmark_long.csv`, `summary_by_genes.csv`, an Excel workbook and figures. `run_with_monitor.sh` wraps each run to record timing and memory.

```bash
# data (1M cells, seed 42)
python analysis/simulation/simulate_gene_count_data.py \
    --n_slices 100 --n_genes 100 --seed 42 \
    --output mytrain/data/simulation/gene_count_test/100wcell_100gene.h5ad

# NTF training/eval (use GPU 6 or 7 per revision/AGENTS.md)
CUDA_VISIBLE_DEVICES=6 python scripts/train_and_eval.py \
    -i mytrain/data/simulation/gene_count_test/100wcell_100gene.h5ad \
    -o mytrain/res/simulation/gene_count_test/100wcell_100gene \
    --slice_id slice_id --split_mode label --split_column group \
    --train_label train --test_label test

# central re-evaluation against real truth, then SVG evaluation
python analysis/simulation/central_eval.py \
    --input  mytrain/data/simulation/gene_count_test/100wcell_100gene.h5ad \
    --result mytrain/res/simulation/gene_count_test/100wcell_100gene/test_results.h5ad \
    --output_dir mytrain/res/simulation/gene_count_test/100wcell_100gene

# final tables/figures (paths inside the script point to the revision experiment dir)
python analysis/simulation/summarize_gene_count_benchmark.py
```

Results and full record: `mytrain/revision/experiment/gene_count_benchmark/`.

### model-component ablation (`run_ablation_sim.sh`, `run_central_eval_sim.sh`, `summarize_ablation.py`, `plot_ablation.py`)

Leave-one-out ablation of the NTF components that can be toggled from the command line (dropout network, PSF modeling, smoothness regularization, slice embeddings, pixel/slice variance, plus a bias-correction-on config and an all-off bare INR), evaluated on the 1M-cell 100-gene simulation (`100wcell_100gene.h5ad`, split by `obs['group']`) and on the STARmap mouse VC 5-gene data (mouseVC-side runner lives in `analysis/STARmap_mouse_VC/run_ablation_mousevc.sh`). All runs go through `run_with_monitor.sh` for timing/memory; simulation-side runs are centrally re-evaluated against `layers['truth']` with `central_eval.py`.

```bash
# simulation side (GPU 6)
GPU=6 bash analysis/simulation/run_ablation_sim.sh
bash analysis/simulation/run_central_eval_sim.sh
# mouseVC side (GPU 7)
GPU=7 bash analysis/STARmap_mouse_VC/run_ablation_mousevc.sh
# summary + figures (paths inside the scripts point to the revision experiment dir)
python analysis/simulation/summarize_ablation.py
python analysis/simulation/plot_ablation.py
```

Results and full record: `mytrain/revision/experiment/r1m7_model_ablation/`.

---

## Scripts

### `simulate_data.py`

Generates a synthetic AnnData object with realistic 3D spatial structure, including multiple cell clusters, spatially-patterned gene expression, per-slice batch effects, and a train/test split column.

#### Arguments

| Argument                | Default | Description                                                   |
| ----------------------- | ------- | ------------------------------------------------------------- |
| `--n_slices`          | 100     | Number of Z-axis slices (total cells = `n_slices × 10000`) |
| `--n_clusters`        | 10      | Number of spatial cell clusters                               |
| `--genes_per_pattern` | 20      | Number of genes to repeat per expression pattern              |
| `--batch_effect`      | 0.2     | Standard deviation of per-slice batch effects                 |
| `--noise`             | 0.1     | Standard deviation of Gaussian noise added to expression      |

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

| Argument           | Default                | Description                                                                      |
| ------------------ | ---------------------- | -------------------------------------------------------------------------------- |
| `--is_simulate`  | False                  | If set, treat `--input_data` as an already-simulated dataset                   |
| `--domain_key`   | `domain`             | `adata.obs` column containing domain/region annotations                        |
| `--n_genes`      | 100                    | Number of genes to simulate (when generating from template)                      |
| `--total_slices` | 100                    | Total number of slices to assign when simulating                                 |
| `--intervals`    | `2 3 4 5 6 7 8 9 10` | Slice sampling intervals to test (k=1 uses all slices, k=2 uses every 2nd, etc.) |

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

| Argument                   | Default                         | Description                                                                                                                                |
| -------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `--slice_interval_k`     | 5                               | Base slice sampling interval (every k-th slice is used)                                                                                    |
| `--high_res_intervals_m` | `1 2 3 4 5 6 10 20 30`        | List of high-resolution frequency values to test. `m=1` means all selected slices are high-res; larger `m` means fewer high-res slices |
| `--bin_factors`          | `2.0 3.0 4.0 5.0 6.0 7.0 8.0` | List of resolution degradation factors for low-res slices                                                                                  |

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

## UniST benchmark 相关

- **`eval_scale_unist.py`** – scale test 四方法（NTF/SpatialZ/FEAST/UniST）集中重评估（统一 vs `layers['truth']` 口径、GT test 坐标、NTF 跳基因规则与 3D SSIM 实现）；产出 per-gene 指标与汇总 csv。
- **`compute_serial_estimates.py`** – UniST 串行等效耗时（serial_estimate_s）：无竞争单 gap 成本模型 t(n)=2.5×n^0.708，n=flanking 细胞数；并行/分片调度污染的修正口径。
- **`build_final_summary.py`** – 汇总 benchmark_long.csv 与多 sheet xlsx。
- **`plot_r2t5_figures.R`** – Fig 2a/2d 更新版与各板块对比图（ggplot classic，svg，图例外置底部）。
- **`render_fig2b.py`** – Fig 2b 六联 3D 定性渲染（noisy/GT/NTF/SpatialZ/FEAST/UniST × 3 基因）。
- **`quick_start_check.py`** – UniST 环境+权重的官方 quick start 验收脚本。

### UniST SVG scale test 评估

- **`run_svg_benchmark_scales_unist.py`** – 与 `run_svg_benchmark_scales.py` 完全相同的协议
  （NTF 参考文件抽样 seed=42 10 万点、3D/2D k=15 对称 binary kNN、Top-20 Jaccard），
  评估 UniST-fast（全 19 档）与 UniST-inr（10/20/30/40/50/100w 共 6 档，其余显式 NA）；
  UniST 边界外推片 prediction 为零填充：3D 指标按 r2t5 集中评估口径保留，2D 逐切片显式记 NA；
  脚本末尾自动做 paired 核查（NTF 10w/100w 同路径复算 vs allscales 结果，容差 0）。
  需与 `run_svg_benchmark.py` 同目录（import 其读者/抽样/构图/Moran's I 实现）。
- **`build_summary_with_unist.py`** – 合并三方法既有结果与 UniST 行为五方法 Excel
  （不动原 xlsx；UniST sheet 多一列 `n_boundary_slices_skipped`），并输出合并长表 csv。
- **`plot_svg_scale_five_methods.R`** – 五方法 2×2 折线图（ggplot classic，svg+png，
  UniST-inr/fast 配色 #D95F02/#F4A582，NA 档不画点）。
