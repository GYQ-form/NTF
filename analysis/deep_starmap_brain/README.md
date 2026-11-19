# NTF Analysis Scripts — Deep-STARmap mouse brain

Analysis scripts for the **Deep-STARmap mouse brain** benchmark (Sui et al. 2025, Nat Methods; Zenodo 10.5281/zenodo.16783354): **NTF vs SpatialZ vs FEAST vs UniST** on 5 prediction scenarios, covering reconstruction metrics (RMSE / 3D SSIM / Pearson / Spearman), SVG conservation (Moran's I corr, Top-K SVG Jaccard), cell typing (ARI / NMI / FMI) and computational efficiency (train + inference time only, excluding I/O). UniST is evaluated on senario1/senario2 only (results under `mytrain/res/UniST/<dataset>/`).

## Data

The benchmark uses the **processed data file published together with this repository** — download `deep_starmap_brain.h5ad` and place it at:

```
mytrain/data/deep_starmap_brain/processed/deep_starmap_brain.h5ad
```

The file is analysis-ready (no preprocessing needed):

- **198,675 cells × 500 genes**; `X` = `normalize_total(1e4)` + `log1p` (CSR sparse), raw counts in `layers['counts']`. The 500 genes were selected from the full 1,017-gene panel by detection rate (top 500) and a 3D spatially-variable-gene test (Ljung-Box, BH adj p ≤ 0.05; all 500 significant).
- `obsm['spatial']` / `obs['x','y','z']`: 3D coordinates over 208 optical z-layers (~150 µm block).
- `obs['z_layer']`: optical z-layer index (1–208); `obs['slice_id']`: tissue slice index (1–13), every 16 consecutive optical layers merged (~one cell thick). All scenario splits are defined on `slice_id`.
- `obs['Harmony_labels']`: cell-type ground truth (22 classes); `obs['FUSEmap_main_level']` / `obs['FUSEmap_sub_level']`: additional annotations.
- **Scenario split columns** in `obs` (train/test; test slices in parentheses): `group` — senario1, interleaved holdout (2,4,6,8,10,12); `group2` — senario2, middle band (6,7,8); `group4` — senario3, both-end boundary extrapolation (1,13); `head` — head extrapolation (1,2); `tail` — tail extrapolation (12,13). senario3/head/tail are NTF-only (SpatialZ/FEAST cannot extrapolate).

## Overview

| File                                  | Type              | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_ntf.sh`                        | Run script        | NTF on all 5 scenarios (default config: bs=8192, n_iter=20000, dropout network enabled, `--cat_keys Harmony_labels`)                                                                                                                                                                                                                                                                                                                                                                                                           |
| `run_baselines.sh`                  | Run script        | SpatialZ and FEAST on senario1/senario2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `eval_recon_metrics_deepstarmap.py` | Standalone script | Unified per-gene reconstruction metrics (pearson/spearman/rmse/ssim_3d) for every method × scenario `test_results.h5ad` with ONE implementation (`NTF.helper_func.calculate_spatial_metrics`); summaries carry mean/median/std across genes                                                                                                                                                                                                                                                                                 |
| `run_svg_benchmark_deepstarmap.py`  | Standalone script | Moran's I corr + Top-K SVG Jaccard (K=5/10/20/50); sparse-X-aware h5py reads; shared sampling (seed=42, slice-stratified) and kNN graph (k=15) across methods                                                                                                                                                                                                                                                                                                                                                                    |
| `eval_celltyping_deepstarmap.py`    | Standalone script | Cell typing ARI/NMI/FMI vs GT `Harmony_labels` (22 classes): NTF/SpatialZ/UniST direct labels, FEAST clustering fallback; alignment check across methods; per-test-slice metrics (`celltyping_per_slice.csv`) with cross-slice std appended as `*_std` rows; spatial map figures with a globally consistent cell-type palette (clustering labels are renamed to their max-overlap GT category via Hungarian matching, so cluster colors line up with GT colors), per-slice ARI in panel titles, and a GT all-slices figure |
| `build_manifest_efficiency.py`      | Standalone script | Compile `manifest.csv` + `efficiency_summary.csv` from monitor JSONs and logs                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `deepstarmap_bm.R`                  | Standalone script | ggplot classic-style SVG summary figures; legends always below the panel (`theme_bm`)                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `plot_per_scenario.R`               | Standalone script | Per-scenario × per-metric compact bar plots (bm.R style: mean±std errorbar where available, no legend, 3.5×3 in) from `refine/deep_starmap_benchmark.xlsx`; 40 SVGs (recon/svg/celltyping × 5 scenarios), methods absent from a scenario are simply omitted                                                                                                                                                                                                                                                                |
| `plot_celltype_legend.py`           | Standalone script | Standalone legend SVG for the cell-typing figures (`--ncol` columns, default 3); replicates the exact sorted tab20/tab20b/tab20c category→color mapping of `eval_celltyping_deepstarmap.py` so legend colors match `gt_all_slices.png` / `spatial_maps_*.png`                                                                                                                                                                                                                                                           |
| `deep_starmap_viewer.py`            | Dash app          | Interactive 3D viewer (collapsible sidebar, hide-axis-background immersive mode, global/adaptive opacity, expression filters, multi-gene view, z-layer range slider): `python deep_starmap_viewer.py --port 8051`. Categorical colors use the same sorted tab20/tab20b/tab20c category→color mapping as `eval_celltyping_deepstarmap.py`, so viewer colors match the cell-typing figures                                                                                                                                    |

## Reproduction

```bash
source /home/gongyuqiao/miniconda3/etc/profile.d/conda.sh
conda activate ntf && cd /home/gongyuqiao/ur_annotation/NTF

# 0) Download the published deep_starmap_brain.h5ad into
#    mytrain/data/deep_starmap_brain/processed/  (see "Data" above)

# 1) NTF, 5 scenarios (dropout network ENABLED — the data is >95% zeros)
bash analysis/deep_starmap_brain/run_ntf.sh            # GPU=6 bash ... to pick a GPU

# 2) SpatialZ + FEAST, senario1/2 (envs spatialz / feast)
bash analysis/deep_starmap_brain/run_baselines.sh

# 3) Evaluation (env ntf)
python analysis/deep_starmap_brain/eval_recon_metrics_deepstarmap.py --output_dir <exp>/recon
python analysis/deep_starmap_brain/run_svg_benchmark_deepstarmap.py --output_dir <exp>/svg
python analysis/deep_starmap_brain/eval_celltyping_deepstarmap.py --output_dir <exp>/celltyping
python analysis/deep_starmap_brain/build_manifest_efficiency.py --exp_dir <exp>

# 4) Figures (R env with tidyverse)
cd <exp> && Rscript analysis/deep_starmap_brain/deepstarmap_bm.R
```
