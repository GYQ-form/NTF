# NTF Analysis Scripts — STARmap mouse VC

This directory contains analysis scripts for the **STARmap mouse visual cortex (mouse VC)** experiments. Currently it covers the R2 Major 5 (Comment 2.5) revision experiment: a **cell typing benchmark** on the 5-gene mouse VC dataset.

---

## Overview

| File                              | Type              | Description                                                                                                                                                                                                                                                             |
| --------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `eval_celltyping_mouseVC.py`    | Standalone script | Cell typing benchmark for NTF / FEAST / SpatialZ / UniST on the mouse VC 5-gene data: method-provided direct labels (or leiden clustering on FEAST's reconstructed expression) vs GT `leiden` via ARI/NMI/FMI, plus spatial label maps with per-slice ARI annotations |
| `eval_recon_metrics_mouseVC.py` | Standalone script | Unified per-gene reconstruction metrics (pearson/spearman/rmse/ssim_3d) for all methods x scenarios, one CSV per method (UniST included; same implementation as `deep_starmap_brain/eval_recon_metrics_deepstarmap.py`)                                               |
| `plot_unist_genes_mouseVC.py`   | Standalone script | UniST per-gene spatial expression panels (scanpy, same style as `scripts/mouseVC.ipynb`), writes `spatialUniST-<slice>.pdf` into `scripts/figures/<scenario>/`                                                                                                    |
| `test_in_silico_query.py`       | Standalone script | R2 Major 1 validation: smoke-test of the model-side in-silico query + occupancy filtering (`NTF/query.py`) on the mouse VC 28-gene data                                                                                                                               |
| `run_ablation_mousevc.sh`       | Shell script      | R1 Major 7: run the model-component ablation configs on the mouse VC 5-gene data (senario1 interpolation + senario3 extrapolation)                                                                                                                                      |

---

## Revision experiments

### cell typing benchmark (`eval_celltyping_mouseVC.py`)

Evaluates cell typing on held-out test slices of `mytrain/data/mouse_VC/processed/mouseVC_5gene.h5ad` (32,845 cells × 5 genes, GT = obs `leiden`, 19 classes from full-transcriptome clustering). Each method contributes one label set per scenario, compared with GT via **ARI / NMI / FMI** (sklearn):

- **NTF → direct labels** `obs['pred_leiden']` (built-in kNN annotator, `--cat_keys leiden`);
- **SpatialZ → direct labels** `obs['pred_label']` (native label transfer);
- **UniST → direct labels** `obs['pred_label']` (official API label output via `scripts/UniST/bm_unist.py --label_key leiden`; senario1/2 only — UniST interpolates between flanking train slices and does not extrapolate);
- **FEAST → clustering labels** — FEAST has no direct-label capability, so leiden clustering (scanpy `neighbors` n_neighbors=15 on the 5-dim expression, no PCA; `leiden` resolution=1.0, seed=42) on its `layers['prediction']` is used instead.

senario1 (split column `group`, test = slice_2,4,6) and senario2 (`group2`, test = slice_3,4,5) are evaluated for all three methods after an alignment check (cell count, spatial coordinates, GT labels; aborts on failure). senario3 (`group4`, test = slice_1,7), head (`head`, test = slice_1,2) and tail (`tail`, test = slice_6,7) are NTF-only extrapolation scenarios. Missing result files are skipped with a warning.

#### Reproduction

```bash
# 1) NTF training with built-in cell typing (per scenario, GPU 7)
CUDA_VISIBLE_DEVICES=7 python scripts/train_and_eval.py \
    -i mytrain/data/mouse_VC/processed/mouseVC_5gene.h5ad \
    -o mytrain/res/mouseVC/mouseVC_5gene/<scenario> \
    --slice_id slice_id --split_mode label --split_column <split列> \
    --train_label train --test_label test \
    --cat_keys leiden --no_dropout
# <scenario>/<split列>: senario1/group, senario2/group2, senario3/group4, head/head, tail/tail

# 2) SpatialZ rerun with real label transfer (conda env spatialz, GPU 6)
cd /home/gongyuqiao/ur_annotation/SpatialZ
CUDA_VISIBLE_DEVICES=6 python train_ur.py \
    -i /home/gongyuqiao/ur_annotation/NTF/mytrain/data/mouse_VC/processed/mouseVC_5gene.h5ad \
    -o res/mouseVC/5gene/<scenario> \
    --group_key <group|group2> --cell_type_key leiden
# train_ur.py carries a wrapper-level patch that maps syn_adata.obs[cell_type_key]
# back to test coordinates with the same KNN indices and writes obs['pred_label'].

# 3) FEAST: no rerun (no direct-label capability); existing results under
#    mytrain/res/FEAST/mouseVC/senario{1,2}/ are reused.

# 3b) UniST with label output (conda env unist, GPU 6; senario1/2 only)
CUDA_VISIBLE_DEVICES=6 env PYTHONNOUSERSITE=1 /data/gongyq/conda_data/envs/unist/bin/python \
    mytrain/scripts/UniST/bm_unist.py \
    -i mytrain/data/mouse_VC/processed/mouseVC_5gene.h5ad \
    -o mytrain/res/UniST/mouseVC_5gene/<scenario> \
    --mode inr --split_column <group|group2> --label_key leiden

# 3c) Unified per-gene reconstruction metrics (all methods; conda env ntf)
python analysis/STARmap_mouse_VC/eval_recon_metrics_mouseVC.py \
    --output_dir mytrain/revision/experiment/cell_typing_benchmark/mouseVC

# 4) Evaluation (conda env ntf)
python analysis/STARmap_mouse_VC/eval_celltyping_mouseVC.py \
    --output_dir mytrain/revision/experiment/cell_typing_benchmark

# 5) Bar plots (R/ggplot2, monocle3-r env, base svg device)
cd mytrain/revision/experiment/cell_typing_benchmark
/home/gongyuqiao/.conda/envs/monocle3-r/bin/Rscript celltyping_bm.R
```

#### Arguments

| Argument                | Default                                                       | Description                                                       |
| ----------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------- |
| `--input_data`        | `mytrain/data/mouse_VC/processed/mouseVC_5gene.h5ad`        | Input dataset (used to cross-check split columns and test slices) |
| `--output_dir`        | (required)                                                    | Output directory                                                  |
| `--ntf_dir`           | `mytrain/res/mouseVC/mouseVC_5gene`                         | NTF results root (`<scenario>/test_results.h5ad`)               |
| `--feast_dir`         | `mytrain/res/FEAST/mouseVC`                                 | FEAST results root                                                |
| `--spatialz_dir`      | `/home/gongyuqiao/ur_annotation/SpatialZ/res/mouseVC/5gene` | SpatialZ results root                                             |
| `--unist_dir`         | `mytrain/res/UniST/mouseVC_5gene`                           | UniST results root                                                |
| `--leiden_resolution` | 1.0                                                           | Leiden resolution for FEAST's clustering labels                   |
| `--seed`              | 42                                                            | Random seed                                                       |

#### Outputs

- `celltyping_metrics.csv` – long table: `method, scenario, label_source (direct/clustering), metric (ari/nmi/fmi), value, status`
- `manifest.csv` – result file path, method, scenario, n_obs, split column, label source, alignment-check status
- `figures/spatial_maps_<scenario>.png` – one spatial map per scenario (5 total): GT + method labels faceted by test slices, each method panel annotated with its per-slice ARI (senario1/2: GT, NTF pred_leiden, SpatialZ pred_label, FEAST clustering, UniST pred_label; senario3/head/tail: GT, NTF pred_leiden)

Statistical bar plots are generated separately: `celltyping_bm.R` (in the experiment directory; reads `celltyping_summary.xlsx`) produces `figures/plot_senario1.svg`, `figures/plot_senario2.svg` and `figures/plot_NTF_extrapolation.svg`; `refine/plot_per_scenario.R` (reads `refine/celltyping_mouseVC.xlsx`, built by `refine/build_benchmark_xlsx.py` from the per-method recon CSVs + `celltyping_metrics.csv`) produces one figure per scenario x metric for senario1/2 and one NTF-only multi-metric figure for senario3/head/tail (`refine/plot_*.svg`). UniST per-gene spatial expression panels are produced by `plot_unist_genes_mouseVC.py` into `scripts/figures/senario{1,2}/spatialUniST-<slice>.pdf`.

Results and full record: `mytrain/revision/experiment/cell_typing_benchmark/` (see `README_results.md` there; the summary workbook `celltyping_summary.xlsx` is derived from `celltyping_metrics.csv`).

### in-silico query validation (`test_in_silico_query.py`)

Validates the model-side in-silico querying feature (`NTF/query.py`, added during revision): dense query spots at user-chosen resolution are predicted by the model, and tissue occupancy is decided from the predicted total expression (auto Otsu threshold) and mean dropout probability, with no pre-estimated gap coordinates and no geometric tissue mask. Trains a smoke-test NTF (2,000 iterations, dropout network enabled) on `mytrain/data/mouse_VC/processed/mouseVC_28gene.h5ad`, queries 307,510 grid spots spanning the full bounding box, and asserts that occupied spots fall on real tissue and that dropout confidence separates occupied from filtered spots.

```bash
CUDA_VISIBLE_DEVICES=6 python analysis/STARmap_mouse_VC/test_in_silico_query.py \
    --output_dir mytrain/revision/experiment/r2m1_insilico_query
```

Results and full record: `mytrain/revision/experiment/r2m1_insilico_query/` (see `README_results.md` there).

### model-component ablation (`run_ablation_mousevc.sh`)

mouseVC side of the R1M7 ablation study (simulation side and shared summary/plot scripts live in `analysis/simulation/`). Runs 9 ablation configs (`--no_dropout` baseline plus single-component removals and a bias-correction-on config) on `mytrain/data/mouse_VC/processed/mouseVC_5gene.h5ad` under two split scenarios — senario1 (`group`, interpolation) and senario3 (`group4`, two-sided boundary extrapolation) — plus one dropout-enabled diagnostic run. Results are written to `mytrain/res/mouseVC/ablation/<config>/<scenario>/`.

```bash
GPU=7 bash analysis/STARmap_mouse_VC/run_ablation_mousevc.sh
```

Results and full record: `mytrain/revision/experiment/r1m7_model_ablation/` (see `README_results.md` there).
