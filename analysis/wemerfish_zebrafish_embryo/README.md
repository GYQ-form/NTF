# NTF Analysis Scripts — weMERFISH zebrafish embryo

Analysis scripts for the **weMERFISH zebrafish whole-embryo** benchmark (Wan et al. 2026; Dryad 10.5061/dryad.j0zpc86v9): **NTF vs SpatialZ vs FEAST vs UniST** on sample `C_6s_E2` (6-somite, the largest of the 6 measured embryos: 25,858 cells × 495 genes) over 5 prediction scenarios, covering reconstruction metrics (RMSE / 3D SSIM / Pearson / Spearman), SVG conservation (Moran's I corr, Top-K SVG Jaccard), cell typing (GT = `clusters`, 51 classes; ARI / NMI / FMI), **niche typing** (GT = `type`, 10 classes; accuracy plus per-niche composition fidelity: cosine / Jensen–Shannon divergence / niche-frequency correlation / TV distance) and computational efficiency (train + inference time only).

The full experiment record (commands, results, conclusions, deviations from plan) is in `mytrain/revision/experiment/wemerfish_benchmark/v1/README_results.md`.

## Data

Input: `weMERFISH_measured_C_6s_E2_rescaled_z.h5ad` (Dryad download) placed in `mytrain/data/wemerfish_zebrafish_embryo/`. Run `prepare_wemerfish.py` to produce the analysis-ready file:

```
mytrain/data/wemerfish_zebrafish_embryo/processed/wemerfish_C6s_E2.h5ad
```

- `X` = `normalize_total(1e4)` + `log1p` of the raw measured expression `obsm['X_raw']` (CSR sparse); `layers['raw']` = raw expression.
- `obsm['spatial']`: native whole-mount 3D coordinates (z rescaled to real sample thickness).
- `obs['slice_id']`: 16 **quantile-binned** virtual z-slices (1,616–1,617 cells each; equal-width binning is unusable because the embryo is a curved shell — the z-density is highly non-uniform). `obs['z']` = slice_id (for SpatialZ/FEAST `--z_col`); `obs['z_cont']` = continuous z.
- Annotation hierarchy in `obs`: `germlayer` (5) → `type` (10, niche ground truth) → `tissue` (39) → `clusters` (51, cell-type ground truth).
- Scenario split columns in `obs` (train/test; test slices in parentheses): `group` — senario1, interleaved holdout (2,4,6,8,10,12,14); `group2` — senario2, middle band (8,9,10); `group4` — senario3, both-end boundary extrapolation (1,2,15,16); `head` — head extrapolation (1,2); `tail` — tail extrapolation (15,16). senario3/head/tail are NTF-only (SpatialZ/FEAST/UniST cannot extrapolate).

## Overview

| File                                   | Type              | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `prepare_wemerfish.py`               | Standalone script | Build the processed h5ad: X_raw → normalize+log1p, quantile z-binning into 16 virtual slices, 5 scenario split columns, QC (shape, split purity, zero-fraction table)                                                                                                                                                                                                                                                                                                                                                   |
| `eval_nichetyping_wemerfish.py`      | Standalone script | Niche typing evaluation: (a) ARI/NMI/FMI of predicted niche labels vs GT `type`; (b) per-niche composition fidelity over the 51 GT `clusters` (and `tissue` as sensitivity): cosine similarity, JS divergence, niche-size Pearson/Spearman, TV distance. NTF/SpatialZ use direct labels; FEAST uses leiden fallback with Hungarian matching to GT niches (unmatched clusters flagged `unmatched::*` and excluded from means). Also writes the niche×cluster composition long table and niche-label spatial maps |
| `wemerfish_bm.R`                     | Standalone script | ggplot classic-style SVG summary figures; legends always below the panel; method colors NTF=#82659C, FEAST=#4883B1, SpatialZ=#5FA24D, UniST=#D95F02. Reads the csv tables produced by the evaluation scripts                                                                                                                                                                                                                                                                                                             |
| `build_summary_xlsx.py`              | Standalone script | Rebuilds `wemerfish_benchmark_summary.xlsx` (main + refine) from the eval csvs; appends UniST rows to the refine sheets without touching the existing refined values                                                                                                                                                                                                                                                                                                                                                   |
| `wemerfish_bm_refine.R`              | Standalone script | Refine-edition overview figures (read the refine xlsx; `position_dodge(preserve="single")` leaves empty slots for methods without results in a scenario)                                                                                                                                                                                                                                                                                                                                                               |
| `plot_celltype_legend_wemerfish.py`  | Standalone script | Renders the GT-clusters legend of the cell-typing spatial maps as a standalone SVG (`legend_gt_labels.svg`), replicating the exact tab20/tab20b/tab20c sorted-category palette of `eval_celltyping_deepstarmap.py`                                                                                                                                                                                                                                                                                                   |
| `validate_wemerfish_full_retrain.py` | Standalone script | Independently validates the five complete in-notebook training/inference runs, recomputes expression and label metrics, checks all 18 section-level ARI values and spatial-map exports, and writes the full-retrain Excel summary                                                                                                                                                                                                                                                                                        |
| `plot_per_scenario.R`                | Standalone script | Refine-edition per-scenario per-metric bar plots (5 scenarios x 11 metrics, recon with cross-gene std errorbars) plus one all-metrics figure per NTF-only scenario (adaptive ylim); same conventions as the deep_starmap_brain per-scenario figures                                                                                                                                                                                                                                                                      |
| `wemerfish_viewer.py`                | Dash app          | Interactive 3D viewer for all 6 weMERFISH embryos (sample dropdown, coordinate-set dropdown, X/X_raw toggle, z-range slider): `python wemerfish_viewer.py --port 8052`                                                                                                                                                                                                                                                                                                                                                 |

Reconstruction, SVG and cell-typing evaluation reuse the **shared parameterized scripts** in `../deep_starmap_brain/`:

- `eval_recon_metrics_deepstarmap.py --dataset wemerfish_C6s_E2` and `run_svg_benchmark_deepstarmap.py --dataset wemerfish_C6s_E2` — the `--dataset` switch selects the result-dir base name (default `deep_starmap_brain`);
- `eval_celltyping_deepstarmap.py --gt_key clusters` — the `--gt_key` switch selects the ground-truth obs column (default `Harmony_labels`).

These parameterizations were added for this benchmark and the public copies in `../deep_starmap_brain/` have been updated accordingly (the pre-parameterization originals are preserved in `analysis_private/deep_starmap_brain/`).

## Reproduction

```bash
source /home/gongyuqiao/miniconda3/etc/profile.d/conda.sh
conda activate ntf && cd /home/gongyuqiao/ur_annotation/NTF
EXP=mytrain/revision/experiment/wemerfish_benchmark   # any output dir works

# 0) Data preparation
python analysis/wemerfish_zebrafish_embryo/prepare_wemerfish.py

# 1) NTF, 5 scenarios (default config: bs=8192, n_iter=20000, dropout enabled)
CUDA_VISIBLE_DEVICES=6 python scripts/train_and_eval.py \
  -i mytrain/data/wemerfish_zebrafish_embryo/processed/wemerfish_C6s_E2.h5ad \
  -o mytrain/res/wemerfish_C6s_E2/<scenario> \
  --slice_id slice_id --split_mode label --split_column <group|group2|group4|head|tail> \
  --train_label train --test_label test --cat_keys clusters type

# 2) SpatialZ (env spatialz, cwd=/home/gongyuqiao/ur_annotation/SpatialZ), senario1/2,
#    twice per scenario: --cell_type_key clusters (output <scenario>) and
#    --cell_type_key type (output <scenario>_niche)
CUDA_VISIBLE_DEVICES=7 python train_ur.py -i <processed h5ad> -o res/wemerfish_C6s_E2/<scenario>[_niche] \
  --slice_id slice_id --group_key <group|group2> --z_col z --syn_mode fast --cell_type_key <clusters|type>

# 3) FEAST (env feast), senario1/2
CUDA_VISIBLE_DEVICES=7 python mytrain/scripts/FEAST/bm_feast.py -i <processed h5ad> \
  -o mytrain/res/FEAST/wemerfish_C6s_E2/<scenario> \
  --slice_id slice_id --group_key <group|group2> --z_col z --ot_method sinkhorn --ot_reg 0.05

# 4) Evaluation (env ntf)
python analysis/deep_starmap_brain/eval_recon_metrics_deepstarmap.py --output_dir $EXP/recon --dataset wemerfish_C6s_E2
python analysis/deep_starmap_brain/run_svg_benchmark_deepstarmap.py --output_dir $EXP/svg --dataset wemerfish_C6s_E2
python analysis/deep_starmap_brain/eval_celltyping_deepstarmap.py --output_dir $EXP/celltyping \
  --input_data mytrain/data/wemerfish_zebrafish_embryo/processed/wemerfish_C6s_E2.h5ad \
  --ntf_dir mytrain/res/wemerfish_C6s_E2 --feast_dir mytrain/res/FEAST/wemerfish_C6s_E2 \
  --spatialz_dir /home/gongyuqiao/ur_annotation/SpatialZ/res/wemerfish_C6s_E2 --gt_key clusters
python analysis/wemerfish_zebrafish_embryo/eval_nichetyping_wemerfish.py --output_dir $EXP/nichetyping

# 5) Summary tables + figures (env monocle3-r): copy the scripts next to the csv tables, then
cd $EXP && python build_summary_xlsx.py && /home/gongyuqiao/.conda/envs/monocle3-r/bin/Rscript wemerfish_bm.R
cd $EXP/refine && /home/gongyuqiao/.conda/envs/monocle3-r/bin/Rscript wemerfish_bm_refine.R \
  && /home/gongyuqiao/.conda/envs/monocle3-r/bin/Rscript plot_per_scenario.R
```

Note: the scripts in this directory are the final (v1, dropout-enabled) benchmark versions. A v2 variant of this benchmark (NTF `--no_dropout` ablation) exists only as an internal record — see `analysis_private/wemerfish_zebrafish_embryo/VERSION_NOTE.md` (not tracked by git).

## Complete in-notebook five-model retraining

`tutorial/02_wemerfish_scenarios.ipynb` includes the complete training process behind `NTF_RUN_TRAINING=1`: five independent 20,000-iteration NTF runs, test-coordinate inference, `clusters/type` kNN annotation and official evaluation, followed by all downstream analyses. Training mode rejects existing checkpoints/predictions and writes new models, full logs and tutorial outputs below `NTF_TUTORIAL_OUTPUT`. The verified development run used physical GPU 7 and is archived at `mytrain/revision/experiment/code_tutorial_notebooks/outputs/02_wemerfish_notebook_full_retrain/`.

```bash
# Use a fresh output directory when repeating the training.
CUDA_VISIBLE_DEVICES=7 NTF_RUN_TRAINING=1 NTF_DEVICE=cuda:0 \
  NTF_TUTORIAL_OUTPUT=<fresh-output-directory> \
  jupyter notebook tutorial/02_wemerfish_scenarios.ipynb
# Select Run All.

# Validate the archived development run and rebuild its Excel summary.
python analysis/wemerfish_zebrafish_embryo/validate_wemerfish_full_retrain.py
```
