# Mouse embryo (E16.5 Stereo-seq MOSTA) analysis scripts

Scripts for the R2 Technical comment 2.t8 revision experiment: comparing the
30-class cell-type distribution and 3D spatial structure between the original
E16.5 mouse embryo data (13 measured slices) and the NTF-reconstructed dense
volume. Full plan/record/results live in
`mytrain/revision/plan/R2_Tech8_mouse_embryo_ct_distribution_plan.md` and
`mytrain/revision/experiment/mouse_embryo_ct_distribution/`.

## Data

- Original: `mytrain/data/mouse_embryo/processed/E16.5_aligned_1030svg.h5ad`
  (1,414,048 cells x 1030 SVG; obs slice/annotation/z/z_scaled; obsm spatial)
- Reconstructed: `mytrain/data/mouse_embryo/generated/1000w-2/1000w.h5ad`
  (7,086,389 cells; obs annotation; obsm spatial; uns generation_info).
  Read in backed mode only (`obs.annotation`, `obsm.spatial`); never load the
  expression matrix (file is ~22 GB).

## Scripts

| File                           | Type              | Description                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------ | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `compare_ct_distribution.py` | Standalone script | 30-class counts/proportions original vs reconstructed, Pearson/Spearman/TV on proportions, generation_info dump, per-z cell-type frequency long table (original 13 slices vs reconstructed 65 z bins)                                                                                                                                                                                              |
| `plot_distribution.R`        | Standalone script | ggplot2 (theme_classic) figures, svg: proportion bar chart, log-log proportion scatter, per-z frequency curves                                                                                                                                                                                                                                                                                     |
| `render_3d_celltypes.py`     | Standalone script | matplotlib 3D, png: per-type original-vs-reconstructed panels (3 views) with a transparent embryo surface (marching cubes on the voxelized full point set) + target-organ scatter; density-honest sampling — original 1% of measured 1.4M vs reconstructed 10% of the 10M preview file (= 0.71% of the true 100M-scale reconstruction); all-30-type overview scatter at 0.2%/2% (same 1:10 ratio) |
| `make_summary_xlsx.py`       | Standalone script | Assemble `mouse_embryo_ct_distribution_summary.xlsx` from the CSVs                                                                                                                                                                                                                                                                                                                               |

## Run

```bash
cd mytrain/revision/experiment/mouse_embryo_ct_distribution   # or any output dir
PY=/home/gongyuqiao/.conda/envs/ntf/bin/python
R=/home/gongyuqiao/.conda/envs/monocle3-r/bin/Rscript
$PY compare_ct_distribution.py
$R plot_distribution.R          # ggrepel required
$PY render_3d_celltypes.py      # ~8 min, CPU
$PY make_summary_xlsx.py
```

Key results (2026-10-08): proportion Pearson r = 0.987, Spearman rho = 0.995,
TV distance = 0.049; reconstructed fills the 13-slice gaps into a continuous
volume while preserving per-type z profiles and 3D morphology.
