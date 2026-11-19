# cycif_crc1_3d —— t-CyCIF 结直肠癌 3D 图谱（CRC1）

数据：`mytrain/data/cycif_crc1_3d/`（Lin et al., Cell 2023, CRC1 标本 25 张 t-CyCIF 切片的单细胞 feature tables，详见数据目录 README）。

## 文件

### 数据处理与可视化

| 文件                         | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `preprocess_cycif_crc1.py` | 预处理：从 25 个 feature table zip 合成 `mytrain/data/cycif_crc1_3d/processed/cycif_crc1.h5ad`（X = 24 个 marker 原始强度 float32，obs = slice_id/slice/AREA/CIRC，obsm['spatial'] = (Xt, Yt, z_um) 像素+µm，obsm['spatial_um'] = 三轴均为 µm——x/y 按官方 OME-TIFF 元数据 0.65 µm/pixel 换算；z 由切片编号 × 厚度算出：#84 及以前 5 µm，#85 起 4 µm）。与 `mytrain/scripts/preprocess_cycif_crc1.py` 相同                                                                                                         |
| `annotate_cycif_crc1.py`   | 两级细胞注释：16 个分型 marker 的 GMM 初分 + 直方图人工调整门（log1p），按 Lin et al. Table S4 层次化布尔规则分配 `cell_type`（21 类）与 `cell_type_main`（9 类功能大类，niche 级），写回上述 h5ad                                                                                                                                                                                                                                                                                                                        |
| `run_maps_alignment.py`    | 切片间 x/y 配准：用 MAPS（[dbjzs/MAPS](https://github.com/dbjzs/MAPS)，本地 clone 在 `mytrain/software/MAPS`，ntf 环境 `pip install --no-deps -e .`）对 25 张切片做 sequential rigid 对齐（epochs=1000, sample_size=20000, enable_scale=False, seed=2026，GPU 6），输入为按切片拆分的 `spatial_um`（µm），输出 `obsm['spatial_aligned_um']`（x/y 已配准、z 不变）写回 `cycif_crc1.h5ad`。中间产物在 `mytrain/data/cycif_crc1_3d/processed/alignment/`（含逐切片导出、对齐后 npy、对照图 `alignment_check.png`） |
| `cycif_crc_viewer.py`      | 3D 交互 viewer（Dash，默认端口 8055），与 `mytrain/scripts/cycif_crc_viewer.py` 相同。坐标系下拉框默认 `spatial_aligned_um`（MAPS 配准后），可切回未配准的 `spatial_um`/`spatial`。注释后可用 `--obs-cols slice cell_type cell_type_main` 按注释着色；分类配色与 cell/niche typing 二维空间图一致（tab20+tab20b+tab20c，类别字典序映射，2026-10-19 起）。使用方式见 `analysis/3D_view.md`                                                                                                                         |

### Benchmark（v1 = 未配准坐标，v2 = MAPS 配准坐标，**以 v2 为准**）

| 文件                                                              | 说明                                                                                                                                                                                                       |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prepare_cycif_crc1_benchmark.py`                               | 生成 benchmark h5ad（全量 1276 万细胞：X = normalize_total+log1p，5 个 holdout 划分列）。`--coord-key spatial_um`（默认，v1）/ `spatial_aligned_um`（v2），另有 `--in-path/--out-path/--qc-dir`      |
| `run_ntf_cycif.sh`                                              | NTF 5 场景运行（关键配置：`--no_dropout --single_precision --weight_decay 0`，原因见 v1 实验记录 §5.1）。环境变量覆盖 `GPU`/`H5AD`/`RES_BASE`/`EXP`（默认值复现 v1）                            |
| `run_maps_alignment.py`                                         | 见上表「数据处理与可视化」                                                                                                                                                                                 |
| `run_feast_scaling_cycif.py` / `feast_scaling_extrapolate.py` | SpatialZ/FEAST 不可行的量化证据：FEAST 在 2k–20k 细胞/片的实测 scaling + 外推到全量（OT 矩阵 1.04 TB/片对）                                                                                               |
| `eval_recon_metrics_deepstarmap.py`                             | 重建指标统一评估（Pearson/Spearman/RMSE/3D SSIM；从 deep_starmap_brain 复制的参数化版本，`--dataset cycif_crc1`（v1）/ `cycif_crc1_aligned`（v2））                                                    |
| `run_svg_benchmark_deepstarmap.py`                              | SVG 保守性评估（Moran's I corr + Top-K Jaccard；同上参数化版本）                                                                                                                                           |
| `eval_celltyping_cycif.py`                                      | cell typing 评估（pred_cell_type vs cell_type，ARI/NMI/FMI + 空间图；NTF only）                                                                                                                            |
| `eval_nichetyping_cycif.py`                                     | niche 层面评估（pred_cell_type_main vs cell_type_main + 21 类组成保真度 cosine/JS/频率分布；改编自 weMERFISH 版）                                                                                          |
| `knn_transfer_ceiling_cycif.py`                                 | kNN(15) 直接转移上限（senario1 test：表达 Pearson 均值 + cell_type_main 标签转移 ARI），作为重建/注释指标的参照系；v1 = 0.234/0.192，v2 = 0.389/0.368                                                      |
| `build_manifest_efficiency_cycif.py`                            | manifest + 效率汇总（只计训练+推理时间）                                                                                                                                                                   |
| `summarize_cycif_crc1_benchmark.py`                             | 把一个版本目录下的 CSV 汇总为 xlsx（`--exp-dir .../v2`）                                                                                                                                                 |
| `cycifcrc_bm.R`                                                 | 全部统计图（ggplot classic 风格 svg，monocle3-r 环境，在某版本目录下运行）                                                                                                                                 |
| `plot_per_scenario_ntf_only.R`                                  | v2/refine 更新版汇总表的按 scenario 组合图（仅 NTF：Reconstruction/SVG conservation/Niche typing 三组紫系配色，ylim 自适应；在 v2 目录下运行 `Rscript refine/plot_per_scenario_ntf_only.R`，2026-10-19） |

## 复现

```bash
source /home/gongyuqiao/miniconda3/etc/profile.d/conda.sh && conda activate ntf
cd /home/gongyuqiao/ur_annotation/NTF
python mytrain/scripts/preprocess_cycif_crc1.py           # 生成 h5ad（约 1.2 GB，12,758,141 细胞 × 24 marker）
python mytrain/scripts/annotate_cycif_crc1.py             # 两级注释
python mytrain/data/cycif_crc1_3d/processed/alignment/run_maps_alignment.py  # MAPS 配准 → obsm['spatial_aligned_um']
python mytrain/scripts/prepare_cycif_crc1_benchmark.py \
  --coord-key spatial_aligned_um \
  --out-path mytrain/data/cycif_crc1_3d/processed/cycif_crc1_bm_aligned.h5ad \
  --qc-dir mytrain/revision/experiment/cycif_crc1_benchmark/v2   # v2 benchmark 数据（配准坐标；默认参数复现 v1）
GPU=6 H5AD=mytrain/data/cycif_crc1_3d/processed/cycif_crc1_bm_aligned.h5ad \
RES_BASE=mytrain/res/cycif_crc1_aligned \
EXP=mytrain/revision/experiment/cycif_crc1_benchmark/v2 \
bash mytrain/scripts/run_ntf_cycif.sh 20000               # NTF 5 场景（v2）
python mytrain/scripts/cycif_crc_viewer.py                # 3D viewer，http://<host>:8055
```
