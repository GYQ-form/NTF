# %% [markdown]
# # Tutorial 1 — Sparse 2D sections to a high-resolution 3D field
#
# This tutorial covers the complete NTF workflow: start from the full mouse-brain
# simulation, remove non-spatial housekeeping genes, aggregate selected input
# sections into low-resolution bins, train NTF, and query expression at every
# original high-resolution coordinate.
#
# ![Sparse, low-resolution sections to a high-resolution 3D field](../figures/super_resolution.jpg)
#
# *NTF learns a continuous 3D field from sparse, low-resolution 2D sections and
# supports sectioning at high-resolution coordinates.*
#
# By default, the notebook reads the complete existing training input and
# reconstruction so that it can be inspected without repeating training. Set
# `NTF_RUN_TRAINING=1` to execute the full preprocessing → resolution reduction →
# training → reconstruction workflow. No cells are randomly sampled, no reference
# sections are removed from the source data, and no points are thinned for plots.

# %%
from pathlib import Path
import os
import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from IPython.display import display

candidates = ([Path(os.environ["NTF_ROOT"])] if os.getenv("NTF_ROOT") else []) + [Path.cwd(), *Path.cwd().parents]
ROOT = next((p.resolve() for p in candidates if (p / "NTF" / "train.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Set NTF_ROOT to the cloned NTF repository root.")
sys.path.insert(0, str(ROOT))

RAW_SIMULATION_PATH = Path(os.getenv(
    "NTF_SIMULATION_RAW",
    ROOT / "mytrain/res/simulation/domain_interval/ground_truth_full.h5ad",
))
SIMULATION_PATH = Path(os.getenv(
    "NTF_SIMULATION_INPUT",
    ROOT / "mytrain/data/simulation/ground_truth_full_no_HK.h5ad",
))
TRAIN_PATH = Path(os.getenv(
    "NTF_SIMULATION_TRAIN",
    ROOT / "mytrain/res/simulation/mixed_resolution-1/train_m30_bf8.0.h5ad",
))
RECONSTRUCTION_PATH = Path(os.getenv(
    "NTF_SIMULATION_RECONSTRUCTION",
    ROOT / "mytrain/res/simulation/mixed_resolution-1/rec_m30_bf8.0_pred.h5ad",
))
GROUND_TRUTH_PATH = Path(os.getenv(
    "NTF_SIMULATION_TRUTH",
    ROOT / "mytrain/res/simulation/mixed_resolution-1/rec_m30_bf8.0.h5ad",
))
OUTPUT_DIR = Path(os.getenv("NTF_TUTORIAL_OUTPUT", ROOT / "tutorial/outputs/01_simulation"))
REPRODUCTION_DIR = OUTPUT_DIR / "reproduction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_PREPROCESSING = os.getenv("NTF_RUN_PREPROCESSING", "0") == "1"
RUN_TRAINING = os.getenv("NTF_RUN_TRAINING", "0") == "1"
TRAINING_ITERATIONS = int(os.getenv("NTF_TRAINING_ITERATIONS", "20000"))
TRAINING_DEVICE = os.getenv("NTF_DEVICE", "cuda:0")

PLOT_SLICE_IDS = ["35", "45", "55"]
PLOT_GENE = "Gene_12_Diff"
SLICE_INTERVAL_K = 5
HIGH_RES_INTERVAL_M = 30
BIN_FACTOR = 8.0

for path in [SIMULATION_PATH, TRAIN_PATH, RECONSTRUCTION_PATH, GROUND_TRUTH_PATH]:
    if not path.is_file() and not (RUN_PREPROCESSING and path == SIMULATION_PATH):
        raise FileNotFoundError(f"Required simulation file not found: {path}\nSee tutorial/README.md for data preparation.")

print(f"HK-free starting data: {os.path.relpath(SIMULATION_PATH, ROOT)}")
print(f"Cached training input: {os.path.relpath(TRAIN_PATH, ROOT)}")
print(f"Cached reconstruction: {os.path.relpath(RECONSTRUCTION_PATH, ROOT)}")
print(f"Run full training workflow: {RUN_TRAINING}")
print(f"Output tables: {os.path.relpath(OUTPUT_DIR, ROOT)}")

# %% [markdown]
# ## 1. Prepare the starting data without housekeeping genes
#
# The original simulation has 1,975,530 cells, 100 sections and 100 genes. The
# last ten genes (`Gene_90_HK`–`Gene_99_HK`) have no spatial expression pattern.
# The tutorial starting file retains every cell, every section and all 90 `Diff`
# genes. The filtering operation is shown below and can be repeated with
# `NTF_RUN_PREPROCESSING=1`; it is skipped when the prepared file already exists.

# %%
def prepare_simulation_without_hk(raw_path, output_path):
    raw_data = ad.read_h5ad(raw_path)
    keep = ~raw_data.var_names.str.endswith("_HK")
    removed = raw_data.var_names[~keep].tolist()
    filtered = raw_data[:, keep].copy()
    filtered.uns["source_file"] = raw_path.name
    filtered.uns["removed_HK_genes"] = removed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_h5ad(output_path, compression="gzip")
    print(f"Removed {len(removed)} HK genes: {removed}")
    print(f"Saved {filtered.shape[0]:,} cells × {filtered.shape[1]} genes to {output_path}")
    return filtered


if RUN_PREPROCESSING:
    simulation_data = prepare_simulation_without_hk(RAW_SIMULATION_PATH, SIMULATION_PATH)

simulation_source = ad.read_h5ad(SIMULATION_PATH, backed="r")
assert simulation_source.shape == (1975530, 90)
assert not simulation_source.var_names.str.endswith("_HK").any()
assert set(simulation_source.obs["slice_id"].astype(str)) == {str(i) for i in range(100)}
print(f"Starting data: {simulation_source.n_obs:,} cells × {simulation_source.n_vars} spatially patterned genes")
print(f"Reference sections retained: {simulation_source.obs['slice_id'].nunique()}")
simulation_source.file.close()

# %% [markdown]
# ## 2. Reduce the resolution of the observed input sections
#
# `create_mixed_resolution_dataset` observes every fifth section (`k=5`), yielding
# 20 input sections. With `m=30`, none of these sections reaches the high-resolution
# interval, so all observed sections are aggregated into spatial bins with
# `bin_factor=8`.
# Aggregation averages expression and computes bin centroids; it does not randomly
# sample cells. The 80 unobserved sections remain intact in `simulation_data` and
# serve as high-resolution query coordinates and ground truth.
#
# This cell runs as part of `NTF_RUN_TRAINING=1`. The complete input is also saved
# before model fitting. Coordinates are scaled by the same bounding-box factor for
# training and reconstruction.

# %%
def build_mixed_resolution_input(simulation_data):
    from analysis.simulation.resolution_utils import create_mixed_resolution_dataset

    raw_bbox_size = np.ptp(simulation_data.obsm["spatial"], axis=0).max()
    spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
    mixed = create_mixed_resolution_dataset(
        simulation_data,
        slice_interval_k=SLICE_INTERVAL_K,
        high_res_interval_m=HIGH_RES_INTERVAL_M,
        bin_factor=BIN_FACTOR,
    )
    mixed.obsm["spatial"] = np.asarray(mixed.obsm["spatial"]) * spatial_scaling
    return mixed, spatial_scaling


if RUN_TRAINING:
    if "simulation_data" not in globals():
        simulation_data = ad.read_h5ad(SIMULATION_PATH)
    reproduced_train_data, spatial_scaling = build_mixed_resolution_input(simulation_data)
    REPRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    reproduced_train_path = REPRODUCTION_DIR / "train_m30_bf8.0_no_HK.h5ad"
    reproduced_train_data.write_h5ad(reproduced_train_path, compression="gzip")
    print(f"Training input: {reproduced_train_data.shape[0]:,} bins/cells × {reproduced_train_data.n_vars} genes")
    print(f"Observed sections: {reproduced_train_data.obs['slice_id'].nunique()}; source sections: {simulation_data.obs['slice_id'].nunique()}")
    print(f"Saved complete training input to {reproduced_train_path}")

# %% [markdown]
# ## 3. Train NTF and reconstruct all high-resolution coordinates
#
# The training call below uses the same `NTF.train` API as the benchmark workflow.
# The default is the full 20,000-iteration run; `NTF_TRAINING_ITERATIONS` and
# `NTF_DEVICE` can override the iteration count and device. The progress bar uses
# `tqdm.auto` with dynamic width and a one-second refresh interval, so updates
# extend one continuous bar instead of filling the notebook with short bars.
#
# After fitting, `sample_points` queries **all** coordinates from all 100 source
# sections. The output retains the original 90-gene `X` as ground truth and stores
# NTF expression in `layers['prediction']`. This potentially long cell is skipped
# during the default cached-result walkthrough.

# %%
if RUN_TRAINING:
    import torch
    from NTF.config import add_shared_args, process_args
    from NTF.sample import sample_points
    from NTF.train import train

    parser = add_shared_args(description="Tutorial mixed-resolution NTF reconstruction")
    args = parser.parse_args([
        "--input_data", str(SIMULATION_PATH),
        "--output_dir", str(REPRODUCTION_DIR),
        "--log_dir", str(REPRODUCTION_DIR / "runs"),
        "--slice_id", "slice_id",
        "--n_iter", str(TRAINING_ITERATIONS),
        "--device", TRAINING_DEVICE,
    ])
    args = process_args(args)
    model_wrapper = train(reproduced_train_data, args)

    query_coordinates = torch.from_numpy(
        np.asarray(simulation_data.obsm["spatial"]) * spatial_scaling
    ).float().to(args.device)
    prediction = sample_points(model_wrapper.inr, query_coordinates, batch_size=8192)
    reproduced_reconstruction = simulation_data.copy()
    reproduced_reconstruction.layers["prediction"] = prediction["expression"].cpu().numpy()
    reproduced_reconstruction_path = REPRODUCTION_DIR / "rec_m30_bf8.0_no_HK.h5ad"
    reproduced_reconstruction.write_h5ad(reproduced_reconstruction_path, compression="gzip")
    print(f"Saved reconstruction with {reproduced_reconstruction.n_obs:,} cells and {reproduced_reconstruction.obs['slice_id'].nunique()} sections")
    print(f"Reconstruction path: {reproduced_reconstruction_path}")

# %% [markdown]
# ## 4. Load the complete existing input and reconstruction
#
# For a fast walkthrough, the remaining cells use the existing complete
# mixed-resolution run. These cached files contain the original 100 genes; the
# metric table below excludes the ten HK genes to match the new 90-gene starting
# data. The qualitative `Gene_12_Diff` comparison is unaffected.
#
# The low-resolution input contains all 12,495 observed bins/cells from 20 input
# sections. Ground truth and prediction contain all 1,975,530 cells from all 100
# sections. The `_pred` file stores predictions in both `X` and
# `layers['prediction']`, so truth is read from the companion file whose original
# `X` remains intact.

# %%
train_data = ad.read_h5ad(TRAIN_PATH)
ground_truth = ad.read_h5ad(GROUND_TRUTH_PATH, backed="r")
reconstructed = ad.read_h5ad(RECONSTRUCTION_PATH, backed="r")
train_data.obs["slice_id"] = train_data.obs["slice_id"].astype(str)
train_slices = set(train_data.obs["slice_id"])
reference_slices = set(ground_truth.obs["slice_id"].astype(str))

assert train_data.shape == (12495, 100)
assert ground_truth.shape == reconstructed.shape == (1975530, 100)
assert ground_truth.obs_names.equals(reconstructed.obs_names)
assert np.array_equal(ground_truth.obsm["spatial"], reconstructed.obsm["spatial"])
assert len(train_slices) == 20
assert reference_slices == {str(i) for i in range(100)}
assert set(PLOT_SLICE_IDS).issubset(train_slices)
assert PLOT_GENE in train_data.var_names and PLOT_GENE in ground_truth.var_names and PLOT_GENE in reconstructed.var_names

print(f"Cached training input: {train_data.n_obs:,} bins/cells × {train_data.n_vars} genes, {len(train_slices)} observed sections")
print(f"Cached reconstruction: {reconstructed.n_obs:,} cells × {reconstructed.n_vars} genes, {len(reference_slices)} sections")
display(pd.DataFrame({
    "dataset": ["HK-free tutorial starting data", "Low-resolution cached NTF input", "High-resolution reference and cached NTF reconstruction"],
    "observations": [1975530, train_data.n_obs, reconstructed.n_obs],
    "genes": [90, train_data.n_vars, reconstructed.n_vars],
    "sections": [100, len(train_slices), len(reference_slices)],
    "cell_or_bin_filtering": ["HK genes only", "none", "none"],
}))

# %% [markdown]
# ## 5. Report metrics for the 90 spatially patterned genes
#
# The stored metrics were calculated with every cell. Only the ten non-spatial HK
# genes are excluded from this table; no observations or sections are filtered.

# %%
spatial_gene_mask = ~reconstructed.var_names.str.endswith("_HK")
gene_metrics = reconstructed.var.loc[spatial_gene_mask, ["pearson_corr", "spearman_corr", "rmse", "ssim_3d"]].copy()
gene_metrics.index.name = "gene"
gene_metrics = gene_metrics.reset_index()
summary = gene_metrics[["pearson_corr", "spearman_corr", "rmse", "ssim_3d"]].mean().to_frame().T
summary.insert(0, "evaluation", f"90 Diff genes; all {reconstructed.n_obs:,} cells; all {len(reference_slices)} sections")
gene_metrics.to_csv(OUTPUT_DIR / "gene_metrics.csv", index=False)
summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
display(gene_metrics.round(3))
display(summary.round(3))

gene_index = reconstructed.var_names.get_loc(PLOT_GENE)
reconstruction_gene = ad.AnnData(
    X=np.asarray(ground_truth.X[:, gene_index]).reshape(-1, 1).astype(np.float32),
    obs=pd.DataFrame(
        {"slice_id": ground_truth.obs["slice_id"].astype(str).to_numpy()},
        index=ground_truth.obs_names.copy(),
    ),
)
reconstruction_gene.var_names = [PLOT_GENE]
reconstruction_gene.layers["prediction"] = np.asarray(
    reconstructed.layers["prediction"][:, gene_index]
).reshape(-1, 1).astype(np.float32)
reconstruction_gene.obsm["spatial"] = np.asarray(reconstructed.obsm["spatial"])
ground_truth.file.close()
reconstructed.file.close()
assert reconstruction_gene.n_obs == 1975530
assert set(reconstruction_gene.obs["slice_id"]) == {str(i) for i in range(100)}

# %% [markdown]
# ## 6. Compare complete 2D sections
#
# Sections 35, 45 and 55 show every high-resolution reference cell, every actual
# low-resolution input bin and every reconstructed cell for `Gene_12_Diff`.
# `plot_sub_slice` follows the mixed-resolution analysis: `scanpy.pl.embedding`
# determines point size automatically. All panels use `Blues` and one shared
# expression range. The colorbar is qualitative (`Low`–`High`) because the
# absolute log-normalized values are not meaningful across datasets.

# %%
train_values = np.asarray(train_data[:, PLOT_GENE].X).ravel()
truth_values = np.asarray(reconstruction_gene.X).ravel()
prediction_values = np.asarray(reconstruction_gene.layers["prediction"]).ravel()
expression_min = min(train_values.min(), truth_values.min(), prediction_values.min())
expression_max = max(train_values.max(), truth_values.max(), prediction_values.max())
expression_norm = plt.Normalize(expression_min, expression_max)


def add_low_high_colorbar(fig, rect):
    colorbar_ax = fig.add_axes(rect)
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=expression_norm, cmap="Blues"),
        cax=colorbar_ax,
        label="Expression",
    )
    colorbar.set_ticks([expression_min, expression_max], labels=["Low", "High"])


def plot_sub_slice(adata, slice_id, gene_name, layer=None, show=False, ax=None, title=None):
    subset = adata[(adata.obs["slice_id"] == slice_id)].copy()
    sc.pl.embedding(
        subset,
        basis="spatial",
        color=gene_name,
        title=title or f"{PLOT_GENE} - {slice_id}",
        layer=layer,
        show=show,
        ax=ax,
        cmap="Blues",
        vmin=expression_min,
        vmax=expression_max,
        colorbar_loc=None,
    )


fig, axes = plt.subplots(len(PLOT_SLICE_IDS), 3, figsize=(13, 12))
for row, sid in enumerate(PLOT_SLICE_IDS):
    columns = [
        (reconstruction_gene, None, "High-resolution truth"),
        (train_data, None, "Low-resolution NTF input"),
        (reconstruction_gene, "prediction", "NTF reconstruction"),
    ]
    for col, (data, layer, title) in enumerate(columns):
        plot_sub_slice(
            data,
            sid,
            PLOT_GENE,
            layer=layer,
            ax=axes[row, col],
            title=f"{PLOT_GENE} - {sid}\n{title}",
        )
fig.subplots_adjust(left=0.03, right=0.90, bottom=0.03, top=0.94, wspace=0.12, hspace=0.22)
add_low_high_colorbar(fig, [0.93, 0.20, 0.015, 0.60])
display(fig)
plt.close(fig)

# %% [markdown]
# ## 7. Compare the complete volumes from three views
#
# The low-resolution input is included alongside the high-resolution truth and
# NTF reconstruction. Every panel uses all observations from its dataset—there
# is no stride, random sampling or thinning. All maps share the same `Blues`
# expression range and a `Low`–`High` colorbar. Only the sparse input row uses
# enlarged points; the high-resolution rows retain the plotting defaults. Extra
# vertical space keeps the three rows from occluding one another.

# %%
views = [
    (20, 45, "View 1 (elev=20°, azim=45°)"),
    (20, 135, "View 2 (elev=20°, azim=135°)"),
    (20, 315, "View 3 (elev=20°, azim=315°)"),
]
volume_data = [
    (train_data, None, "Low-resolution NTF input", True),
    (reconstruction_gene, None, "High-resolution truth", False),
    (reconstruction_gene, "prediction", "NTF reconstruction", False),
]
fig, axes = plt.subplots(3, 3, figsize=(16, 18), subplot_kw={"projection": "3d"})
for row, (data, layer, data_title, enlarge_points) in enumerate(volume_data):
    for col, (elev, azim, view_title) in enumerate(views):
        ax = axes[row, col]
        sc.pl.embedding(
            data,
            basis="spatial",
            color=PLOT_GENE,
            layer=layer,
            projection="3d",
            title=f"{data_title}\n{view_title}",
            ax=ax,
            show=False,
            cmap="Blues",
            vmin=expression_min,
            vmax=expression_max,
            colorbar_loc=None,
        )
        if enlarge_points:
            ax.collections[0].set_sizes([72.0])
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(np.ptp(data.obsm["spatial"], axis=0))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
fig.suptitle(f"{PLOT_GENE}: complete input and reconstruction from three views", fontsize=15)
fig.subplots_adjust(left=0.02, right=0.90, bottom=0.04, top=0.94, wspace=0.08, hspace=0.38)
add_low_high_colorbar(fig, [0.93, 0.20, 0.015, 0.60])
display(fig)
plt.close(fig)

print("Tutorial 1 finished successfully.")
