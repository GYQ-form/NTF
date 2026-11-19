# %% [markdown]
# # Tutorial 4 — Full mouse-brain simulation rerun
#
# This notebook reruns the complete NTF mixed-resolution workflow from
# `mytrain/res/simulation/domain_interval/ground_truth_full.h5ad`. It does **not** load an
# existing training input, reconstruction, checkpoint or metric table.
#
# ![Sparse, low-resolution sections to a high-resolution 3D field](../figures/super_resolution.jpg)
#
# The run retains all 1,975,530 cells, all 100 sections and all 100 genes:
# 90 spatially patterned Diff genes plus 10 non-spatial HK noise controls. Overall
# performance is summarised only over the 90 Diff genes, consistent with the paper.
# The only resolution reduction is the spatial bin aggregation used to construct
# the low-resolution measurements. There is no random cell sampling,
# reference-section deletion or point thinning for visualisation. Training uses
# 20,000 requested iterations and the model's default early stopping, followed by
# prediction at every original high-resolution coordinate.

# %%
from pathlib import Path
import json
import os
import sys
import time

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from IPython.display import display
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity

candidates = ([Path(os.environ["NTF_ROOT"])] if os.getenv("NTF_ROOT") else []) + [Path.cwd(), *Path.cwd().parents]
ROOT = next((p.resolve() for p in candidates if (p / "NTF" / "train.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Set NTF_ROOT to the cloned NTF repository root.")
sys.path.insert(0, str(ROOT))

try:
    import mpmath
except ImportError:
    mpmath_wheel = ROOT / "mytrain/revision/experiment/simulation_mousebrain_100gene_rerun/environment/mpmath-1.3.0-py3-none-any.whl"
    if not mpmath_wheel.is_file():
        raise
    for module_name in tuple(sys.modules):
        if module_name == "mpmath" or module_name.startswith("mpmath."):
            del sys.modules[module_name]
    sys.path.insert(0, str(mpmath_wheel))
    import mpmath
    print(f"Using bundled mpmath {mpmath.__version__}: {os.path.relpath(mpmath_wheel, ROOT)}")

INPUT_PATH = Path(os.getenv(
    "NTF_SIMULATION_MOUSEBRAIN",
    ROOT / "mytrain/res/simulation/domain_interval/ground_truth_full.h5ad",
)).resolve()
OUTPUT_DIR = Path(os.getenv(
    "NTF_TUTORIAL_OUTPUT",
    ROOT / "tutorial/outputs/04_simulation_mousebrain_full_rerun",
)).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not INPUT_PATH.is_file():
    raise FileNotFoundError(f"Simulation input not found: {INPUT_PATH}")
if not torch.cuda.is_available():
    raise RuntimeError("This full training tutorial requires a CUDA-capable GPU.")

DEVICE = "cuda:0"
N_ITER = 20000
SLICE_INTERVAL_K = 5
HIGH_RES_INTERVAL_M = 30
BIN_FACTOR = 8.0
QUERY_BATCH_SIZE = 8192
PLOT_SLICE_IDS = ["35", "45", "55"]
PLOT_GENE = "Gene_12_Diff"
run_start = time.time()
run_config = {
    "input_data": os.path.relpath(INPUT_PATH, ROOT),
    "output_dir": os.path.relpath(OUTPUT_DIR, ROOT),
    "device": DEVICE,
    "requested_training_iterations": N_ITER,
    "slice_interval_k": SLICE_INTERVAL_K,
    "high_res_interval_m": HIGH_RES_INTERVAL_M,
    "bin_factor": BIN_FACTOR,
    "query_batch_size": QUERY_BATCH_SIZE,
    "cached_results_used": False,
    "random_cell_sampling": False,
    "reference_sections_removed": False,
    "plot_point_thinning": False,
    "expression_normalization_for_plots": False,
    "3d_point_alpha": 0.35,
}
(OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

print(f"Input: {os.path.relpath(INPUT_PATH, ROOT)}")
print(f"New outputs: {os.path.relpath(OUTPUT_DIR, ROOT)}")
print(f"Device: {DEVICE}; requested iterations: {N_ITER:,}")
print("Cached training inputs, reconstructions, checkpoints and metrics are not used.")

# %% [markdown]
# ## 1. Load and verify the complete starting data
#
# `X` contains normalize-total + log1p expression. The source object remains in
# memory throughout input construction and full-coordinate reconstruction so all
# unobserved sections remain available as ground truth.

# %%
simulation_data = ad.read_h5ad(INPUT_PATH)
simulation_data.obs["slice_id"] = simulation_data.obs["slice_id"].astype(str)
assert simulation_data.shape == (1975530, 100)
assert simulation_data.obs["slice_id"].nunique() == 100
assert set(simulation_data.obs["slice_id"]) == {str(i) for i in range(100)}
assert simulation_data.var_names[:90].str.endswith("_Diff").all()
assert simulation_data.var_names[90:].str.endswith("_HK").all()
assert simulation_data.var_names[90:].tolist() == [f"Gene_{i}_HK" for i in range(90, 100)]
assert simulation_data.obsm["spatial"].shape == (1975530, 3)
assert np.isfinite(np.asarray(simulation_data.X)).all()
assert np.isfinite(np.asarray(simulation_data.obsm["spatial"])).all()
assert PLOT_GENE in simulation_data.var_names

raw_coordinates = np.asarray(simulation_data.obsm["spatial"])
raw_bbox_size = np.ptp(raw_coordinates, axis=0).max()
spatial_scaling = 1.0 / raw_bbox_size if raw_bbox_size > 0 else 1.0
run_config.update({
    "input_observations": int(simulation_data.n_obs),
    "input_genes": int(simulation_data.n_vars),
    "input_sections": int(simulation_data.obs["slice_id"].nunique()),
    "spatial_scaling": float(spatial_scaling),
})
print(f"Loaded {simulation_data.n_obs:,} cells × {simulation_data.n_vars} genes")
print(f"Sections: {simulation_data.obs['slice_id'].nunique()}; spatial scaling: {spatial_scaling:.8f}")
display(pd.DataFrame({
    "property": ["cells", "genes", "sections", "HK genes", "coordinate dimensions"],
    "value": [simulation_data.n_obs, simulation_data.n_vars, simulation_data.obs["slice_id"].nunique(), 10, 3],
}))

# %% [markdown]
# ## 2. Construct the mixed-resolution training input from scratch
#
# Every fifth section is observed, giving 20 input sections. With `m=30`, none of
# these 20 observed sections reaches the high-resolution interval, so all 12,495
# input observations are low-resolution bins generated with `bin_factor=8`.
# Aggregation averages all cells within each spatial bin and uses their centroid
# as the bin coordinate. No cells are randomly selected.
#
# The other 80 sections are not removed from `simulation_data`; they remain
# available for full reconstruction and evaluation.

# %%
from analysis.simulation.resolution_utils import create_mixed_resolution_dataset

preprocessing_start = time.time()
train_data = create_mixed_resolution_dataset(
    simulation_data,
    slice_interval_k=SLICE_INTERVAL_K,
    high_res_interval_m=HIGH_RES_INTERVAL_M,
    bin_factor=BIN_FACTOR,
)
train_data.obsm["spatial"] = np.asarray(train_data.obsm["spatial"]) * spatial_scaling
train_data.obs["slice_id"] = train_data.obs["slice_id"].astype(str)
expected_observed_sections = {str(i) for i in range(0, 100, SLICE_INTERVAL_K)}
assert train_data.shape == (12495, 100)
assert set(train_data.obs["slice_id"]) == expected_observed_sections
assert train_data.obs["slice_id"].nunique() == 20
assert np.isfinite(np.asarray(train_data.X)).all()
assert np.isfinite(np.asarray(train_data.obsm["spatial"])).all()

train_path = OUTPUT_DIR / "mixed_resolution_input.h5ad"
train_data.write_h5ad(train_path, compression="gzip")
resolution_counts = train_data.obs.groupby("resolution_type", observed=True).size()
run_config.update({
    "training_observations": int(train_data.n_obs),
    "training_genes": int(train_data.n_vars),
    "observed_sections": int(train_data.obs["slice_id"].nunique()),
    "high_resolution_observations": int(resolution_counts.get("high_res", 0)),
    "low_resolution_bins": int(resolution_counts.get("low_res", 0)),
    "preprocessing_seconds": float(time.time() - preprocessing_start),
})
print(f"New training input: {train_data.shape[0]:,} observations × {train_data.shape[1]} genes")
print(f"Observed sections: {train_data.obs['slice_id'].nunique()}; source sections retained: {simulation_data.obs['slice_id'].nunique()}")
display(resolution_counts.rename("observations").to_frame())

# %% [markdown]
# ## 3. Train NTF from the newly constructed input
#
# This is a real 20,000-iteration-configuration training run using the standard
# `NTF.train` API. Model defaults, including early stopping, are unchanged. The
# continuous progress bar uses dynamic width and a one-second refresh interval.
# The final INR state is saved separately from any early-stopping checkpoint.

# %%
from NTF.config import add_shared_args, process_args
from NTF.train import train

parser = add_shared_args(description="Full mouse-brain mixed-resolution tutorial")
args = parser.parse_args([
    "--input_data", str(INPUT_PATH),
    "--output_dir", str(OUTPUT_DIR),
    "--log_dir", str(OUTPUT_DIR / "runs"),
    "--slice_id", "slice_id",
    "--n_iter", str(N_ITER),
    "--device", DEVICE,
])
args = process_args(args)
training_start = time.time()
model_wrapper = train(train_data, args)
training_seconds = time.time() - training_start
trained_inr_path = OUTPUT_DIR / "trained_inr.pt"
torch.save(model_wrapper.inr.state_dict(), trained_inr_path)
run_config.update({
    "training_seconds": float(training_seconds),
    "trained_inr": trained_inr_path.name,
    "early_stopping_checkpoint_present": (OUTPUT_DIR / "best_model.pth").is_file(),
})
print(f"Training wall time: {training_seconds / 60:.2f} minutes")
print(f"Saved new final INR state: {trained_inr_path}")

# %% [markdown]
# ## 4. Reconstruct every original high-resolution coordinate
#
# `sample_points` queries all 1,975,530 source coordinates. The new reconstruction
# keeps the original 100-gene `X` as ground truth and stores the new NTF output in
# `layers['prediction']`. Only expression, observation metadata and spatial
# coordinates are copied; unrelated large auxiliary embeddings are not duplicated.
# No observation, gene or section is filtered.

# %%
from NTF.sample import sample_points

query_start = time.time()
query_coordinates = torch.from_numpy(raw_coordinates * spatial_scaling).float().to(DEVICE)
prediction_result = sample_points(model_wrapper.inr, query_coordinates, batch_size=QUERY_BATCH_SIZE)
prediction = prediction_result["expression"].cpu().numpy()
query_seconds = time.time() - query_start
assert prediction.shape == (1975530, 100)
assert np.isfinite(prediction).all()
assert not np.array_equal(prediction, np.asarray(simulation_data.X))

reconstruction = ad.AnnData(
    X=np.asarray(simulation_data.X).copy(),
    obs=simulation_data.obs.copy(),
    var=pd.DataFrame(index=simulation_data.var_names.copy()),
)
reconstruction.obsm["spatial"] = raw_coordinates.copy()
reconstruction.layers["prediction"] = prediction
assert reconstruction.obs_names.equals(simulation_data.obs_names)
assert np.array_equal(reconstruction.obsm["spatial"], simulation_data.obsm["spatial"])
assert set(reconstruction.obs["slice_id"]) == {str(i) for i in range(100)}
run_config.update({
    "reconstruction_observations": int(reconstruction.n_obs),
    "reconstruction_genes": int(reconstruction.n_vars),
    "reconstruction_sections": int(reconstruction.obs["slice_id"].nunique()),
    "query_seconds": float(query_seconds),
})
print(f"New prediction: {prediction.shape[0]:,} cells × {prediction.shape[1]} genes")
print(f"Prediction covers {reconstruction.obs['slice_id'].nunique()} sections; query time: {query_seconds / 60:.2f} minutes")
del prediction_result, query_coordinates
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ## 5. Recompute all metrics from the new prediction
#
# Pearson correlation, Spearman correlation, RMSE and 3D SSIM are computed for all
# 100 genes using every cell. The per-gene table retains all genes and labels the
# 90 Diff / 10 HK groups. Only the 90 Diff genes enter the overall mean, matching
# the paper's evaluation of spatially patterned reconstruction. No cell or section
# subsetting is applied.

# %%
def safe_correlation(function, truth, predicted):
    if np.std(truth) == 0 or np.std(predicted) == 0:
        return np.nan
    return float(function(truth, predicted).statistic)


def calculate_3d_ssim(truth, predicted, coordinates):
    grid_dimensions = coordinates.astype(int).max(axis=0) + 1
    minimum_dimension = int(min(grid_dimensions))
    window_size = minimum_dimension if minimum_dimension % 2 else minimum_dimension - 1
    if window_size < 3:
        return np.nan, grid_dimensions, window_size
    truth_volume = np.zeros(grid_dimensions, dtype=np.float64)
    prediction_volume = np.zeros(grid_dimensions, dtype=np.float64)
    integer_coordinates = coordinates.astype(int)
    x = integer_coordinates[:, 0]
    y = integer_coordinates[:, 1]
    z = integer_coordinates[:, 2]
    truth_volume[x, y, z] = truth
    prediction_volume[x, y, z] = predicted
    data_range = max(truth_volume.max(), prediction_volume.max()) - min(truth_volume.min(), prediction_volume.min())
    score = 1.0 if data_range == 0 else structural_similarity(
        truth_volume,
        prediction_volume,
        data_range=data_range,
        win_size=window_size,
    )
    return float(score), grid_dimensions, window_size


metrics_start = time.time()
truth_matrix = np.asarray(reconstruction.X)
prediction_matrix = np.asarray(reconstruction.layers["prediction"])
pearson_values = []
spearman_values = []
ssim_values = []
grid_dimensions = None
ssim_window_size = None
for gene_index in range(reconstruction.n_vars):
    truth = truth_matrix[:, gene_index]
    predicted = prediction_matrix[:, gene_index]
    pearson_values.append(safe_correlation(pearsonr, truth, predicted))
    spearman_values.append(safe_correlation(spearmanr, truth, predicted))
    ssim_score, grid_dimensions, ssim_window_size = calculate_3d_ssim(truth, predicted, raw_coordinates)
    ssim_values.append(ssim_score)
    if (gene_index + 1) % 10 == 0 or gene_index + 1 == reconstruction.n_vars:
        print(f"Metrics calculated for {gene_index + 1}/{reconstruction.n_vars} genes")

reconstruction.var["pearson_corr"] = pearson_values
reconstruction.var["spearman_corr"] = spearman_values
reconstruction.var["rmse"] = np.sqrt(np.mean((truth_matrix - prediction_matrix) ** 2, axis=0))
reconstruction.var["ssim_3d"] = ssim_values
reconstruction.var["gene_group"] = ["Diff"] * 90 + ["HK"] * 10
reconstruction.var["included_in_overall"] = reconstruction.var["gene_group"] == "Diff"
gene_metrics = reconstruction.var[["gene_group", "included_in_overall", "pearson_corr", "spearman_corr", "rmse", "ssim_3d"]].copy()
gene_metrics.index.name = "gene"
gene_metrics = gene_metrics.reset_index()
overall_gene_metrics = gene_metrics.loc[gene_metrics["included_in_overall"]]
summary = overall_gene_metrics[["pearson_corr", "spearman_corr", "rmse", "ssim_3d"]].mean().to_frame().T
summary.insert(0, "evaluation", "new NTF run; overall mean over 90 Diff genes only (10 HK excluded); all 1,975,530 cells; all 100 sections")
metrics_seconds = time.time() - metrics_start

gene_metrics.to_csv(OUTPUT_DIR / "gene_metrics.csv", index=False)
summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
reconstruction_path = OUTPUT_DIR / "reconstruction.h5ad"
reconstruction.write_h5ad(reconstruction_path, compression="gzip")
run_config.update({
    "metrics_seconds": float(metrics_seconds),
    "ssim_grid_dimensions": np.asarray(grid_dimensions).tolist(),
    "ssim_window_size": int(ssim_window_size),
    "reconstruction_file": reconstruction_path.name,
    "overall_evaluation_genes": 90,
    "excluded_HK_genes": 10,
})
print(f"SSIM grid: {grid_dimensions}; window: {ssim_window_size}; metric time: {metrics_seconds:.2f} seconds")
print(f"Saved new reconstruction: {reconstruction_path}")
display(gene_metrics.round(4))
display(summary.round(4))

# %% [markdown]
# ## 6. Inspect the ten HK noise controls
#
# HK genes contain non-zero random expression without a spatial pattern. Thus,
# “no false positives” does **not** mean zero expression or a high correlation
# with unpredictable noise. It means NTF should not invent coherent hotspots or
# other localized spatial patterns. A smooth non-zero background is acceptable.
#
# The table uses every cell and compares truth/prediction means, variability and
# the range of all 100 section means. The raw-expression maps then inspect every
# HK gene on section 45. HK genes remain excluded from the overall metric mean
# because their random cell-level noise is not a reconstructable spatial signal.

# %%
hk_gene_names = reconstruction.var_names[reconstruction.var["gene_group"] == "HK"].tolist()
section_numbers = reconstruction.obs["slice_id"].astype(int).to_numpy()
section_counts = np.bincount(section_numbers, minlength=100)
hk_rows = []
for gene_name in hk_gene_names:
    gene_index = reconstruction.var_names.get_loc(gene_name)
    truth = truth_matrix[:, gene_index]
    predicted = prediction_matrix[:, gene_index]
    truth_section_means = np.bincount(section_numbers, weights=truth, minlength=100) / section_counts
    prediction_section_means = np.bincount(section_numbers, weights=predicted, minlength=100) / section_counts
    hk_rows.append({
        "gene": gene_name,
        "truth_mean": float(np.mean(truth)),
        "prediction_mean": float(np.mean(predicted)),
        "truth_std": float(np.std(truth)),
        "prediction_std": float(np.std(predicted)),
        "truth_section_mean_min": float(np.min(truth_section_means)),
        "truth_section_mean_max": float(np.max(truth_section_means)),
        "prediction_section_mean_min": float(np.min(prediction_section_means)),
        "prediction_section_mean_max": float(np.max(prediction_section_means)),
        "prediction_section_mean_range": float(np.ptp(prediction_section_means)),
    })
hk_diagnostics = pd.DataFrame(hk_rows)
hk_diagnostics.to_csv(OUTPUT_DIR / "hk_diagnostics.csv", index=False)
display(hk_diagnostics.round(4))

hk_section = reconstruction[reconstruction.obs["slice_id"] == "45", hk_gene_names].copy()
hk_truth_values = np.asarray(hk_section.X).ravel()
hk_prediction_values = np.asarray(hk_section.layers["prediction"]).ravel()
hk_expression_min = min(hk_truth_values.min(), hk_prediction_values.min())
hk_expression_max = max(hk_truth_values.max(), hk_prediction_values.max())
fig, axes = plt.subplots(4, 5, figsize=(22, 14))
for column in range(5):
    for gene_offset in range(2):
        gene_name = hk_gene_names[gene_offset * 5 + column]
        for prediction_row, (layer, title) in enumerate([(None, "Ground truth"), ("prediction", "NTF reconstruction")]):
            row = gene_offset * 2 + prediction_row
            sc.pl.embedding(
                hk_section,
                basis="spatial",
                color=gene_name,
                layer=layer,
                title=f"{gene_name}\n{title}",
                ax=axes[row, column],
                show=False,
                cmap="Blues",
                vmin=hk_expression_min,
                vmax=hk_expression_max,
                colorbar_loc=None,
            )
fig.suptitle("HK random-noise controls on section 45: raw expression, no normalization", fontsize=15)
fig.subplots_adjust(left=0.03, right=0.91, bottom=0.05, top=0.88, wspace=0.16, hspace=0.26)
hk_mappable = plt.cm.ScalarMappable(cmap="Blues")
hk_mappable.set_clim(hk_expression_min, hk_expression_max)
hk_colorbar = fig.colorbar(hk_mappable, cax=fig.add_axes([0.94, 0.20, 0.012, 0.60]), label="Expression")
hk_colorbar.set_ticks([hk_expression_min, hk_expression_max], labels=["Low", "High"])
display(fig)
plt.close(fig)

# %% [markdown]
# ## 7. Compare complete 2D sections from the new run
#
# Sections 35, 45 and 55 show every high-resolution reference cell, every actual
# input observation and every newly reconstructed cell for `Gene_12_Diff`.
# `scanpy.pl.embedding` determines 2D point sizes automatically. Expression values
# are plotted directly without min–max scaling, z-scoring or any other
# normalization. A shared raw-value range, `Blues`, and a qualitative `Low`–`High`
# colorbar keep the panels comparable without changing the data.

# %%
reconstruction_gene = reconstruction[:, [PLOT_GENE]].copy()
train_values = np.asarray(train_data[:, PLOT_GENE].X).ravel()
truth_values = np.asarray(reconstruction_gene.X).ravel()
prediction_values = np.asarray(reconstruction_gene.layers["prediction"]).ravel()
expression_min = min(train_values.min(), truth_values.min(), prediction_values.min())
expression_max = max(train_values.max(), truth_values.max(), prediction_values.max())


def add_low_high_colorbar(fig, rect):
    colorbar_ax = fig.add_axes(rect)
    mappable = plt.cm.ScalarMappable(cmap="Blues")
    mappable.set_clim(expression_min, expression_max)
    colorbar = fig.colorbar(mappable, cax=colorbar_ax, label="Expression")
    colorbar.set_ticks([expression_min, expression_max], labels=["Low", "High"])


def plot_sub_slice(adata, slice_id, gene_name, layer=None, show=False, ax=None, title=None):
    subset = adata[adata.obs["slice_id"] == slice_id].copy()
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
for row, section_id in enumerate(PLOT_SLICE_IDS):
    columns = [
        (reconstruction_gene, None, "High-resolution truth"),
        (train_data, None, "Low-resolution NTF input"),
        (reconstruction_gene, "prediction", "NTF reconstruction"),
    ]
    for column, (data, layer, title) in enumerate(columns):
        plot_sub_slice(
            data,
            section_id,
            PLOT_GENE,
            layer=layer,
            ax=axes[row, column],
            title=f"{PLOT_GENE} - {section_id}\n{title}",
        )
fig.subplots_adjust(left=0.03, right=0.90, bottom=0.03, top=0.94, wspace=0.12, hspace=0.22)
add_low_high_colorbar(fig, [0.93, 0.20, 0.015, 0.60])
display(fig)
plt.close(fig)

# %% [markdown]
# ## 8. Compare the complete new volumes from three views
#
# All observations are drawn without stride, random sampling or thinning, using
# the original expression values without normalization. Semi-transparent points
# (`alpha=0.35`) reveal structures that would otherwise be occluded. The sparse-
# input row uses larger points, and extra vertical space separates the three rows.

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
    for column, (elevation, azimuth, view_title) in enumerate(views):
        ax = axes[row, column]
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
            alpha=0.35,
            colorbar_loc=None,
        )
        if enlarge_points:
            ax.collections[0].set_sizes([72.0])
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_box_aspect(np.ptp(data.obsm["spatial"], axis=0))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
fig.suptitle(f"{PLOT_GENE}: new full run from three views", fontsize=15)
fig.subplots_adjust(left=0.02, right=0.90, bottom=0.04, top=0.94, wspace=0.08, hspace=0.38)
add_low_high_colorbar(fig, [0.93, 0.20, 0.015, 0.60])
display(fig)
plt.close(fig)

# %% [markdown]
# ## 9. Export the final run summary
#
# The workbook contains the new run configuration, all 100 per-gene metric rows,
# overall means over the 90 Diff genes, and the ten HK noise-control diagnostics.
# All values come from this notebook's newly trained model.

# %%
run_config["total_seconds_before_notebook_export"] = float(time.time() - run_start)
run_config["mean_pearson_corr"] = float(summary["pearson_corr"].iloc[0])
run_config["mean_spearman_corr"] = float(summary["spearman_corr"].iloc[0])
run_config["mean_rmse"] = float(summary["rmse"].iloc[0])
run_config["mean_ssim_3d"] = float(summary["ssim_3d"].iloc[0])
(OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
run_table = pd.DataFrame({"property": list(run_config), "value": [str(value) for value in run_config.values()]})
workbook_path = OUTPUT_DIR / "simulation_mousebrain_full_rerun_summary.xlsx"
with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
    run_table.to_excel(writer, sheet_name="Run", index=False)
    summary.to_excel(writer, sheet_name="Metrics_summary", index=False)
    gene_metrics.to_excel(writer, sheet_name="Gene_metrics", index=False)
    hk_diagnostics.to_excel(writer, sheet_name="HK_diagnostics", index=False)

print(f"Saved Excel summary: {workbook_path}")
print(f"Full rerun finished successfully in {(time.time() - run_start) / 60:.2f} minutes before notebook export.")
