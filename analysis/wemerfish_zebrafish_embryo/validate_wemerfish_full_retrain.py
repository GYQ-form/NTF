#!/usr/bin/env python
from pathlib import Path
import re

import anndata as ad
import nbformat
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score


def find_root():
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    return next(path for path in candidates if (path / "NTF" / "train.py").is_file())


ROOT = find_root()
EXPERIMENT_DIR = ROOT / "mytrain/revision/experiment/code_tutorial_notebooks"
RUN_DIR = EXPERIMENT_DIR / "outputs/02_wemerfish_notebook_full_retrain"
MODEL_DIR = RUN_DIR / "reproduction/models"
LOG_DIR = RUN_DIR / "reproduction/logs"
ANALYSIS_DIR = RUN_DIR
INPUT_PATH = ROOT / "mytrain/data/wemerfish_zebrafish_embryo/processed/wemerfish_C6s_E2.h5ad"
SCENARIOS = {
    "senario1": {"split": "group", "sections": [2, 4, 6, 8, 10, 12, 14]},
    "senario2": {"split": "group2", "sections": [8, 9, 10]},
    "senario3": {"split": "group4", "sections": [1, 2, 15, 16]},
    "head": {"split": "head", "sections": [1, 2]},
    "tail": {"split": "tail", "sections": [15, 16]},
}

validation_rows = []
manifest_rows = []


def record(check, passed, detail=""):
    validation_rows.append({"check": check, "status": "PASS" if passed else "FAIL", "detail": str(detail)})


def dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


input_data = ad.read_h5ad(INPUT_PATH)
results = {}
for scenario, info in SCENARIOS.items():
    scenario_dir = MODEL_DIR / scenario
    for filename in ["best_model.pth", "label_annotator.pkl", "test_results.h5ad", "gene_metrics.csv"]:
        path = scenario_dir / filename
        record(f"{scenario}: new {filename}", path.is_file() and path.stat().st_size > 0, path.relative_to(ROOT))

    log_path = LOG_DIR / f"{scenario}.log"
    log_text = log_path.read_text()
    record(f"{scenario}: training completed", "Training finished. Time elapsed:" in log_text)
    record(f"{scenario}: no cached checkpoint evaluation", "eval_only: loaded checkpoint" not in log_text)
    record(f"{scenario}: label annotator trained", "KNN label annotator built" in log_text)
    record(f"{scenario}: test inference completed", "Inference finished. Time elapsed:" in log_text)
    record(f"{scenario}: test results saved", "Test results saved" in log_text)

    result = ad.read_h5ad(scenario_dir / "test_results.h5ad")
    results[scenario] = result
    expected = input_data[input_data.obs[info["split"]].astype(str) == "test"]
    record(f"{scenario}: cell order", result.obs_names.equals(expected.obs_names))
    record(f"{scenario}: coordinates", np.allclose(result.obsm["spatial"], expected.obsm["spatial"]))
    for key in ["type", "clusters", "tissue"]:
        record(f"{scenario}: {key} labels", result.obs[key].astype(str).equals(expected.obs[key].astype(str)))
    prediction = dense(result.layers["prediction"])
    record(f"{scenario}: prediction shape", prediction.shape == (result.n_obs, result.n_vars), prediction.shape)
    record(f"{scenario}: finite prediction", np.isfinite(prediction).all())
    for key in ["pred_clusters", "pred_type"]:
        record(f"{scenario}: {key}", key in result.obs and result.obs[key].notna().all())
    record(f"{scenario}: 3D SSIM", "ssim_3d" in result.var and np.isfinite(result.var["ssim_3d"]).all())
    record(f"{scenario}: complete test sections", sorted(result.obs["slice_id"].unique().astype(int)) == info["sections"])

    timestamps = re.findall(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}", log_text, flags=re.MULTILINE)
    iterations = re.findall(r"Iter: (\d+),", log_text)
    training_time = re.search(r"Training finished\. Time elapsed: ([\d.]+) seconds", log_text)
    inference_time = re.search(r"Inference finished\. Time elapsed: ([\d.]+) seconds", log_text)
    split = re.search(r"Data split into training set \((\d+) cells\) and test set \((\d+) cells\)", log_text)
    manifest_rows.append({
        "scenario": scenario,
        "split_column": info["split"],
        "requested_iterations": 20000,
        "final_logged_iteration": int(iterations[-1]) if iterations else np.nan,
        "batch_size": 8192,
        "dropout_enabled": True,
        "cat_keys": "clusters type",
        "physical_gpu": 7,
        "n_train_cells": int(split.group(1)),
        "n_test_cells": int(split.group(2)),
        "training_seconds": float(training_time.group(1)),
        "inference_seconds": float(inference_time.group(1)),
        "start_time": timestamps[0],
        "end_time": timestamps[-1],
        "results_dir": str(scenario_dir.relative_to(ROOT)),
    })

notebook = nbformat.read(ROOT / "tutorial/02_wemerfish_scenarios.ipynb", as_version=4)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
record("Notebook: sequential execution", [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1)))
errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type == "error"]
record("Notebook: no execution errors", not errors)
training_cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "subprocess.Popen" in cell.source)
training_output = "".join(output.get("text", "") for output in training_cell.outputs if output.output_type == "stream")
record("Notebook: in-notebook subprocess training", "train_and_eval.py" in training_cell.source)
record("Notebook: five completed training runs", training_output.count("outputs passed freshness checks") == 5)
training_runs = pd.read_csv(ANALYSIS_DIR / "training_runs.csv")
record("Notebook: training run table", len(training_runs) == 5 and training_runs["scenario"].tolist() == list(SCENARIOS))
spatial_cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "celltype_per_section = pd.DataFrame" in cell.source)
embedded_maps = sum("image/png" in output.get("data", {}) for output in spatial_cell.outputs)
record("Notebook: five embedded complete spatial maps", embedded_maps == 5, embedded_maps)

gene_export = pd.read_csv(ANALYSIS_DIR / "reconstruction_gene_metrics.csv")
label_export = pd.read_csv(ANALYSIS_DIR / "label_metrics.csv")
section_export = pd.read_csv(ANALYSIS_DIR / "celltype_per_section_metrics.csv")
record("Per-section ARI: 18 rows", len(section_export) == 18, len(section_export))

for scenario, result in results.items():
    truth = dense(result.X).astype(np.float64)
    prediction = dense(result.layers["prediction"]).astype(np.float64)
    pearson, spearman = [], []
    for j in range(result.n_vars):
        observed = truth[:, j]
        predicted = prediction[:, j]
        nonzero = observed > 0
        if nonzero.sum() < 2 or np.var(observed[nonzero]) < 1e-10 or np.var(predicted[nonzero]) < 1e-10:
            pearson.append(np.nan)
            spearman.append(np.nan)
        else:
            pearson.append(pearsonr(observed, predicted).statistic)
            spearman.append(spearmanr(observed, predicted).statistic)
    expected_gene = pd.DataFrame({
        "gene": result.var_names,
        "pearson": pearson,
        "spearman": spearman,
        "rmse": np.sqrt(((truth - prediction) ** 2).mean(axis=0)),
        "ssim_3d": result.var["ssim_3d"].to_numpy(),
    }).set_index("gene")
    actual_gene = gene_export[gene_export["scenario"] == scenario].set_index("gene").loc[result.var_names]
    for metric in ["pearson", "spearman", "rmse", "ssim_3d"]:
        record(
            f"{scenario}: independently recomputed {metric}",
            np.allclose(actual_gene[metric], expected_gene[metric], rtol=0, atol=1e-10, equal_nan=True),
        )

    for level, truth_key, prediction_key in [("cell type (clusters)", "clusters", "pred_clusters"), ("niche (type)", "type", "pred_type")]:
        truth_labels = result.obs[truth_key].astype(str)
        predicted_labels = result.obs[prediction_key].astype(str)
        expected_metrics = {
            "ARI": adjusted_rand_score(truth_labels, predicted_labels),
            "NMI": normalized_mutual_info_score(truth_labels, predicted_labels),
            "FMI": fowlkes_mallows_score(truth_labels, predicted_labels),
        }
        actual = label_export[(label_export["scenario"] == scenario) & (label_export["level"] == level)].iloc[0]
        for metric, value in expected_metrics.items():
            record(f"{scenario}: {level} {metric}", np.isclose(actual[metric], value, rtol=0, atol=1e-12))

    sections = result.obs["slice_id"].astype(int).to_numpy()
    truth = result.obs["clusters"].astype(str).to_numpy()
    prediction = result.obs["pred_clusters"].astype(str).to_numpy()
    for section in SCENARIOS[scenario]["sections"]:
        selected = sections == section
        expected_ari = adjusted_rand_score(truth[selected], prediction[selected])
        actual = section_export[(section_export["scenario"] == scenario) & (section_export["slice_id"] == section)]
        record(
            f"{scenario}: section {section} ARI",
            len(actual) == 1 and np.isclose(actual.iloc[0]["ARI"], expected_ari, rtol=0, atol=1e-12),
            expected_ari,
        )
    for extension in ["png", "svg"]:
        path = ANALYSIS_DIR / "celltype_spatial" / f"celltype_spatial_{scenario}.{extension}"
        record(f"{scenario}: complete spatial {extension}", path.is_file() and path.stat().st_size > 10_000, path.relative_to(ROOT))

validation = pd.DataFrame(validation_rows)
manifest = pd.DataFrame(manifest_rows)
validation.to_csv(RUN_DIR / "full_retrain_validation.csv", index=False)
manifest.to_csv(RUN_DIR / "training_manifest.csv", index=False)
with pd.ExcelWriter(RUN_DIR / "full_retrain_summary.xlsx", engine="openpyxl") as writer:
    manifest.to_excel(writer, sheet_name="Training_runs", index=False)
    pd.read_csv(ANALYSIS_DIR / "reconstruction_summary.csv").to_excel(writer, sheet_name="Reconstruction", index=False)
    label_export.to_excel(writer, sheet_name="Label_metrics", index=False)
    section_export.to_excel(writer, sheet_name="Celltype_section_ARI", index=False)
    pd.read_csv(ANALYSIS_DIR / "niche_heterogeneity_summary.csv").to_excel(writer, sheet_name="Niche_heterogeneity", index=False)
    validation.to_excel(writer, sheet_name="Validation", index=False)

failed = validation[validation["status"] == "FAIL"]
print(f"Validation: {len(validation) - len(failed)}/{len(validation)} PASS")
print(manifest[["scenario", "final_logged_iteration", "training_seconds", "inference_seconds"]].to_string(index=False))
print(RUN_DIR / "full_retrain_summary.xlsx")
if not failed.empty:
    print(failed.to_string(index=False))
    raise RuntimeError(f"{len(failed)} full-retrain validation checks failed")
