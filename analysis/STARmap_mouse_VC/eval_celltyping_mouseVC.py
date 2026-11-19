#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cell typing benchmark for NTF, FEAST and SpatialZ on STARmap mouse VC 5-gene data.

Each method contributes one label set per scenario, compared with the
ground-truth ``leiden`` labels via ARI / NMI / FMI:

- NTF: direct model-side labels ``obs['pred_leiden']`` (built-in kNN annotator);
- SpatialZ: direct labels ``obs['pred_label']`` (native label transfer);
- UniST: direct labels ``obs['pred_label']`` (official API label output via
  ``bm_unist.py --label_key leiden``; only senario1/senario2 — UniST
  interpolates between flanking slices and does not extrapolate);
- FEAST: leiden clustering on its reconstructed expression
  (``layers['prediction']``), since FEAST has no direct-label capability.

senario1/senario2 are evaluated for all three methods after an alignment check
(cell count, coordinates, GT labels); senario3/head/tail are NTF-only
extrapolation scenarios.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.metrics import (
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

SCENARIO_SPLIT = {
    "senario1": "group",
    "senario2": "group2",
    "senario3": "group4",
    "head": "head",
    "tail": "tail",
}
SHARED_SCENARIOS = ["senario1", "senario2"]
NTF_ONLY_SCENARIOS = ["senario3", "head", "tail"]
METHODS = ["NTF", "FEAST", "SpatialZ", "UniST"]
DIRECT_LABEL_KEY = {"NTF": "pred_leiden", "SpatialZ": "pred_label", "UniST": "pred_label"}
LABEL_SOURCE = {"NTF": "direct", "SpatialZ": "direct", "FEAST": "clustering", "UniST": "direct"}
METRICS = ("ari", "nmi", "fmi")


def load_result(path):
    """Load a test_results.h5ad; return None with a warning if missing."""
    path = Path(path)
    if not path.exists():
        warnings.warn(f"Result file missing, skipped: {path}")
        return None
    return sc.read_h5ad(path)


def obs_labels(adata, key):
    labels = adata.obs[key]
    if hasattr(labels, "cat"):
        return labels.astype(str).to_numpy()
    return labels.to_numpy().astype(str)


def check_alignment(adatas, scenario):
    """Verify the three methods share the same test cells for a scenario."""
    names = [m for m in METHODS if adatas.get(m) is not None]
    messages = []
    ok = True
    ref = adatas[names[0]]
    for name in names[1:]:
        other = adatas[name]
        if other.n_obs != ref.n_obs:
            ok = False
            messages.append(f"n_obs mismatch: {names[0]}={ref.n_obs} vs {name}={other.n_obs}")
        elif not np.allclose(np.asarray(ref.obsm["spatial"]), np.asarray(other.obsm["spatial"])):
            ok = False
            messages.append(f"spatial coordinates differ between {names[0]} and {name}")
        gt_ref = obs_labels(ref, "leiden")
        gt_other = obs_labels(other, "leiden")
        if ref.obs["leiden"].isna().any() or other.obs["leiden"].isna().any():
            ok = False
            messages.append("missing leiden labels detected")
        elif not np.array_equal(gt_ref, gt_other):
            ok = False
            messages.append(f"leiden labels differ between {names[0]} and {name}")
    status = "pass" if ok else "FAIL: " + "; ".join(messages)
    print(f"[alignment] {scenario}: {status} ({', '.join(names)}, n_obs={ref.n_obs})")
    return ok, status


def cluster_labels(prediction, resolution, seed):
    """Leiden clustering on the reconstructed expression (no PCA)."""
    prediction = np.asarray(prediction, dtype=np.float32)
    finite = np.isfinite(prediction).all(axis=1)
    n_dropped = int((~finite).sum())
    if n_dropped:
        warnings.warn(f"{n_dropped} cells with NaN/Inf prediction dropped from clustering")
    sub = ad.AnnData(X=prediction[finite])
    sc.pp.neighbors(sub, n_neighbors=15)
    sc.tl.leiden(sub, resolution=resolution, random_state=seed)
    return sub.obs["leiden"].astype(str).to_numpy(), finite


def label_metrics(gt, pred):
    return {
        "ari": float(adjusted_rand_score(gt, pred)),
        "nmi": float(normalized_mutual_info_score(gt, pred)),
        "fmi": float(fowlkes_mallows_score(gt, pred)),
    }


def get_method_labels(adata, method, scenario, args):
    """Return (labels, valid_mask) for one method; (None, None) if unavailable."""
    if LABEL_SOURCE[method] == "direct":
        key = DIRECT_LABEL_KEY[method]
        if key not in adata.obs.columns:
            warnings.warn(f"{scenario}/{method}: obs['{key}'] missing, skipped")
            return None, None
        return obs_labels(adata, key), np.ones(adata.n_obs, dtype=bool)
    if "prediction" not in adata.layers:
        warnings.warn(f"{scenario}/{method}: layers['prediction'] missing, clustering skipped")
        return None, None
    return cluster_labels(adata.layers["prediction"], args.leiden_resolution, args.seed)


def evaluate_scenario(scenario, adatas, args, rows, label_cache):
    ref = next(a for a in adatas.values() if a is not None)
    gt = obs_labels(ref, "leiden")
    for method in METHODS:
        adata = adatas.get(method)
        if adata is None:
            continue
        pred, valid = get_method_labels(adata, method, scenario, args)
        if pred is None:
            for metric in METRICS:
                rows.append({
                    "method": method, "scenario": scenario,
                    "label_source": LABEL_SOURCE[method], "metric": metric,
                    "value": np.nan, "status": "not_applicable",
                })
            continue
        if len(pred) != int(valid.sum()):
            raise ValueError(f"{scenario}/{method}: label length mismatch")
        metrics = label_metrics(gt[valid], pred)
        for metric, value in metrics.items():
            rows.append({
                "method": method, "scenario": scenario,
                "label_source": LABEL_SOURCE[method], "metric": metric,
                "value": value, "status": "ok",
            })
        full = np.full(len(gt), "NA", dtype=object)
        full[valid] = pred
        label_cache[(scenario, method)] = full.astype(str)
        print(f"[{LABEL_SOURCE[method]}] {scenario}/{method}: "
              + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


def scenario_palette(labels):
    cats = sorted(pd.unique(labels))
    cmap = plt.get_cmap("tab20")
    return cats, {c: cmap(i % 20) for i, c in enumerate(cats)}


def plot_spatial_panel(ax, coords, labels, title, palette, point_size=1.0):
    for cat in palette:
        mask = labels == cat
        ax.scatter(coords[mask, 0], coords[mask, 1], s=point_size, c=[palette[cat]], linewidths=0, rasterized=True)
    if title:
        ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_spatial_maps_scenario(scenario, adatas, label_cache, outpath):
    """GT + per-method labels, faceted by the scenario's test slices."""
    ref = next(a for a in adatas.values() if a is not None)
    coords = np.asarray(ref.obsm["spatial"])
    slices = obs_labels(ref, "slice_id")
    slice_ids = sorted(pd.unique(slices))
    gt = obs_labels(ref, "leiden")
    panels = [("GT (leiden)", gt)]
    for method in METHODS:
        key = (scenario, method)
        if key in label_cache and adatas.get(method) is not None:
            source = LABEL_SOURCE[method]
            tag = {"direct": DIRECT_LABEL_KEY.get(method, "direct"), "clustering": "clustering"}[source]
            panels.append((f"{method} {tag}", label_cache[key]))
    all_labels = np.concatenate([p[1] for p in panels])
    _, palette = scenario_palette(all_labels)
    gt_cats = sorted(pd.unique(gt))
    n_rows, n_cols = len(panels), len(slice_ids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows), squeeze=False)
    for i, (name, labels) in enumerate(panels):
        for j, sid in enumerate(slice_ids):
            mask = slices == sid
            if i == 0:
                title = f"{name}\n{sid}" if j == 0 else sid
            else:
                ari = adjusted_rand_score(gt[mask], labels[mask])
                title = (f"{name}\n{sid}, ARI={ari:.3f}" if j == 0
                         else f"{sid}, ARI={ari:.3f}")
            plot_spatial_panel(axes[i][j], coords[mask], labels[mask], title, palette)
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=3, color=palette[c]) for c in gt_cats]
    fig.legend(handles, gt_cats, loc="center right", fontsize=5, frameon=False, ncol=1, title="GT label")
    fig.suptitle(f"{scenario}: GT vs method labels (test slices)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.88, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[figure] {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_data", default="mytrain/data/mouse_VC/processed/mouseVC_5gene.h5ad")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ntf_dir", default="mytrain/res/mouseVC/mouseVC_5gene")
    ap.add_argument("--feast_dir", default="mytrain/res/FEAST/mouseVC")
    ap.add_argument("--spatialz_dir", default="/home/gongyuqiao/ur_annotation/SpatialZ/res/mouseVC/5gene")
    ap.add_argument("--unist_dir", default="mytrain/res/UniST/mouseVC_5gene")
    ap.add_argument("--leiden_resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    sc.settings.verbosity = 1
    outdir = Path(args.output_dir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_data)
    input_obs = None
    if input_path.exists():
        input_adata = sc.read_h5ad(input_path)
        input_obs = input_adata.obs
        print(f"[input] {input_path}: {input_adata.n_obs} cells x {input_adata.n_vars} genes")
    else:
        warnings.warn(f"Input data missing, split-column cross-check skipped: {input_path}")

    method_dirs = {"NTF": args.ntf_dir, "FEAST": args.feast_dir,
                   "SpatialZ": args.spatialz_dir, "UniST": args.unist_dir}
    rows, manifest = [], []
    label_cache = {}
    loaded = {}

    for scenario, split_col in SCENARIO_SPLIT.items():
        adatas = {}
        for method in METHODS:
            path = Path(method_dirs[method]) / scenario / "test_results.h5ad"
            adata = load_result(path)
            adatas[method] = adata
            manifest.append({
                "scenario": scenario, "method": method, "result_path": str(path),
                "n_obs": adata.n_obs if adata is not None else "",
                "split_column": split_col,
                "label_source": LABEL_SOURCE[method],
                "alignment_check": "",
            })
        loaded[scenario] = adatas
        available = {m: a for m, a in adatas.items() if a is not None}
        if not available:
            warnings.warn(f"No result available for {scenario}, skipped")
            continue
        if input_obs is not None and split_col in input_obs.columns:
            test_slices = sorted(input_obs.loc[input_obs[split_col].astype(str) == "test", "slice_id"].astype(str).unique())
            n_expected = int((input_obs[split_col].astype(str) == "test").sum())
            ref = next(iter(available.values()))
            match = "pass" if ref.n_obs == n_expected else f"n_obs {ref.n_obs} != input test {n_expected}"
            print(f"[split] {scenario} ({split_col}): test slices {test_slices}, expected {n_expected} cells -> {match}")
            for entry in manifest:
                if entry["scenario"] == scenario:
                    entry["alignment_check"] = f"input split check: {match}"
        if scenario in SHARED_SCENARIOS and len(available) >= 2:
            ok, status = check_alignment(available, scenario)
            for entry in manifest:
                if entry["scenario"] == scenario and entry["method"] in available:
                    entry["alignment_check"] = status
            if not ok:
                raise SystemExit(f"Alignment check failed for {scenario}: {status}")
        evaluate_scenario(scenario, available, args, rows, label_cache)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "celltyping_metrics.csv", index=False)
    pd.DataFrame(manifest).to_csv(outdir / "manifest.csv", index=False)
    print(f"[table] {outdir / 'celltyping_metrics.csv'} ({len(df)} rows)")
    print(f"[table] {outdir / 'manifest.csv'}")

    for scenario, adatas in loaded.items():
        if any(a is not None for a in adatas.values()):
            plot_spatial_maps_scenario(scenario, adatas, label_cache, figdir / f"spatial_maps_{scenario}.png")

    pivot = df.pivot_table(index=["method", "scenario", "label_source"], columns="metric", values="value", aggfunc="first")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
