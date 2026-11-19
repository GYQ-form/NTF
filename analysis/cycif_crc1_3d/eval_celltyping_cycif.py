#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cell typing evaluation for NTF on the t-CyCIF CRC1 3D atlas.

Adapted from analysis/deep_starmap_brain/eval_celltyping_deepstarmap.py.
Differences:

- NTF only (SpatialZ/FEAST infeasible at 12.8M cells — see the experiment's
  baseline_feasibility section): no leiden fallback, no alignment check.
- Ground truth: ``obs['cell_type']`` (21 types; GMM gating + hierarchical
  boolean rules per Lin et al. 2023 Table S4); NTF labels:
  ``obs['pred_cell_type']`` (built-in kNN annotator, k=5 majority vote).
- Spatial-map panels are subsampled to <=100k points per slice (CRC1 slices
  hold ~0.5M cells each).

Metrics (test cells only): ARI / NMI / FMI per scenario. Figures: GT vs NTF
label spatial maps per scenario + GT maps for all slices.

Usage:
    python eval_celltyping_cycif.py --output_dir <out>
"""
from __future__ import annotations

import argparse
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
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
GT_KEY = "cell_type"
PRED_KEY = "pred_cell_type"
METRICS = ("ari", "nmi", "fmi")
MAX_POINTS_PER_PANEL = 100_000


def load_result(path):
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


def label_metrics(gt, pred):
    return {
        "ari": float(adjusted_rand_score(gt, pred)),
        "nmi": float(normalized_mutual_info_score(gt, pred)),
        "fmi": float(fowlkes_mallows_score(gt, pred)),
    }


def build_global_palette(categories):
    colors = (list(plt.get_cmap("tab20").colors)
              + list(plt.get_cmap("tab20b").colors)
              + list(plt.get_cmap("tab20c").colors))
    cats = sorted(pd.unique(np.asarray(categories, dtype=str)))
    return cats, {c: colors[i % len(colors)] for i, c in enumerate(cats)}


def plot_spatial_panel(ax, coords, labels, title, palette, seed=42):
    if len(labels) > MAX_POINTS_PER_PANEL:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(labels), MAX_POINTS_PER_PANEL, replace=False)
        coords, labels = coords[idx], labels[idx]
    for cat in palette:
        mask = labels == cat
        ax.scatter(coords[mask, 0], coords[mask, 1], s=1.0, c=[palette[cat]], linewidths=0, rasterized=True)
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def add_bottom_legend(fig, palette, gt_cats, fig_h_in):
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=3, color=palette[c]) for c in gt_cats]
    n_leg_col = 4
    wrapped_cats = [textwrap.fill(str(c), 30) for c in gt_cats]
    n_leg_rows = int(np.ceil(len(gt_cats) / n_leg_col))
    leg_h_in = 0.25 + n_leg_rows * 0.22 + 0.15
    bottom = min((leg_h_in + 0.12) / fig_h_in, 0.45)
    fig.legend(handles, wrapped_cats, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               fontsize=7, frameon=False, ncol=n_leg_col, title="GT label",
               title_fontsize=8, columnspacing=0.8, handletextpad=0.3)
    return bottom


def plot_spatial_maps_scenario(scenario, adata, pred_labels, slice_col, palette, gt_cats, outpath):
    coords = np.asarray(adata.obsm["spatial"])
    slices = obs_labels(adata, slice_col)
    slice_ids = sorted(pd.unique(slices), key=lambda s: int(s))
    if len(slice_ids) > 6:
        idx = np.linspace(0, len(slice_ids) - 1, 6).round().astype(int)
        slice_ids = [slice_ids[i] for i in idx]
    gt = obs_labels(adata, GT_KEY)
    panels = [(f"GT ({GT_KEY})", gt), (f"NTF {PRED_KEY}", pred_labels)]
    n_rows, n_cols = len(panels), len(slice_ids)
    panel_w, panel_h = 4.4, 4.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows), squeeze=False)
    for i, (name, labels) in enumerate(panels):
        for j, sid in enumerate(slice_ids):
            mask = slices == sid
            if i == 0:
                title = f"{name}\nslice {sid}" if j == 0 else f"slice {sid}"
            else:
                ari = adjusted_rand_score(gt[mask], labels[mask])
                title = (f"{name}\nslice {sid}, ARI={ari:.3f}" if j == 0
                         else f"slice {sid}, ARI={ari:.3f}")
            plot_spatial_panel(axes[i][j], coords[mask], labels[mask], title, palette)
    bottom = add_bottom_legend(fig, palette, gt_cats, panel_h * n_rows)
    fig.suptitle(f"{scenario}: GT vs NTF cell-type labels (test slices)", fontsize=13)
    fig.tight_layout(rect=[0, bottom, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[figure] {outpath}")


def plot_gt_all_slices(input_adata, palette, gt_cats, slice_col, outpath):
    coords = np.asarray(input_adata.obsm["spatial"])
    slices = obs_labels(input_adata, slice_col)
    slice_ids = sorted(pd.unique(slices), key=lambda s: int(s))
    gt = obs_labels(input_adata, GT_KEY)
    n_cols = min(5, len(slice_ids))
    n_rows = int(np.ceil(len(slice_ids) / n_cols))
    panel_w, panel_h = 4.4, 4.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows), squeeze=False)
    for j, sid in enumerate(slice_ids):
        ax = axes[j // n_cols][j % n_cols]
        mask = slices == sid
        plot_spatial_panel(ax, coords[mask], gt[mask], f"slice {sid}", palette)
    for j in range(len(slice_ids), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    bottom = add_bottom_legend(fig, palette, gt_cats, panel_h * n_rows)
    fig.suptitle(f"GT ({GT_KEY}) labels: all slices", fontsize=13)
    fig.tight_layout(rect=[0, bottom, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[figure] {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_data", default="/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/processed/cycif_crc1_bm.h5ad")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ntf_dir", default="/home/gongyuqiao/ur_annotation/NTF/mytrain/res/cycif_crc1")
    ap.add_argument("--slice_col", default="slice_id")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    outdir = Path(args.output_dir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    input_adata = None
    input_path = Path(args.input_data)
    if input_path.exists():
        input_adata = sc.read_h5ad(input_path)
        print(f"[input] {input_path}: {input_adata.n_obs} cells x {input_adata.n_vars} markers")

    rows, manifest = [], []
    pred_cache = {}

    for scenario, split_col in SCENARIO_SPLIT.items():
        path = Path(args.ntf_dir) / scenario / "test_results.h5ad"
        adata = load_result(path)
        manifest.append({"scenario": scenario, "method": "NTF", "result_path": str(path),
                         "n_obs": adata.n_obs if adata is not None else "",
                         "split_column": split_col})
        if adata is None:
            for metric in METRICS:
                rows.append({"method": "NTF", "scenario": scenario, "label_source": "direct",
                             "metric": metric, "value": np.nan, "status": "not_applicable"})
            continue
        if input_adata is not None and split_col in input_adata.obs.columns:
            n_expected = int((input_adata.obs[split_col].astype(str) == "test").sum())
            match = "pass" if adata.n_obs == n_expected else f"n_obs {adata.n_obs} != input test {n_expected}"
            print(f"[split] {scenario} ({split_col}): expected {n_expected} test cells -> {match}")
            manifest[-1]["alignment_check"] = f"input split check: {match}"
        if PRED_KEY not in adata.obs.columns:
            warnings.warn(f"{scenario}/NTF: obs['{PRED_KEY}'] missing, skipped")
            continue
        gt = obs_labels(adata, GT_KEY)
        pred = obs_labels(adata, PRED_KEY)
        pred_cache[scenario] = (adata, pred)
        metrics = label_metrics(gt, pred)
        for metric, value in metrics.items():
            rows.append({"method": "NTF", "scenario": scenario, "label_source": "direct",
                         "metric": metric, "value": value, "status": "ok"})
        print(f"[direct] {scenario}/NTF: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "celltyping_metrics.csv", index=False)
    pd.DataFrame(manifest).to_csv(outdir / "manifest.csv", index=False)
    print(f"[table] {outdir / 'celltyping_metrics.csv'} ({len(df)} rows)")

    # figures
    palette_cats = []
    if input_adata is not None:
        palette_cats.append(obs_labels(input_adata, GT_KEY))
    for adata, pred in pred_cache.values():
        palette_cats += [obs_labels(adata, GT_KEY), pred]
    gt_cats, palette = build_global_palette(np.concatenate(palette_cats))

    if input_adata is not None:
        plot_gt_all_slices(input_adata, palette, gt_cats, args.slice_col,
                           figdir / "gt_celltypes_all_slices.png")
    for scenario, (adata, pred) in pred_cache.items():
        plot_spatial_maps_scenario(scenario, adata, pred, args.slice_col, palette, gt_cats,
                                   figdir / f"spatial_maps_{scenario}.png")

    pivot = df.pivot_table(index=["method", "scenario"], columns="metric", values="value", aggfunc="first")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
