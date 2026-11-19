#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cell typing benchmark for NTF, FEAST, SpatialZ and UniST on Deep-STARmap mouse brain data.

Each method contributes one label set per scenario, compared with the
ground-truth ``Harmony_labels`` labels via ARI / NMI / FMI:

- NTF: direct model-side labels ``obs['pred_Harmony_labels']`` (built-in kNN annotator);
- SpatialZ: direct labels ``obs['pred_label']`` (native label transfer);
- UniST: direct labels ``obs['pred_label']`` (native label transfer);
- FEAST: leiden clustering on its reconstructed expression
  (``layers['prediction']``), since FEAST has no direct-label capability.

senario1/senario2 are evaluated for all four methods after an alignment check
(cell count, coordinates, GT labels); senario3/head/tail are NTF-only
extrapolation scenarios (FEAST/SpatialZ/UniST have no result there).

Besides the per-scenario ARI/NMI/FMI on all test cells, per-test-slice metrics
are written to ``celltyping_per_slice.csv`` and their cross-slice std is
appended to ``celltyping_metrics.csv`` as ``*_std`` metric rows.

For plotting, clustering-based labels (FEAST) are renamed to the GT category
with maximum cell overlap (Hungarian matching on the cluster x GT contingency
table), so cluster colors line up with the GT color scheme; metrics are always
computed on the raw labels (ARI/NMI/FMI are permutation-invariant).
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
import anndata as ad
from scipy.optimize import linear_sum_assignment
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
GT_KEY = "Harmony_labels"
DIRECT_LABEL_KEY = {"NTF": "pred_Harmony_labels", "SpatialZ": "pred_label", "UniST": "pred_label"}
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
        gt_ref = obs_labels(ref, GT_KEY)
        gt_other = obs_labels(other, GT_KEY)
        if ref.obs[GT_KEY].isna().any() or other.obs[GT_KEY].isna().any():
            ok = False
            messages.append("missing GT labels detected")
        elif not np.array_equal(gt_ref, gt_other):
            ok = False
            messages.append(f"GT labels differ between {names[0]} and {name}")
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


def map_clusters_to_gt(gt, clusters):
    """Rename cluster ids to GT categories by maximum cell overlap.

    Hungarian assignment on the cluster x GT contingency table gives each
    cluster a distinct GT category where possible; leftover clusters (more
    clusters than GT categories) fold to their own top-overlap GT category.
    Used only for coloring spatial maps, so clustering-based maps line up
    with the GT color scheme.
    """
    ct = pd.crosstab(pd.Series(np.asarray(clusters).astype(str), name="cluster"),
                     pd.Series(np.asarray(gt).astype(str), name="gt"))
    row_ind, col_ind = linear_sum_assignment(-ct.values)
    mapping = {ct.index[r]: ct.columns[c] for r, c in zip(row_ind, col_ind)}
    for cl in ct.index:
        if cl not in mapping:
            mapping[cl] = ct.columns[int(np.argmax(ct.loc[cl].values))]
    return np.asarray([mapping[c] for c in np.asarray(clusters).astype(str)])


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


def evaluate_scenario(scenario, adatas, args, rows, label_cache, slice_rows):
    ref = next(a for a in adatas.values() if a is not None)
    gt = obs_labels(ref, GT_KEY)
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
        plot_labels = pred
        if LABEL_SOURCE[method] == "clustering":
            plot_labels = map_clusters_to_gt(gt[valid], pred)
        full = np.full(len(gt), "NA", dtype=object)
        full[valid] = plot_labels
        label_cache[(scenario, method)] = full.astype(str)
        print(f"[{LABEL_SOURCE[method]}] {scenario}/{method}: "
              + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        # per-test-slice metrics (raw labels) for cross-slice std
        slices_v = obs_labels(ref, args_global.slice_col)[valid]
        gt_v = gt[valid]
        for sid in sorted(pd.unique(slices_v), key=lambda s: int(s)):
            smask = slices_v == sid
            slice_metrics = label_metrics(gt_v[smask], pred[smask])
            for metric, value in slice_metrics.items():
                slice_rows.append({
                    "method": method, "scenario": scenario,
                    "label_source": LABEL_SOURCE[method], "slice_id": sid,
                    "metric": metric, "value": value,
                })


def scenario_palette(labels):
    cats = sorted(pd.unique(labels))
    cmap = plt.get_cmap("tab20")
    return cats, {c: cmap(i % 20) for i, c in enumerate(cats)}


def build_global_palette(categories):
    """One fixed category -> color mapping shared by ALL figures of a run.

    Colors come from the concatenated tab20/tab20b/tab20c palettes (60 distinct
    colors), so every cell type keeps the same color in every scenario figure.
    """
    colors = (list(plt.get_cmap("tab20").colors)
              + list(plt.get_cmap("tab20b").colors)
              + list(plt.get_cmap("tab20c").colors))
    cats = sorted(pd.unique(np.asarray(categories, dtype=str)))
    return cats, {c: colors[i % len(colors)] for i, c in enumerate(cats)}


def plot_spatial_panel(ax, coords, labels, title, palette, point_size=1.5):
    for cat in palette:
        mask = labels == cat
        ax.scatter(coords[mask, 0], coords[mask, 1], s=point_size, c=[palette[cat]], linewidths=0, rasterized=True)
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


args_global = None


def plot_spatial_maps_scenario(scenario, adatas, label_cache, palette, gt_cats, outpath):
    """GT + per-method labels, faceted by the scenario's test slices."""
    ref = next(a for a in adatas.values() if a is not None)
    coords = np.asarray(ref.obsm["spatial"])
    slices = obs_labels(ref, args_global.slice_col)
    slice_ids = sorted(pd.unique(slices), key=lambda s: int(s))
    if len(slice_ids) > 6:
        idx = np.linspace(0, len(slice_ids) - 1, 6).round().astype(int)
        slice_ids = [slice_ids[i] for i in idx]  # plot at most 6 evenly spaced test slices
    gt = obs_labels(ref, GT_KEY)
    panels = [(f"GT ({GT_KEY})", gt)]
    for method in METHODS:
        key = (scenario, method)
        if key in label_cache and adatas.get(method) is not None:
            source = LABEL_SOURCE[method]
            tag = {"direct": DIRECT_LABEL_KEY.get(method, "direct"),
                   "clustering": "clustering (GT-mapped colors)"}[source]
            panels.append((f"{method} {tag}", label_cache[key]))
    n_rows, n_cols = len(panels), len(slice_ids)
    panel_w, panel_h = 4.4, 4.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows), squeeze=False)
    for i, (name, labels) in enumerate(panels):
        for j, sid in enumerate(slice_ids):
            mask = slices == sid
            if i == 0:
                title = f"{name}\n{sid}" if j == 0 else str(sid)
            else:
                ari = adjusted_rand_score(gt[mask], labels[mask])
                title = (f"{name}\n{sid}, ARI={ari:.3f}" if j == 0
                         else f"{sid}, ARI={ari:.3f}")
            plot_spatial_panel(axes[i][j], coords[mask], labels[mask], title, palette)
    bottom = add_bottom_legend(fig, palette, gt_cats, panel_h * n_rows)
    fig.suptitle(f"{scenario}: GT vs method labels (test slices)", fontsize=13)
    fig.tight_layout(rect=[0, bottom, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[figure] {outpath}")


def add_bottom_legend(fig, palette, gt_cats, fig_h_in):
    """Shared bottom legend (4 columns, wrapped names), anchored at the figure
    bottom so it never overlaps panels nor gets clipped."""
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


def plot_gt_all_slices(input_adata, palette, gt_cats, slice_col, outpath, max_panels=36):
    """GT label map for every slice of the input data (all cells, train+test)."""
    coords = np.asarray(input_adata.obsm["spatial"])
    slices = obs_labels(input_adata, slice_col)
    slice_ids = sorted(pd.unique(slices), key=lambda s: int(s))
    if len(slice_ids) > max_panels:
        idx = np.linspace(0, len(slice_ids) - 1, max_panels).round().astype(int)
        slice_ids = [slice_ids[i] for i in idx]
        print(f"[figure] too many slices, plotting {max_panels} evenly spaced ones")
    gt = obs_labels(input_adata, GT_KEY)
    n_cols = min(4, len(slice_ids))
    n_rows = int(np.ceil(len(slice_ids) / n_cols))
    panel_w, panel_h = 4.4, 4.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows), squeeze=False)
    for j, sid in enumerate(slice_ids):
        ax = axes[j // n_cols][j % n_cols]
        mask = slices == sid
        plot_spatial_panel(ax, coords[mask], gt[mask], str(sid), palette)
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
    ap.add_argument("--input_data", default="mytrain/data/deep_starmap_brain/processed/deep_starmap_brain.h5ad")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ntf_dir", default="mytrain/res/deep_starmap_brain")
    ap.add_argument("--feast_dir", default="mytrain/res/FEAST/deep_starmap_brain")
    ap.add_argument("--spatialz_dir", default="/home/gongyuqiao/ur_annotation/SpatialZ/res/deep_starmap_brain")
    ap.add_argument("--unist_dir", default="mytrain/res/UniST/deep_starmap_brain")
    ap.add_argument("--slice_col", default="slice_id",
                    help="obs column identifying slices for split check and plot facets (v2: slice_id_v2)")
    ap.add_argument("--gt_key", default="Harmony_labels", help="obs column with ground-truth labels")
    ap.add_argument("--leiden_resolution", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global GT_KEY, DIRECT_LABEL_KEY
    GT_KEY = args.gt_key
    DIRECT_LABEL_KEY = {"NTF": f"pred_{args.gt_key}", "SpatialZ": "pred_label", "UniST": "pred_label"}
    np.random.seed(args.seed)
    sc.settings.verbosity = 1
    outdir = Path(args.output_dir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    global args_global
    args_global = args
    input_path = Path(args.input_data)
    input_obs = None
    if input_path.exists():
        input_adata = sc.read_h5ad(input_path)
        input_obs = input_adata.obs
        print(f"[input] {input_path}: {input_adata.n_obs} cells x {input_adata.n_vars} genes")
    else:
        warnings.warn(f"Input data missing, split-column cross-check skipped: {input_path}")

    method_dirs = {"NTF": args.ntf_dir, "FEAST": args.feast_dir, "SpatialZ": args.spatialz_dir, "UniST": args.unist_dir}
    rows, manifest, slice_rows = [], [], []
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
            test_slices = sorted(input_obs.loc[input_obs[split_col].astype(str) == "test", args.slice_col].astype(str).unique(), key=lambda s: int(s))
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
        evaluate_scenario(scenario, available, args, rows, label_cache, slice_rows)

    slice_df = pd.DataFrame(slice_rows)
    slice_df.to_csv(outdir / "celltyping_per_slice.csv", index=False)
    # cross-slice std rows (metric suffix "_std") appended to the main table
    std_df = (slice_df.groupby(["method", "scenario", "label_source", "metric"])["value"]
              .std().reset_index())
    std_df["metric"] = std_df["metric"] + "_std"
    std_df["status"] = "ok (std across test slices)"
    df = pd.concat([pd.DataFrame(rows), std_df], ignore_index=True)
    df.to_csv(outdir / "celltyping_metrics.csv", index=False)
    pd.DataFrame(manifest).to_csv(outdir / "manifest.csv", index=False)
    print(f"[table] {outdir / 'celltyping_metrics.csv'} ({len(df)} rows)")
    print(f"[table] {outdir / 'celltyping_per_slice.csv'} ({len(slice_df)} rows)")
    print(f"[table] {outdir / 'manifest.csv'}")

    # one global palette for every figure: GT categories (sorted) get the
    # first colors, method-specific extra labels (FEAST cluster ids, "NA")
    # take colors after them, so GT colors match the 3D viewer exactly
    # (viewer maps sorted GT categories onto tab20/tab20b/tab20c)
    gt_set = set()
    if input_obs is not None and GT_KEY in input_obs.columns:
        gt_set |= set(input_obs[GT_KEY].astype(str).to_numpy())
    for a_s in loaded.values():
        for a in a_s.values():
            if a is not None:
                gt_set |= set(obs_labels(a, GT_KEY))
    extra_set = set()
    for labels in label_cache.values():
        extra_set |= set(np.asarray(labels, dtype=str))
    palette_cats = sorted(gt_set) + sorted(extra_set - gt_set)
    # NOTE: build directly — build_global_palette() re-sorts its input, which
    # would destroy the GT-first order
    _colors = (list(plt.get_cmap("tab20").colors)
               + list(plt.get_cmap("tab20b").colors)
               + list(plt.get_cmap("tab20c").colors))
    gt_cats = list(palette_cats)
    palette = {c: _colors[i % len(_colors)] for i, c in enumerate(gt_cats)}
    # legend only covers GT categories (method-specific cluster ids share the palette)
    gt_cats_legend = [c for c in gt_cats if c in gt_set and not c.isdigit() and c != "NA"]

    if input_obs is not None:
        plot_gt_all_slices(input_adata, palette, gt_cats_legend, args.slice_col,
                           figdir / "gt_all_slices.png")

    for scenario, adatas in loaded.items():
        if any(a is not None for a in adatas.values()):
            plot_spatial_maps_scenario(scenario, adatas, label_cache, palette, gt_cats_legend,
                                       figdir / f"spatial_maps_{scenario}.png")

    pivot = df.pivot_table(index=["method", "scenario", "label_source"], columns="metric", values="value", aggfunc="first")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
