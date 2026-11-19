#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Niche typing evaluation for NTF on the t-CyCIF CRC1 3D atlas.

Adapted from analysis/wemerfish_zebrafish_embryo/eval_nichetyping_wemerfish.py.
Differences from the weMERFISH version:

- NTF only (SpatialZ/FEAST infeasible at 12.8M cells — see the experiment's
  baseline_feasibility section), so no leiden fallback and no Hungarian
  matching: predicted niche labels match GT niches by name.
- Ground-truth niches are ``obs['cell_type_main']`` (~9 functional groups,
  the operational niche definition); niche composition is measured on the
  fine-grained ``obs['cell_type']`` (21 types, Table S4 of Lin et al. 2023).
- NTF labels: direct model-side ``obs['pred_cell_type_main']`` (built-in
  kNN annotator, k=5 majority vote).
- Spatial-map panels are subsampled to <=100k points per slice (CRC1 slices
  hold ~0.5M cells each).

Evaluation (all on test cells):

(a) niche labelling accuracy: pred niche vs GT ``cell_type_main`` ->
    ARI / NMI / FMI;
(b) niche composition fidelity: for each niche n the composition vector
    c(n) = fraction of cells per fine cell type. GT composition from
    GT ``cell_type_main`` x ``cell_type``; predicted composition from pred
    niche labels x true ``cell_type``. Per-niche cosine similarity and
    Jensen-Shannon divergence (log2, in [0, 1]); overall niche-frequency
    Pearson/Spearman correlation and total variation distance.

Usage:
    python eval_nichetyping_cycif.py --output_dir <out>
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
METHODS = ["NTF"]
NICHE_GT_KEY = "cell_type_main"
COMPOSITION_LEVELS = ["cell_type"]
DIRECT_LABEL_KEY = {"NTF": "pred_cell_type_main"}
LABEL_SOURCE = {"NTF": "direct_pred_cell_type_main_knn5"}
LABEL_METRICS = ("ari", "nmi", "fmi")
COMPOSITION_METRICS = (
    "mean_cosine", "median_cosine", "mean_js", "median_js",
    "freq_pearson", "freq_spearman", "tv_distance",
)
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


def js_divergence(p, q):
    """Jensen-Shannon divergence with log2 base, in [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cosine_similarity(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    denom = np.linalg.norm(p) * np.linalg.norm(q)
    if denom == 0:
        return np.nan
    return float(np.dot(p, q) / denom)


def composition_matrix(niche_labels, comp_labels):
    """niche x component cross table, rows normalized to fractions."""
    ct = pd.crosstab(pd.Series(niche_labels, name="niche"),
                     pd.Series(comp_labels, name="component"))
    ct = ct.div(ct.sum(axis=1), axis=0).fillna(0.0)
    return ct


def evaluate_composition(pred_labels, gt_niche, comp_labels):
    """Niche composition fidelity (direct labels match GT niches by name)."""
    comp_gt = composition_matrix(gt_niche, comp_labels)
    comp_pred = composition_matrix(pred_labels, comp_labels)
    n_total = len(gt_niche)
    counts_gt = pd.Series(gt_niche).value_counts()
    counts_pred = pd.Series(pred_labels).value_counts()

    per_niche_rows, comp_rows = [], []
    cosines, jss = [], []

    freq_niches = sorted(set(comp_gt.index) | set(comp_pred.index))
    for gt_name in freq_niches:
        c_gt = comp_gt.loc[gt_name] if gt_name in comp_gt.index else pd.Series(dtype=float)
        c_pred = comp_pred.loc[gt_name] if gt_name in comp_pred.index else pd.Series(dtype=float)
        components = c_gt.index.union(c_pred.index)
        v_gt = c_gt.reindex(components, fill_value=0.0).to_numpy()
        v_pred = c_pred.reindex(components, fill_value=0.0).to_numpy()
        cos = cosine_similarity(v_pred, v_gt)
        js = js_divergence(v_pred, v_gt) if v_pred.sum() > 0 else np.nan
        cosines.append(cos)
        jss.append(js)
        per_niche_rows.append({
            "niche": gt_name, "pred_label": gt_name,
            "cosine": cos, "js_divergence": js,
            "n_cells_pred": int(counts_pred.get(gt_name, 0)),
            "n_cells_gt": int(counts_gt.get(gt_name, 0)),
        })
        for component in components:
            comp_rows.append({"niche": gt_name, "cluster": str(component),
                              "fraction": float(c_pred.get(component, 0.0)), "source": "pred"})
    p = np.array([counts_gt.get(n, 0) / n_total for n in freq_niches])
    q = np.array([counts_pred.get(n, 0) / n_total for n in freq_niches])
    overall = {
        "mean_cosine": float(np.nanmean(cosines)),
        "median_cosine": float(np.nanmedian(cosines)),
        "mean_js": float(np.nanmean(jss)),
        "median_js": float(np.nanmedian(jss)),
        "freq_pearson": float(np.corrcoef(p, q)[0, 1]),
        "freq_spearman": float(pd.Series(p).corr(pd.Series(q), method="spearman")),
        "tv_distance": float(0.5 * np.abs(p - q).sum()),
    }
    return per_niche_rows, comp_rows, overall


def metric_row(method, scenario, composition_level, metric, value, status):
    return {
        "method": method, "scenario": scenario,
        "label_source": LABEL_SOURCE[method],
        "composition_level": composition_level,
        "metric": metric, "value": value, "status": status,
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
               fontsize=7, frameon=False, ncol=n_leg_col, title=f"GT niche ({NICHE_GT_KEY})",
               title_fontsize=8, columnspacing=0.8, handletextpad=0.3)
    return bottom


def plot_spatial_maps_scenario(scenario, adata, pred_labels, slice_col, palette, gt_cats, outpath):
    """GT vs NTF niche labels, faceted by the scenario's test slices."""
    coords = np.asarray(adata.obsm["spatial"])
    slices = obs_labels(adata, slice_col)
    slice_ids = sorted(pd.unique(slices), key=lambda s: int(s))
    if len(slice_ids) > 6:
        idx = np.linspace(0, len(slice_ids) - 1, 6).round().astype(int)
        slice_ids = [slice_ids[i] for i in idx]
    gt = obs_labels(adata, NICHE_GT_KEY)
    panels = [(f"GT ({NICHE_GT_KEY})", gt), ("NTF pred_cell_type_main", pred_labels)]
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
    fig.suptitle(f"{scenario}: GT vs NTF niche labels (test slices)", fontsize=13)
    fig.tight_layout(rect=[0, bottom, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[figure] {outpath}")


def plot_gt_all_slices(input_adata, palette, gt_cats, slice_col, outpath):
    """GT niche map for every slice of the input data (all cells, train+test)."""
    coords = np.asarray(input_adata.obsm["spatial"])
    slices = obs_labels(input_adata, slice_col)
    slice_ids = sorted(pd.unique(slices), key=lambda s: int(s))
    gt = obs_labels(input_adata, NICHE_GT_KEY)
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
    fig.suptitle(f"GT niches ({NICHE_GT_KEY}): all slices", fontsize=13)
    fig.tight_layout(rect=[0, bottom, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[figure] {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_data", default="/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/processed/cycif_crc1_bm.h5ad")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ntf_dir", default="/home/gongyuqiao/ur_annotation/NTF/mytrain/res/cycif_crc1")
    ap.add_argument("--slice_col", default="slice_id",
                    help="obs column identifying slices for split check and plot facets")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    outdir = Path(args.output_dir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_data)
    input_adata = None
    if input_path.exists():
        input_adata = sc.read_h5ad(input_path)
        print(f"[input] {input_path}: {input_adata.n_obs} cells x {input_adata.n_vars} markers")
        niche_sizes = input_adata.obs[NICHE_GT_KEY].value_counts()
        print(f"[input] GT niches ({NICHE_GT_KEY}): {len(niche_sizes)} classes, "
              f"smallest: {niche_sizes.idxmin()} ({niche_sizes.min()} cells)")

    rows, per_niche_rows, comp_rows, manifest = [], [], [], []
    pred_cache = {}

    for scenario, split_col in SCENARIO_SPLIT.items():
        path = Path(args.ntf_dir) / scenario / "test_results.h5ad"
        adata = load_result(path)
        manifest.append({
            "scenario": scenario, "method": "NTF", "result_path": str(path),
            "n_obs": adata.n_obs if adata is not None else "",
            "split_column": split_col, "label_source": LABEL_SOURCE["NTF"],
        })
        if adata is None:
            for m in LABEL_METRICS:
                rows.append(metric_row("NTF", scenario, "na", m, np.nan, "not_applicable"))
            continue
        if input_adata is not None and split_col in input_adata.obs.columns:
            n_expected = int((input_adata.obs[split_col].astype(str) == "test").sum())
            match = "pass" if adata.n_obs == n_expected else f"n_obs {adata.n_obs} != input test {n_expected}"
            print(f"[split] {scenario} ({split_col}): expected {n_expected} test cells -> {match}")
            manifest[-1]["alignment_check"] = f"input split check: {match}"

        key = DIRECT_LABEL_KEY["NTF"]
        if key not in adata.obs.columns:
            warnings.warn(f"{scenario}/NTF: obs['{key}'] missing, skipped")
            continue
        pred = obs_labels(adata, key)
        gt_niche = obs_labels(adata, NICHE_GT_KEY)
        pred_cache[scenario] = (adata, pred)

        # (a) niche labelling accuracy
        metrics = label_metrics(gt_niche, pred)
        for metric, value in metrics.items():
            rows.append(metric_row("NTF", scenario, "na", metric, value, "ok"))
        print(f"[niche accuracy] {scenario}/NTF: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        # (b) niche composition fidelity
        for level in COMPOSITION_LEVELS:
            comp_labels = obs_labels(adata, level)
            pn_rows, c_rows, overall = evaluate_composition(pred, gt_niche, comp_labels)
            for r in pn_rows:
                r.update({"scenario": scenario, "method": "NTF",
                          "label_source": LABEL_SOURCE["NTF"], "composition_level": level})
            per_niche_rows.extend(pn_rows)
            for r in c_rows:
                r.update({"scenario": scenario, "method": "NTF",
                          "label_source": LABEL_SOURCE["NTF"], "composition_level": level})
            comp_rows.extend(c_rows)
            for metric, value in overall.items():
                rows.append(metric_row("NTF", scenario, level, metric, value, "ok"))
            print(f"  [composition:{level}] {scenario}/NTF: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in overall.items()))

        # GT composition rows (once per scenario x composition level)
        for level in COMPOSITION_LEVELS:
            comp_gt = composition_matrix(gt_niche, obs_labels(adata, level))
            for niche, row in comp_gt.iterrows():
                for component, frac in row.items():
                    comp_rows.append({
                        "scenario": scenario, "method": "GT",
                        "label_source": "ground_truth", "composition_level": level,
                        "niche": niche, "cluster": str(component),
                        "fraction": float(frac), "source": "gt",
                    })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "nichetyping_metrics.csv", index=False)
    pd.DataFrame(per_niche_rows).to_csv(outdir / "niche_per_niche_metrics.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(outdir / "niche_composition_long.csv", index=False)
    pd.DataFrame(manifest).to_csv(outdir / "manifest.csv", index=False)
    print(f"[table] {outdir / 'nichetyping_metrics.csv'} ({len(df)} rows)")

    # figures
    palette_cats = []
    if input_adata is not None:
        palette_cats.append(obs_labels(input_adata, NICHE_GT_KEY))
    for adata, pred in pred_cache.values():
        palette_cats += [obs_labels(adata, NICHE_GT_KEY), pred]
    gt_cats, palette = build_global_palette(np.concatenate(palette_cats))

    if input_adata is not None:
        plot_gt_all_slices(input_adata, palette, gt_cats, args.slice_col,
                           figdir / "gt_niches_all_slices.png")
    for scenario, (adata, pred) in pred_cache.items():
        plot_spatial_maps_scenario(scenario, adata, pred, args.slice_col, palette, gt_cats,
                                   figdir / f"spatial_maps_{scenario}.png")

    pivot = df.pivot_table(index=["method", "scenario", "label_source", "composition_level"],
                           columns="metric", values="value", aggfunc="first")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
