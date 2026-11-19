#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Niche typing benchmark for NTF, FEAST and SpatialZ on weMERFISH zebrafish embryo.

Ground-truth niches are ``obs['type']`` (10 classes); niche composition is
measured on the fine-grained ``obs['clusters']`` (51 classes), with
``obs['tissue']`` (39 classes) as a sensitivity analysis. Each method
contributes one niche label set per scenario:

- NTF: direct model-side labels ``obs['pred_type']`` (built-in kNN annotator,
  k=5 majority vote);
- SpatialZ: direct labels ``obs['pred_label']`` (1-NN inverse-distance
  weighted) from the ``<scenario>_niche`` runs (``--cell_type_key type``);
- FEAST: leiden clustering on its reconstructed expression
  (``layers['prediction']``), since FEAST has no direct-label capability.
  Resolution is auto-selected so the cluster count falls in [8, 15]
  (candidates: 1.0, then 0.5, 1.5, 2.0, 0.3, 3.0); the chosen resolution is
  recorded in the metrics table and the log.

Evaluation (all on test cells):

(a) niche labelling accuracy: pred niche vs GT ``type`` -> ARI / NMI / FMI;
(b) niche composition fidelity: for each niche n the composition vector
    c(n) = fraction of cells per fine cluster. GT composition from
    GT ``type`` x ``clusters``; predicted composition from pred niche labels
    x true ``clusters``. Direct labels are matched to GT niches by name;
    FEAST clusters are matched by Hungarian assignment on cell overlap
    (unmatched clusters are reported but excluded from matched-niche means).
    Per-niche cosine similarity and Jensen-Shannon divergence (log2, in
    [0, 1]); overall niche-frequency Pearson/Spearman correlation and total
    variation distance. Everything is computed twice: composition on
    ``clusters`` (main) and on ``tissue`` (sensitivity).

senario1/senario2 are evaluated for all three methods after an alignment check
(cell count, coordinates, GT labels); senario3/head/tail are NTF-only
extrapolation scenarios. Missing result files are skipped gracefully and
marked ``not_applicable`` in the metrics table.
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
METHODS = ["NTF", "FEAST", "SpatialZ", "UniST"]
NICHE_GT_KEY = "type"
COMPOSITION_LEVELS = ["clusters", "tissue"]  # main metric + sensitivity
DIRECT_LABEL_KEY = {"NTF": "pred_type", "SpatialZ": "pred_label", "UniST": "pred_label"}
LABEL_SOURCE = {
    "NTF": "direct_pred_type_knn5",
    "SpatialZ": "direct_pred_label_1nn_idw",
    "FEAST": "leiden_fallback",
    "UniST": "direct_pred_label_unist_hybrid",
}
LABEL_METRICS = ("ari", "nmi", "fmi")
COMPOSITION_METRICS = (
    "mean_cosine", "median_cosine", "mean_js", "median_js",
    "freq_pearson", "freq_spearman", "tv_distance",
)
# FEAST fallback: first try res=1.0, then these candidates; target cluster
# count in [8, 15] (GT niches = 10)
LEIDEN_TARGET_RANGE = (8, 15)
LEIDEN_RESOLUTION_CANDIDATES = [0.5, 1.5, 2.0, 0.3, 3.0]
UNMATCHED_PREFIX = "unmatched::"


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
    """Verify the methods share the same test cells for a scenario.

    Checks cell count, spatial coordinates and both GT label sets
    (``type`` and ``clusters``); any mismatch aborts the run.
    """
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
        for gt_key in (NICHE_GT_KEY, COMPOSITION_LEVELS[0]):
            gt_ref = obs_labels(ref, gt_key)
            gt_other = obs_labels(other, gt_key)
            if ref.obs[gt_key].isna().any() or other.obs[gt_key].isna().any():
                ok = False
                messages.append(f"missing GT labels detected in '{gt_key}'")
            elif not np.array_equal(gt_ref, gt_other):
                ok = False
                messages.append(f"GT labels ('{gt_key}') differ between {names[0]} and {name}")
    status = "pass" if ok else "FAIL: " + "; ".join(messages)
    print(f"[alignment] {scenario}: {status} ({', '.join(names)}, n_obs={ref.n_obs})")
    return ok, status


def cluster_labels_auto_resolution(prediction, seed):
    """Leiden clustering on the reconstructed expression (no PCA).

    Tries resolution 1.0 first, then LEIDEN_RESOLUTION_CANDIDATES, and keeps
    the first whose cluster count falls in LEIDEN_TARGET_RANGE. Returns
    (labels, finite_mask, resolution, n_clusters).
    """
    prediction = np.asarray(prediction, dtype=np.float32)
    finite = np.isfinite(prediction).all(axis=1)
    n_dropped = int((~finite).sum())
    if n_dropped:
        warnings.warn(f"{n_dropped} cells with NaN/Inf prediction dropped from clustering")
    sub = ad.AnnData(X=prediction[finite])
    sc.pp.neighbors(sub, n_neighbors=15)
    lo, hi = LEIDEN_TARGET_RANGE
    chosen = None
    for resolution in [1.0] + LEIDEN_RESOLUTION_CANDIDATES:
        sc.tl.leiden(sub, resolution=resolution, random_state=seed, key_added="leiden_tmp")
        n_clusters = sub.obs["leiden_tmp"].nunique()
        print(f"[leiden] resolution={resolution}: {n_clusters} clusters (target [{lo}, {hi}])")
        if lo <= n_clusters <= hi:
            chosen = (resolution, n_clusters)
            break
    if chosen is None:
        warnings.warn(
            f"no leiden resolution reached the target range {LEIDEN_TARGET_RANGE}; "
            "falling back to resolution=1.0"
        )
        sc.tl.leiden(sub, resolution=1.0, random_state=seed, key_added="leiden_tmp")
        chosen = (1.0, int(sub.obs["leiden_tmp"].nunique()))
    resolution, n_clusters = chosen
    print(f"[leiden] using resolution={resolution} ({n_clusters} clusters)")
    return sub.obs["leiden_tmp"].astype(str).to_numpy(), finite, resolution, n_clusters


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


def match_labels(pred_labels, gt_labels, method):
    """Match predicted niche labels to GT niche names.

    Direct labels match by name; FEAST cluster ids are matched by Hungarian
    assignment on cell overlap. Returns (mapping pred_label -> gt_name,
    unmatched_pred_labels).
    """
    pred_cats = sorted(pd.unique(pred_labels))
    gt_cats = sorted(pd.unique(gt_labels))
    if LABEL_SOURCE[method] != "leiden_fallback":
        mapping = {c: c for c in pred_cats if c in gt_cats}
        unmatched = [c for c in pred_cats if c not in gt_cats]
        return mapping, unmatched
    overlap = pd.crosstab(pd.Series(pred_labels, name="pred"),
                          pd.Series(gt_labels, name="gt"))
    overlap = overlap.reindex(index=pred_cats, columns=gt_cats, fill_value=0)
    row_ind, col_ind = linear_sum_assignment(-overlap.to_numpy())
    mapping = {}
    matched_pred = set()
    print(f"[hungarian] {method} cluster -> GT niche (overlap cells):")
    for r, c in zip(row_ind, col_ind):
        pred_cat, gt_cat = overlap.index[r], overlap.columns[c]
        mapping[pred_cat] = gt_cat
        matched_pred.add(pred_cat)
        print(f"    cluster {pred_cat:>3s} -> {gt_cat:<25s} ({int(overlap.iloc[r, c])} cells)")
    unmatched = [c for c in pred_cats if c not in matched_pred]
    if unmatched:
        print(f"    unmatched clusters (excluded from matched-niche means): {unmatched}")
    return mapping, unmatched


def evaluate_composition(pred_labels, gt_niche, comp_labels, mapping, unmatched):
    """Niche composition fidelity for one composition level.

    ``mapping``/``unmatched`` come from ``match_labels`` (computed once per
    method/scenario, shared by all composition levels). Returns
    (per_niche_rows, composition_long_rows, overall_metrics). All arrays are
    already restricted to valid cells.
    """
    comp_gt = composition_matrix(gt_niche, comp_labels)
    comp_pred = composition_matrix(pred_labels, comp_labels)
    n_total = len(gt_niche)
    counts_gt = pd.Series(gt_niche).value_counts()
    counts_pred = pd.Series(pred_labels).value_counts()

    per_niche_rows, comp_rows = [], []
    cosines, jss = [], []

    def emit_comp(niche, comp, frac, source):
        comp_rows.append({"niche": niche, "cluster": str(comp),
                          "fraction": float(frac), "source": source})

    # overall niche frequency vectors over the union of matched niche names
    freq_niches = sorted(set(mapping.values()) | set(comp_gt.index))
    for gt_name in freq_niches:
        c_gt = comp_gt.loc[gt_name] if gt_name in comp_gt.index else pd.Series(dtype=float)
        pred_matches = [p for p, g in mapping.items() if g == gt_name]
        pred_name = pred_matches[0] if pred_matches else None
        c_pred = comp_pred.loc[pred_name] if pred_name is not None else pd.Series(dtype=float)
        # align both vectors to the union of components
        components = c_gt.index.union(c_pred.index)
        v_gt = c_gt.reindex(components, fill_value=0.0).to_numpy()
        v_pred = c_pred.reindex(components, fill_value=0.0).to_numpy()
        cos = cosine_similarity(v_pred, v_gt)
        js = js_divergence(v_pred, v_gt) if v_pred.sum() > 0 else np.nan
        n_pred = int(counts_pred.get(pred_name, 0)) if pred_name is not None else 0
        n_gt = int(counts_gt.get(gt_name, 0))
        cosines.append(cos)
        jss.append(js)
        per_niche_rows.append({
            "niche": gt_name, "pred_label": pred_name if pred_name is not None else "",
            "cosine": cos, "js_divergence": js,
            "n_cells_pred": n_pred, "n_cells_gt": n_gt,
        })
        for component in components:
            emit_comp(gt_name, component, c_pred.get(component, 0.0), "pred")
    # unmatched clusters: composition reported, excluded from matched means
    for pred_name in unmatched:
        for component, frac in comp_pred.loc[pred_name].items():
            emit_comp(f"{UNMATCHED_PREFIX}{pred_name}", component, frac, "pred")
    p = np.array([counts_gt.get(n, 0) / n_total for n in freq_niches])
    q = np.array([
        sum(counts_pred.get(pp, 0) for pp, g in mapping.items() if g == n) / n_total
        for n in freq_niches
    ])
    overall = {
        "mean_cosine": float(np.nanmean(cosines)),
        "median_cosine": float(np.nanmedian(cosines)),
        "mean_js": float(np.nanmean(jss)),
        "median_js": float(np.nanmedian(jss)),
        "freq_pearson": float(np.corrcoef(p, q)[0, 1]),
        "freq_spearman": float(pd.Series(p).corr(pd.Series(q), method="spearman")),
        "tv_distance": float(0.5 * np.abs(p - q).sum()),
    }
    return per_niche_rows, comp_rows, overall, len(unmatched)


def metric_row(method, scenario, composition_level, metric, value, status):
    return {
        "method": method, "scenario": scenario,
        "label_source": LABEL_SOURCE[method],
        "composition_level": composition_level,
        "metric": metric, "value": value, "status": status,
    }


def not_applicable_rows(method, scenario):
    rows = [metric_row(method, scenario, "na", m, np.nan, "not_applicable")
            for m in LABEL_METRICS]
    for level in COMPOSITION_LEVELS:
        rows += [metric_row(method, scenario, level, m, np.nan, "not_applicable")
                 for m in COMPOSITION_METRICS]
    return rows


def result_path(method, scenario, args):
    """Result file for one method/scenario (SpatialZ niche runs carry a
    ``_niche`` suffix)."""
    if method == "NTF":
        return Path(args.ntf_dir) / scenario / "test_results.h5ad"
    if method == "FEAST":
        return Path(args.feast_dir) / scenario / "test_results.h5ad"
    if method == "UniST":
        return Path(args.unist_dir) / f"{scenario}_niche" / "test_results.h5ad"
    return Path(args.spatialz_dir) / f"{scenario}_niche" / "test_results.h5ad"


def get_method_labels(adata, method, scenario, args, feast_log):
    """Return (labels, valid_mask) for one method; (None, None) if unavailable."""
    if LABEL_SOURCE[method] != "leiden_fallback":
        key = DIRECT_LABEL_KEY[method]
        if key not in adata.obs.columns:
            warnings.warn(f"{scenario}/{method}: obs['{key}'] missing, skipped")
            return None, None
        return obs_labels(adata, key), np.ones(adata.n_obs, dtype=bool)
    if "prediction" not in adata.layers:
        warnings.warn(f"{scenario}/{method}: layers['prediction'] missing, clustering skipped")
        return None, None
    labels, finite, resolution, n_clusters = cluster_labels_auto_resolution(
        adata.layers["prediction"], args.seed)
    feast_log.append({"scenario": scenario, "leiden_resolution": resolution,
                      "n_clusters": n_clusters})
    return labels, finite


def evaluate_scenario(scenario, adatas, args, rows, per_niche_rows, comp_rows,
                      label_cache, feast_log):
    ref = next(a for a in adatas.values() if a is not None)
    gt_niche_full = obs_labels(ref, NICHE_GT_KEY)
    # GT composition rows (once per scenario x composition level, method="GT")
    for level in COMPOSITION_LEVELS:
        comp_gt = composition_matrix(gt_niche_full, obs_labels(ref, level))
        for niche, row in comp_gt.iterrows():
            for component, frac in row.items():
                comp_rows.append({
                    "scenario": scenario, "method": "GT",
                    "label_source": "ground_truth", "composition_level": level,
                    "niche": niche, "cluster": str(component),
                    "fraction": float(frac), "source": "gt",
                })
    for method in METHODS:
        adata = adatas.get(method)
        if adata is None:
            rows.extend(not_applicable_rows(method, scenario))
            continue
        pred, valid = get_method_labels(adata, method, scenario, args, feast_log)
        if pred is None:
            rows.extend(not_applicable_rows(method, scenario))
            continue
        if len(pred) != int(valid.sum()):
            raise ValueError(f"{scenario}/{method}: label length mismatch")
        gt_niche = gt_niche_full[valid]
        # (a) niche labelling accuracy
        metrics = label_metrics(gt_niche, pred)
        for metric, value in metrics.items():
            rows.append(metric_row(method, scenario, "na", metric, value, "ok"))
        if LABEL_SOURCE[method] == "leiden_fallback":
            res = feast_log[-1]["leiden_resolution"]
            rows.append(metric_row(method, scenario, "na", "leiden_resolution", res, "ok"))
        print(f"[{LABEL_SOURCE[method]}] {scenario}/{method}: "
              + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        # label matching (once per method/scenario, shared by both levels)
        mapping, unmatched = match_labels(pred, gt_niche, method)
        # (b) niche composition fidelity, per composition level
        for level in COMPOSITION_LEVELS:
            comp_labels = obs_labels(adata, level)[valid]
            pn_rows, c_rows, overall, n_unmatched = evaluate_composition(
                pred, gt_niche, comp_labels, mapping, unmatched)
            for r in pn_rows:
                r.update({"scenario": scenario, "method": method,
                          "label_source": LABEL_SOURCE[method],
                          "composition_level": level})
            per_niche_rows.extend(pn_rows)
            for r in c_rows:
                r.update({"scenario": scenario, "method": method,
                          "label_source": LABEL_SOURCE[method],
                          "composition_level": level})
            comp_rows.extend(c_rows)
            for metric, value in overall.items():
                rows.append(metric_row(method, scenario, level, metric, value, "ok"))
            print(f"  [composition:{level}] {scenario}/{method}: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in overall.items())
                  + (f" | {n_unmatched} unmatched clusters" if n_unmatched else ""))
        full = np.full(len(gt_niche_full), "NA", dtype=object)
        full[valid] = pred
        label_cache[(scenario, method)] = full.astype(str)


def build_global_palette(categories):
    """One fixed category -> color mapping shared by ALL figures of a run.

    Colors come from the concatenated tab20/tab20b/tab20c palettes (60 distinct
    colors), so every niche keeps the same color in every scenario figure.
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
    """GT + per-method niche labels, faceted by the scenario's test slices."""
    ref = next(a for a in adatas.values() if a is not None)
    coords = np.asarray(ref.obsm["spatial"])
    slices = obs_labels(ref, args_global.slice_col)
    slice_ids = sorted(pd.unique(slices), key=lambda s: int(s))
    if len(slice_ids) > 6:
        idx = np.linspace(0, len(slice_ids) - 1, 6).round().astype(int)
        slice_ids = [slice_ids[i] for i in idx]  # plot at most 6 evenly spaced test slices
    gt = obs_labels(ref, NICHE_GT_KEY)
    panels = [("GT (type)", gt)]
    for method in METHODS:
        key = (scenario, method)
        if key in label_cache and adatas.get(method) is not None:
            source = LABEL_SOURCE[method]
            tag = {"leiden_fallback": "leiden"}.get(source, DIRECT_LABEL_KEY.get(method, "direct"))
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
    fig.suptitle(f"{scenario}: GT vs method niche labels (test slices)", fontsize=13)
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
               fontsize=7, frameon=False, ncol=n_leg_col, title="GT niche (type)",
               title_fontsize=8, columnspacing=0.8, handletextpad=0.3)
    return bottom


def plot_gt_all_slices(input_adata, palette, gt_cats, slice_col, outpath, max_panels=36):
    """GT niche map for every slice of the input data (all cells, train+test)."""
    coords = np.asarray(input_adata.obsm["spatial"])
    slices = obs_labels(input_adata, slice_col)
    slice_ids = sorted(pd.unique(slices), key=lambda s: int(s))
    if len(slice_ids) > max_panels:
        idx = np.linspace(0, len(slice_ids) - 1, max_panels).round().astype(int)
        slice_ids = [slice_ids[i] for i in idx]
        print(f"[figure] too many slices, plotting {max_panels} evenly spaced ones")
    gt = obs_labels(input_adata, NICHE_GT_KEY)
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
    fig.suptitle("GT niches (type): all slices", fontsize=13)
    fig.tight_layout(rect=[0, bottom, 1, 0.96])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[figure] {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_data", default="mytrain/data/wemerfish_zebrafish_embryo/processed/wemerfish_C6s_E2.h5ad")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ntf_dir", default="mytrain/res/wemerfish_C6s_E2")
    ap.add_argument("--feast_dir", default="mytrain/res/FEAST/wemerfish_C6s_E2")
    ap.add_argument("--spatialz_dir", default="/home/gongyuqiao/ur_annotation/SpatialZ/res/wemerfish_C6s_E2")
    ap.add_argument("--unist_dir", default="mytrain/res/UniST/wemerfish_C6s_E2")
    ap.add_argument("--slice_col", default="slice_id",
                    help="obs column identifying slices for split check and plot facets")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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
        niche_sizes = input_obs[NICHE_GT_KEY].value_counts()
        print(f"[input] GT niches ({NICHE_GT_KEY}): {len(niche_sizes)} classes, "
              f"smallest: {niche_sizes.idxmin()} ({niche_sizes.min()} cells) -- "
              "tiny niches (e.g. PGC) are kept and flagged in the summary")
    else:
        warnings.warn(f"Input data missing, split-column cross-check skipped: {input_path}")

    rows, per_niche_rows, comp_rows, manifest, feast_log = [], [], [], [], []
    label_cache = {}
    loaded = {}

    for scenario, split_col in SCENARIO_SPLIT.items():
        adatas = {}
        for method in METHODS:
            path = result_path(method, scenario, args)
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
            rows.extend(r for m in METHODS for r in not_applicable_rows(m, scenario))
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
        evaluate_scenario(scenario, available, args, rows, per_niche_rows,
                          comp_rows, label_cache, feast_log)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "nichetyping_metrics.csv", index=False)
    df_pn = pd.DataFrame(per_niche_rows)
    df_pn.to_csv(outdir / "niche_per_niche_metrics.csv", index=False)
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(outdir / "niche_composition_long.csv", index=False)
    pd.DataFrame(manifest).to_csv(outdir / "manifest.csv", index=False)
    if feast_log:
        pd.DataFrame(feast_log).to_csv(outdir / "feast_leiden_resolution.csv", index=False)
    print(f"[table] {outdir / 'nichetyping_metrics.csv'} ({len(df)} rows)")
    print(f"[table] {outdir / 'niche_per_niche_metrics.csv'} ({len(df_pn)} rows)")
    print(f"[table] {outdir / 'niche_composition_long.csv'} ({len(df_comp)} rows)")
    print(f"[table] {outdir / 'manifest.csv'}")

    # one global palette for every figure: GT niche categories (sorted) get
    # the first colors, method-specific extra labels (FEAST cluster ids, "NA")
    # take colors after them, so GT niche colors match the 3D viewer exactly
    # (viewer maps sorted GT categories onto tab20/tab20b/tab20c)
    gt_set = set()
    if input_obs is not None and NICHE_GT_KEY in input_obs.columns:
        gt_set |= set(input_obs[NICHE_GT_KEY].astype(str).to_numpy())
    for a_s in loaded.values():
        for a in a_s.values():
            if a is not None:
                gt_set |= set(obs_labels(a, NICHE_GT_KEY))
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
                           figdir / "gt_niches_all_slices.png")

    for scenario, adatas in loaded.items():
        if any(a is not None for a in adatas.values()):
            plot_spatial_maps_scenario(scenario, adatas, label_cache, palette, gt_cats_legend,
                                       figdir / f"spatial_maps_{scenario}.png")

    pivot = df.pivot_table(index=["method", "scenario", "label_source", "composition_level"],
                           columns="metric", values="value", aggfunc="first")
    print(pivot.round(4).to_string())


if __name__ == "__main__":
    main()
