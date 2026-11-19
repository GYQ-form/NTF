#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UniST (fast/inr) patch for the all-scale SVG benchmark (R2.t5).

Adds UniST-fast (19 scales) and UniST-inr (6 scales) to the Moran's I corr /
Top-20 SVG Jaccard evaluation of run_svg_benchmark_scales.py, under the exact
same protocol: NTF reference file drives stratified sampling (sample_size,
seed, by slice_id), the 3D kNN graph (k=15, symmetric binary) and the per-2D-
slice graphs are built from the reference coordinates and shared by all
methods (paired).  UniST result h5ad files have the same layout as the other
methods (layers['truth']/layers['prediction'], obs/slice_id, obsm['spatial']);
the only adaptation is a tolerance-based alignment check, because UniST writes
spatial as float64 while the NTF reference stores float32 (max diff ~4e-6,
pure float32 round-trip; obs order verified identical).

UniST coverage caveat (r2t5 convention): test slices without both-side train
flanking (boundary extrapolation) are skipped by the official API and the
wrapper leaves their prediction zero-filled.  Following the r2t5 centralized
evaluation (eval_scale_unist.py), those zero-filled slices are kept as-is in
the 3D whole-volume metrics; for the per-2D-slice metrics an all-zero
prediction slice has zero variance in every gene and is reported as explicit
NA with status "boundary slice not covered by UniST".

inr tiers without results (60w-90w, >=200w) are explicit NA.

Also recomputes NTF at 10w/100w through this exact code path and compares
against experiment/svg_benchmark_allscales/scale_benchmark_summary.csv with
tolerance 0 (paired check of sampling indices and graph construction).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_svg_benchmark import (  # noqa: E402
    make_knn_graph,
    metric_pair,
    read_rows,
    resolve_expression,
    stratified_indices,
)

SCALES = [f"{i}w" for i in range(10, 101, 10)] + [f"{i}w" for i in range(200, 1001, 100)]
INR_SCALES = {"10w", "20w", "30w", "40w", "50w", "100w"}
MIN_SLICE_POINTS = 30
COORD_ATOL = 1e-4

BASE = Path("/home/gongyuqiao/ur_annotation/NTF/mytrain")
ALLSCALES_DIR = Path("/home/gongyuqiao/ur_annotation/NTF/mytrain/revision/experiment/svg_benchmark_allscales")


def scale_to_k(scale):
    return int(scale[:-1]) * 10


def ntf_path(scale):
    return str(BASE / f"res/simulation/scale_test/{scale}/test_results.h5ad")


def unist_paths(scale):
    root = BASE / "res/UniST/scale_test"
    fast = root / ("10w_fast" if scale == "10w" else scale) / "test_results.h5ad"
    inr = (root / scale / "test_results.h5ad") if scale == "10w" else (root / f"{scale}_inr" / "test_results.h5ad")
    return {"UniST-fast": str(fast), "UniST-inr": (str(inr) if scale in INR_SCALES else None)}


def zero_prediction_slices(path, sample_idx, slice_labels, keep):
    """Slices whose sampled prediction rows are all zero (UniST-skipped boundary)."""
    prediction = read_rows(path, sample_idx, "layers/prediction")[:, keep]
    per_slice = {}
    labels = pd.unique(slice_labels)
    for label in labels:
        local = np.where(slice_labels == label)[0]
        per_slice[str(label)] = bool(np.abs(prediction[local]).sum() == 0.0)
    return per_slice, prediction


def evaluate_scale(scale, sample_size, graph_k, seed, top_k, min_slice_points, ntf_recheck):
    dataset = f"scale_{scale}"
    ref_path = ntf_path(scale)
    ref_meta, ref_truth_dataset = resolve_expression(ref_path, "scale")
    sample_idx = stratified_indices(ref_meta["slice_id"], sample_size, seed)
    coords = ref_meta["spatial"][sample_idx]
    slice_labels = np.asarray(ref_meta["slice_id"])[sample_idx].astype(str)
    graph_3d = make_knn_graph(coords, graph_k)
    keep = np.asarray([not str(name).endswith("_HK") for name in ref_meta["var_names"]])
    reference_truth = read_rows(ref_path, sample_idx, ref_truth_dataset)[:, keep]
    names = ref_meta["var_names"][keep]
    ref_obs = np.asarray([str(x) for x in read_obs_index(ref_path)], dtype=object)

    slices = {label: np.where(slice_labels == label)[0] for label in pd.unique(slice_labels)}
    slice_graphs = {
        label: make_knn_graph(coords[local][:, :2], graph_k)
        for label, local in slices.items() if len(local) >= min_slice_points
    }

    methods = unist_paths(scale)
    if ntf_recheck:
        methods = {"NTF_recheck": ref_path, **methods}

    rows, slice_rows, gene_details = [], [], []
    for method, path in methods.items():
        base_row = {
            "dataset": dataset, "scale_k": scale_to_k(scale), "method": method,
            "Moran_I_corr_3D": np.nan, "SVG_Jaccard_3D": np.nan,
            "Moran_I_corr_2D_mean": np.nan, "Moran_I_corr_2D_std": np.nan,
            "SVG_Jaccard_2D_mean": np.nan, "SVG_Jaccard_2D_std": np.nan,
            "n_slices_evaluated": 0, "TopK": top_k,
            "n_genes": int(keep.sum()), "n_points": int(len(sample_idx)),
            "n_boundary_slices_skipped": 0,
        }
        if not path or not Path(path).exists():
            rows.append({**base_row, "status": "NA: no existing result (UniST inr tier not run)"})
            continue
        meta, _ = resolve_expression(path, "scale")
        if meta["n_obs"] != ref_meta["n_obs"] or not np.array_equal(meta["var_names"], ref_meta["var_names"]):
            raise ValueError(f"Method result is not aligned with reference for {dataset}: {path}")
        if not np.allclose(meta["spatial"].astype(np.float64), ref_meta["spatial"].astype(np.float64),
                           atol=COORD_ATOL, rtol=0):
            raise ValueError(f"Spatial coordinates differ beyond float32 tolerance for {dataset}: {path}")
        method_obs = np.asarray([str(x) for x in read_obs_index(path)], dtype=object)
        if not np.array_equal(method_obs, ref_obs):
            raise ValueError(f"obs_names order differs from reference for {dataset}: {path}")

        zero_slices, prediction = zero_prediction_slices(path, sample_idx, slice_labels, keep)
        skipped = sorted(label for label, is_zero in zero_slices.items() if is_zero)

        corr3d, jac3d, _, gt_i, pred_i = metric_pair(reference_truth, prediction, graph_3d, names, top_k)
        detail = pd.DataFrame({"gene": names, "I_gt": gt_i, "I_pred": pred_i})
        detail.insert(0, "method", method)
        detail.insert(0, "dataset", dataset)
        gene_details.append(detail)

        slice_corrs, slice_jacs = [], []
        for label, local in slices.items():
            graph_2d = slice_graphs.get(label)
            if graph_2d is None:
                slice_rows.append({
                    "dataset": dataset, "method": method, "slice_id": label,
                    "n_points": int(len(local)), "Moran_I_corr": np.nan,
                    "SVG_Jaccard": np.nan,
                    "status": f"NA: fewer than {min_slice_points} sampled points",
                })
                continue
            if zero_slices.get(str(label), False):
                slice_rows.append({
                    "dataset": dataset, "method": method, "slice_id": label,
                    "n_points": int(len(local)), "Moran_I_corr": np.nan,
                    "SVG_Jaccard": np.nan,
                    "status": "NA: boundary slice not covered by UniST (zero-filled prediction)",
                })
                continue
            corr2d, jac2d, _, _, _ = metric_pair(reference_truth[local], prediction[local], graph_2d, names, top_k)
            slice_rows.append({
                "dataset": dataset, "method": method, "slice_id": label,
                "n_points": int(len(local)), "Moran_I_corr": corr2d,
                "SVG_Jaccard": jac2d, "status": "completed",
            })
            if np.isfinite(corr2d):
                slice_corrs.append(corr2d)
            if np.isfinite(jac2d):
                slice_jacs.append(jac2d)
        rows.append({
            **base_row,
            "Moran_I_corr_3D": corr3d, "SVG_Jaccard_3D": jac3d,
            "Moran_I_corr_2D_mean": float(np.mean(slice_corrs)) if slice_corrs else np.nan,
            "Moran_I_corr_2D_std": float(np.std(slice_corrs)) if slice_corrs else np.nan,
            "SVG_Jaccard_2D_mean": float(np.mean(slice_jacs)) if slice_jacs else np.nan,
            "SVG_Jaccard_2D_std": float(np.std(slice_jacs)) if slice_jacs else np.nan,
            "n_slices_evaluated": int(len(slice_corrs)),
            "n_boundary_slices_skipped": int(len(skipped)),
            "status": "completed" if not skipped else f"completed; boundary slices skipped (zero-filled): {','.join(skipped)}",
        })
    metadata = {
        "dataset": dataset, "reference": os.path.abspath(ref_path),
        "sample_size": int(len(sample_idx)), "graph_k": graph_k, "graph_edges_3d": int(graph_3d.nnz),
        "seed": seed, "n_total": int(ref_meta["n_obs"]), "n_genes_total": int(ref_meta["n_vars"]),
        "n_genes_evaluated": int(keep.sum()), "HK_genes_excluded": int((~keep).sum()),
        "n_slices": int(len(slices)), "n_slices_2d_evaluated": int(len(slice_graphs)),
        "min_slice_points": min_slice_points,
    }
    statuses = [
        {
            "Dataset": dataset, "Method": method,
            "Result_path": path or "",
            "Status": "completed" if path and Path(path).exists() else "NA",
            "Note": ("Existing prediction reused" if path and Path(path).exists()
                     else "UniST inr tier not run (cost budget); explicit NA, not imputed"),
        }
        for method, path in methods.items() if not method.startswith("NTF_recheck")
    ]
    return rows, slice_rows, gene_details, metadata, statuses


def read_obs_index(path):
    import h5py

    def decode(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    with h5py.File(path, "r") as f:
        return np.asarray([decode(x) for x in f["obs/_index"][:]], dtype=object)


METRIC_COLUMNS = [
    "Moran_I_corr_3D", "SVG_Jaccard_3D",
    "Moran_I_corr_2D_mean", "Moran_I_corr_2D_std",
    "SVG_Jaccard_2D_mean", "SVG_Jaccard_2D_std",
]


def paired_check(summary_df):
    """Recomputed NTF at 10w/100w must equal the allscales results exactly.

    Both CSVs are read as strings and parsed with Python's float() (correctly
    rounded); pandas' fast float parser can be 1 ulp off, which would break the
    tolerance-0 comparison even when the stored text is identical.
    """
    ref = pd.read_csv(ALLSCALES_DIR / "scale_benchmark_summary.csv", dtype=str)
    summary_s = summary_df.astype(str)
    checks = []
    ok_all = True
    for scale_k in (100, 1000):
        rec = summary_s[(summary_s["method"] == "NTF_recheck") & (summary_s["scale_k"] == str(scale_k))]
        old = ref[(ref["method"] == "NTF") & (ref["scale_k"] == str(scale_k))]
        for column in METRIC_COLUMNS + ["n_slices_evaluated", "n_points"]:
            new_v = float(rec[column].iloc[0])
            old_v = float(old[column].iloc[0])
            equal = (new_v == old_v) or (not np.isfinite(new_v) and not np.isfinite(old_v))
            ok_all &= equal
            checks.append({
                "scale_k": scale_k, "metric": column,
                "allscales_value": old_v, "recomputed_value": new_v,
                "exact_match": bool(equal),
            })
    return pd.DataFrame(checks), ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sample_size", type=int, default=100000)
    ap.add_argument("--graph_k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_slice_points", type=int, default=MIN_SLICE_POINTS)
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows, slice_rows, all_details, metadata, statuses = [], [], [], [], []
    for scale in SCALES:
        ntf_recheck = scale in ("10w", "100w")
        rows, srows, details, meta, stat = evaluate_scale(
            scale, args.sample_size, args.graph_k, args.seed, args.top_k, args.min_slice_points, ntf_recheck)
        summary_rows.extend(rows)
        slice_rows.extend(srows)
        all_details.extend(details)
        metadata.append(meta)
        statuses.extend(stat)
        print(f"[done] scale_{scale}", flush=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "unist_svg_scale_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(outdir / "unist_svg_slice_metrics.csv", index=False)
    if all_details:
        pd.concat(all_details, ignore_index=True).to_csv(outdir / "unist_svg_gene_moran_values.csv", index=False)
    pd.DataFrame(statuses).to_csv(outdir / "unist_svg_method_status.csv", index=False)
    with open(outdir / "unist_svg_benchmark_metadata.json", "w") as handle:
        json.dump({"sample_size": args.sample_size, "graph_k": args.graph_k, "seed": args.seed,
                   "top_k": args.top_k, "min_slice_points": args.min_slice_points,
                   "scales": SCALES, "inr_scales": sorted(INR_SCALES, key=scale_to_k),
                   "datasets": metadata}, handle, indent=2)
    check_df, ok = paired_check(summary_df)
    check_df.to_csv(outdir / "ntf_paired_check.csv", index=False)
    print("[paired-check] NTF 10w/100w recomputed vs allscales, exact match:", ok, flush=True)
    if not ok:
        raise SystemExit("Paired check failed: recomputed NTF metrics differ from allscales results")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
