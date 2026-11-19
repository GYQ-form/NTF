#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""All-scale SVG benchmark for NTF, FEAST and SpatialZ on simple simulation data.

Extends run_svg_benchmark.py to every simulated scale (10w-100w by 10w and
200w-1000w by 100w) and adds per-2D-slice Moran's I correlation and SVG
Jaccard next to the whole-volume 3D metrics.  Reuses the h5py chunked
readers, stratified sampler, kNN graph and Moran's I implementation from
run_svg_benchmark.py so the evaluation protocol is identical.
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
MIN_SLICE_POINTS = 30


def scale_to_k(scale):
    return int(scale[:-1]) * 10


def method_paths(scale):
    root = Path("/home/gongyuqiao/ur_annotation/NTF")
    spatialz = Path("/home/gongyuqiao/ur_annotation/SpatialZ/res/simulation")
    return {
        "NTF": str(root / f"mytrain/res/simulation/scale_test/{scale}/test_results.h5ad"),
        "FEAST": str(root / f"mytrain/res/FEAST/scale_test/{scale}/test_results.h5ad"),
        "SpatialZ": str(spatialz / f"{scale}/test_results.h5ad"),
    }


def evaluate_scale(scale, sample_size, graph_k, seed, top_k, min_slice_points):
    dataset = f"scale_{scale}"
    methods = method_paths(scale)
    available = {name: path for name, path in methods.items() if Path(path).exists()}
    if not available:
        raise FileNotFoundError(f"No method result exists for {dataset}")
    reference_name = next(iter(available))
    ref_meta, ref_truth_dataset = resolve_expression(available[reference_name], "scale")
    sample_idx = stratified_indices(ref_meta["slice_id"], sample_size, seed)
    coords = ref_meta["spatial"][sample_idx]
    slice_labels = np.asarray(ref_meta["slice_id"])[sample_idx].astype(str)
    graph_3d = make_knn_graph(coords, graph_k)
    keep = np.asarray([not str(name).endswith("_HK") for name in ref_meta["var_names"]])
    reference_truth = read_rows(available[reference_name], sample_idx, ref_truth_dataset)[:, keep]
    names = ref_meta["var_names"][keep]

    slices = {}
    for label in pd.unique(slice_labels):
        local = np.where(slice_labels == label)[0]
        slices[label] = local
    slice_graphs = {}
    for label, local in slices.items():
        if len(local) >= min_slice_points:
            slice_graphs[label] = make_knn_graph(coords[local][:, :2], graph_k)

    rows, slice_rows, gene_details = [], [], []
    for method, path in methods.items():
        if not Path(path).exists():
            rows.append({
                "dataset": dataset, "scale_k": scale_to_k(scale), "method": method,
                "Moran_I_corr_3D": np.nan, "SVG_Jaccard_3D": np.nan,
                "Moran_I_corr_2D_mean": np.nan, "Moran_I_corr_2D_std": np.nan,
                "SVG_Jaccard_2D_mean": np.nan, "SVG_Jaccard_2D_std": np.nan,
                "n_slices_evaluated": 0, "TopK": top_k,
                "n_genes": int(keep.sum()), "n_points": int(len(sample_idx)),
                "status": "NA: no existing result",
            })
            continue
        meta, _ = resolve_expression(path, "scale")
        if meta["n_obs"] != ref_meta["n_obs"] or not np.array_equal(meta["var_names"], ref_meta["var_names"]):
            raise ValueError(f"Method result is not aligned with reference for {dataset}: {path}")
        if not np.array_equal(meta["spatial"], ref_meta["spatial"]):
            raise ValueError(f"Spatial coordinates are not aligned with reference for {dataset}: {path}")
        prediction = read_rows(path, sample_idx, "layers/prediction")[:, keep]
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
            "dataset": dataset, "scale_k": scale_to_k(scale), "method": method,
            "Moran_I_corr_3D": corr3d, "SVG_Jaccard_3D": jac3d,
            "Moran_I_corr_2D_mean": float(np.mean(slice_corrs)) if slice_corrs else np.nan,
            "Moran_I_corr_2D_std": float(np.std(slice_corrs)) if slice_corrs else np.nan,
            "SVG_Jaccard_2D_mean": float(np.mean(slice_jacs)) if slice_jacs else np.nan,
            "SVG_Jaccard_2D_std": float(np.std(slice_jacs)) if slice_jacs else np.nan,
            "n_slices_evaluated": int(len(slice_corrs)), "TopK": top_k,
            "n_genes": int(keep.sum()), "n_points": int(len(sample_idx)),
            "status": "completed",
        })
    metadata = {
        "dataset": dataset, "reference": os.path.abspath(available[reference_name]),
        "sample_size": int(len(sample_idx)), "graph_k": graph_k, "graph_edges_3d": int(graph_3d.nnz),
        "seed": seed, "n_total": int(ref_meta["n_obs"]), "n_genes_total": int(ref_meta["n_vars"]),
        "n_genes_evaluated": int(keep.sum()), "HK_genes_excluded": int((~keep).sum()),
        "n_slices": int(len(slices)), "n_slices_2d_evaluated": int(len(slice_graphs)),
        "min_slice_points": min_slice_points,
    }
    statuses = [
        {
            "Dataset": dataset, "Method": method, "Result_path": path,
            "Status": "completed" if Path(path).exists() else "NA",
            "Note": "Existing prediction reused" if Path(path).exists() else "No compatible existing result; not imputed",
        }
        for method, path in methods.items()
    ]
    return rows, slice_rows, gene_details, metadata, statuses


METRIC_COLUMNS = [
    "Moran_I_corr_3D", "SVG_Jaccard_3D",
    "Moran_I_corr_2D_mean", "Moran_I_corr_2D_std",
    "SVG_Jaccard_2D_mean", "SVG_Jaccard_2D_std",
]


def write_workbook(summary_rows, statuses, metadata, output):
    import xlsxwriter  # noqa: F401

    def clean(value):
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            return "NA"
        if isinstance(value, np.generic):
            return value.item()
        return value

    summary_df = pd.DataFrame(summary_rows)
    frames = {}
    for method in ("NTF", "FEAST", "SpatialZ"):
        sub = summary_df[summary_df["method"] == method].sort_values("scale_k")
        frame = pd.DataFrame({"scale (*10^3)": sub["scale_k"].tolist()})
        for column in METRIC_COLUMNS:
            frame[column] = [clean(v) for v in sub[column].tolist()]
        frame["n_slices_evaluated"] = [clean(v) for v in sub["n_slices_evaluated"].tolist()]
        frame["n_points"] = [clean(v) for v in sub["n_points"].tolist()]
        frames[method] = frame
    status_df = pd.DataFrame(statuses)[["Dataset", "Method", "Result_path", "Status", "Note"]]
    metadata_df = pd.DataFrame(metadata)
    summary_cols = ["dataset", "scale_k", "method"] + METRIC_COLUMNS + ["n_slices_evaluated", "TopK", "n_genes", "n_points", "status"]
    summary_long = summary_df[summary_cols].copy()
    for column in METRIC_COLUMNS:
        summary_long[column] = [clean(v) for v in summary_long[column].tolist()]

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_format = workbook.add_format({"font_name": "Arial", "bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "align": "center", "valign": "vcenter"})
        body_format = workbook.add_format({"font_name": "Arial", "font_size": 10, "valign": "vcenter"})
        number_format = workbook.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.000", "valign": "vcenter"})

        sheets = list(frames.items()) + [("Summary", summary_long), ("Method_Status", status_df), ("Metadata", metadata_df)]
        for name, frame in sheets:
            frame = frame.copy()
            for column in frame.columns:
                frame[column] = [clean(v) for v in frame[column].tolist()]
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes(1, 0)
            sheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
            sheet.set_default_row(18)
            sheet.set_row(0, 22, header_format)
            for idx, column in enumerate(frame.columns):
                values = [str(column)] + [str(value) for value in frame.iloc[:, idx].tolist()]
                width = min(max(max(map(len, values), default=8) + 2, 12), 42)
                fmt = number_format if column in METRIC_COLUMNS else body_format
                sheet.set_column(idx, idx, width, fmt)
            sheet.hide_gridlines(2)


def write_plot(summary_rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_df = pd.DataFrame(summary_rows)
    colors = {"NTF": "#C0392B", "FEAST": "#2471A3", "SpatialZ": "#1E8449"}
    panels = [
        ("Moran_I_corr_3D", "Moran's I corr (3D)", False),
        ("SVG_Jaccard_3D", "SVG Jaccard Top-20 (3D)", False),
        ("Moran_I_corr_2D_mean", "Moran's I corr (2D slices)", "Moran_I_corr_2D_std"),
        ("SVG_Jaccard_2D_mean", "SVG Jaccard Top-20 (2D slices)", "SVG_Jaccard_2D_std"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (column, title, std_column) in zip(axes.flat, panels):
        for method in ("NTF", "FEAST", "SpatialZ"):
            sub = summary_df[summary_df["method"] == method].sort_values("scale_k")
            sub = sub[np.isfinite(sub[column].astype(float))]
            if sub.empty:
                continue
            x = sub["scale_k"].to_numpy(dtype=float)
            y = sub[column].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.5, color=colors[method], label=method)
            if std_column:
                std = sub[std_column].to_numpy(dtype=float)
                ax.fill_between(x, y - std, y + std, color=colors[method], alpha=0.12, linewidth=0)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("scale (*10^3 cells)")
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("SVG benchmark across simulated scales (simple simulation)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=200)
    plt.close(fig)


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
        rows, srows, details, meta, stat = evaluate_scale(scale, args.sample_size, args.graph_k, args.seed, args.top_k, args.min_slice_points)
        summary_rows.extend(rows)
        slice_rows.extend(srows)
        all_details.extend(details)
        metadata.append(meta)
        statuses.extend(stat)
        print(f"[done] scale_{scale}", flush=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "scale_benchmark_summary.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(outdir / "slice_metrics.csv", index=False)
    if all_details:
        pd.concat(all_details, ignore_index=True).to_csv(outdir / "gene_moran_values.csv", index=False)
    pd.DataFrame(statuses).to_csv(outdir / "method_status.csv", index=False)
    with open(outdir / "benchmark_metadata.json", "w") as handle:
        json.dump({"sample_size": args.sample_size, "graph_k": args.graph_k, "seed": args.seed,
                   "top_k": args.top_k, "min_slice_points": args.min_slice_points,
                   "scales": SCALES, "datasets": metadata}, handle, indent=2)
    write_workbook(summary_rows, statuses, metadata, outdir / "svg_benchmark_allscales_summary.xlsx")
    write_plot(summary_rows, outdir / "benchmark_metrics_allscales.png")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
