#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified Moran's-I/SVG-overlap benchmark for NTF, FEAST and SpatialZ.

The script reads existing h5ad results through h5py, so the 1,000w-cell files
are never materialized as AnnData objects.  A single sampled coordinate graph is
shared by all methods for each dataset, making method comparisons paired.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


def decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def read_meta(path):
    with h5py.File(path, "r") as f:
        names = np.asarray([decode(x) for x in f["var/_index"][:]], dtype=object)
        meta = {
            "n_obs": int(f["X"].shape[0]),
            "n_vars": int(f["X"].shape[1]),
            "var_names": names,
            "spatial": np.asarray(f["obsm/spatial"][:], dtype=np.float32),
        }
        for key in ("slice_id", "domain"):
            p = f"obs/{key}"
            if f"{p}/codes" in f:
                categories = np.asarray([decode(x) for x in f[f"{p}/categories"][:]], dtype=object)
                codes = np.asarray(f[f"{p}/codes"][:])
                meta[key] = categories[codes]
            elif p in f:
                meta[key] = np.asarray(f[p][:])
        meta["layers"] = list(f["layers"].keys()) if "layers" in f else []
    return meta


def read_rows(path, indices, dataset):
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_idx = indices[order]
    with h5py.File(path, "r") as f:
        dset = f[dataset]
        blocks = []
        pos = 0
        while pos < len(sorted_idx):
            first = int(sorted_idx[pos])
            end = pos + 1
            while end < len(sorted_idx) and sorted_idx[end] < first + 100000:
                end += 1
            stop = min(first + 100000, dset.shape[0])
            block = np.asarray(dset[first:stop], dtype=np.float32)
            blocks.append(block[sorted_idx[pos:end] - first])
            pos = end
    sorted_rows = np.concatenate(blocks, axis=0) if blocks else np.empty((0, dset.shape[1]), dtype=np.float32)
    rows = np.empty_like(sorted_rows)
    rows[order] = sorted_rows
    return rows


def stratified_indices(labels, sample_size, seed=42, labels2=None):
    labels = np.asarray(labels).astype(str)
    if labels2 is not None:
        labels = np.char.add(np.char.add(labels, "||"), np.asarray(labels2).astype(str))
    n = min(int(sample_size), labels.size)
    if n == labels.size:
        return np.arange(labels.size, dtype=np.int64)
    groups = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    keys = list(groups)
    alloc = {key: max(1, int(round(n * len(groups[key]) / labels.size))) for key in keys}
    while sum(alloc.values()) > n:
        eligible = [key for key in keys if alloc[key] > 1]
        if not eligible:
            break
        key = max(eligible, key=lambda x: alloc[x])
        alloc[key] -= 1
    while sum(alloc.values()) < n:
        key = max(keys, key=lambda x: len(groups[x]) - alloc[x])
        if alloc[key] >= len(groups[key]):
            break
        alloc[key] += 1
    rng = np.random.default_rng(seed)
    picked = []
    for key in keys:
        group = np.asarray(groups[key], dtype=np.int64)
        picked.append(rng.choice(group, size=min(alloc[key], len(group)), replace=False))
    return np.sort(np.concatenate(picked))


def make_knn_graph(coords, k=15):
    n = len(coords)
    if n < 3:
        return sparse.csr_matrix((n, n), dtype=np.float32)
    kk = min(k + 1, n)
    _, neighbors = cKDTree(np.asarray(coords, dtype=np.float64)).query(coords, k=kk, workers=-1)
    neighbors = np.atleast_2d(neighbors)
    source = np.repeat(np.arange(n), kk - 1)
    target = neighbors[:, 1:].reshape(-1)
    keep = source != target
    source, target = source[keep], target[keep]
    rows = np.concatenate([source, target])
    cols = np.concatenate([target, source])
    graph = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))
    graph.data[:] = 1.0
    graph.eliminate_zeros()
    return graph


def moran_i(values, graph):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3 or graph.nnz == 0:
        return np.nan
    centered = values - np.nanmean(values)
    centered[~np.isfinite(centered)] = 0.0
    denominator = np.dot(centered, centered)
    if not np.isfinite(denominator) or denominator <= 1e-14:
        return np.nan
    return float((len(values) / graph.sum()) * np.dot(centered, graph.dot(centered)) / denominator)


def metric_pair(truth, prediction, graph, var_names, top_k=20):
    truth_i = np.asarray([moran_i(truth[:, j], graph) for j in range(truth.shape[1])])
    pred_i = np.asarray([moran_i(prediction[:, j], graph) for j in range(prediction.shape[1])])
    valid = np.isfinite(truth_i) & np.isfinite(pred_i)
    if valid.sum() > 2:
        corr = float(spearmanr(truth_i[valid], pred_i[valid]).statistic)
    else:
        corr = np.nan
    valid_names = np.asarray(var_names)[valid]
    valid_truth = truth_i[valid]
    valid_pred = pred_i[valid]
    truth_order = np.argsort(-valid_truth, kind="mergesort")
    pred_order = np.argsort(-valid_pred, kind="mergesort")
    truth_top = set(valid_names[truth_order[:top_k]])
    pred_top = set(valid_names[pred_order[:top_k]])
    union = truth_top | pred_top
    jaccard = len(truth_top & pred_top) / len(union) if union else np.nan
    return corr, float(jaccard), int(valid.sum()), truth_i, pred_i


def resolve_expression(path, dataset_kind):
    meta = read_meta(path)
    if "prediction" not in meta["layers"]:
        raise ValueError(f"Prediction layer is missing: {path}")
    truth_dataset = "layers/truth" if "truth" in meta["layers"] else "X"
    return meta, truth_dataset


def evaluate_dataset(dataset, methods, dataset_kind, sample_size, graph_k, seed, top_k):
    available = {name: path for name, path in methods.items() if path and Path(path).exists()}
    if not available:
        raise FileNotFoundError(f"No method result exists for {dataset}")
    reference_name = next(iter(available))
    ref_meta, ref_truth_dataset = resolve_expression(available[reference_name], dataset_kind)
    if dataset_kind == "domain":
        sample_idx = stratified_indices(ref_meta["slice_id"], sample_size, seed, ref_meta.get("domain"))
    else:
        sample_idx = stratified_indices(ref_meta["slice_id"], sample_size, seed)
    coords = ref_meta["spatial"][sample_idx]
    graph = make_knn_graph(coords, graph_k)
    keep = np.asarray([not str(name).endswith("_HK") for name in ref_meta["var_names"]])
    reference_truth = read_rows(available[reference_name], sample_idx, ref_truth_dataset)[:, keep]
    names = ref_meta["var_names"][keep]
    rows = []
    gene_details = []
    for method, path in methods.items():
        if not path or not Path(path).exists():
            rows.append({
                "dataset": dataset, "method": method, "Moran_I_corr": np.nan,
                "SVG_Jaccard": np.nan, "TopK": top_k, "HK_excluded": True,
                "n_genes": int(keep.sum()), "n_points": int(len(sample_idx)),
                "status": "NA: no existing result",
            })
            continue
        meta, truth_dataset = resolve_expression(path, dataset_kind)
        if meta["n_obs"] != ref_meta["n_obs"] or not np.array_equal(meta["var_names"], ref_meta["var_names"]):
            raise ValueError(f"Method result is not aligned with reference for {dataset}: {path}")
        if not np.array_equal(meta["spatial"], ref_meta["spatial"]):
            raise ValueError(f"Spatial coordinates are not aligned with reference for {dataset}: {path}")
        truth = reference_truth
        prediction = read_rows(path, sample_idx, "layers/prediction")[:, keep]
        corr, jaccard, n_valid, gt_i, pred_i = metric_pair(truth, prediction, graph, names, top_k)
        rows.append({
            "dataset": dataset, "method": method, "Moran_I_corr": corr,
            "SVG_Jaccard": jaccard, "TopK": top_k, "HK_excluded": True,
            "n_genes": int(keep.sum()), "n_points": int(len(sample_idx)),
            "status": "completed",
        })
        detail = pd.DataFrame({"gene": names, "I_gt": gt_i, "I_pred": pred_i})
        detail.insert(0, "method", method)
        detail.insert(0, "dataset", dataset)
        gene_details.append(detail)
    return rows, gene_details, {
        "dataset": dataset, "reference": os.path.abspath(available[reference_name]),
        "sample_size": int(len(sample_idx)), "graph_k": graph_k, "graph_edges": int(graph.nnz),
        "seed": seed, "n_total": int(ref_meta["n_obs"]), "n_genes_total": int(ref_meta["n_vars"]),
        "n_genes_evaluated": int(keep.sum()), "HK_genes_excluded": int((~keep).sum()),
    }


def write_workbook(summary, statuses, metadata, output):
    import xlsxwriter

    def clean(value):
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            return "NA"
        if isinstance(value, np.generic):
            return value.item()
        return value

    summary_cols = ["Dataset", "Method", "Moran_I_corr", "SVG_Jaccard"]
    summary_keys = ["dataset", "method", "Moran_I_corr", "SVG_Jaccard"]
    summary_df = pd.DataFrame([[clean(row.get(key)) for key in summary_keys] for row in summary], columns=summary_cols)
    status_cols = ["Dataset", "Method", "Result_path", "Status", "Note"]
    status_df = pd.DataFrame([[clean(row.get(key)) for key in status_cols] for row in statuses], columns=status_cols)
    metadata_cols = ["Dataset", "Reference_result", "n_total", "sample_size", "graph_k", "graph_edges", "seed", "n_genes_total", "n_genes_evaluated", "HK_genes_excluded"]
    metadata_keys = ["dataset", "reference", "n_total", "sample_size", "graph_k", "graph_edges", "seed", "n_genes_total", "n_genes_evaluated", "HK_genes_excluded"]
    metadata_df = pd.DataFrame([[clean(row.get(key)) for key in metadata_keys] for row in metadata], columns=metadata_cols)

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_format = workbook.add_format({"font_name": "Arial", "bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "align": "center", "valign": "vcenter"})
        body_format = workbook.add_format({"font_name": "Arial", "font_size": 10, "valign": "vcenter"})
        number_format = workbook.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.000", "valign": "vcenter"})

        for name, frame in (("Summary", summary_df), ("Method_Status", status_df), ("Metadata", metadata_df)):
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes(1, 0)
            sheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
            sheet.set_default_row(18)
            sheet.set_row(0, 22, header_format)
            for idx, column in enumerate(frame.columns):
                values = [str(column)] + [str(value) for value in frame.iloc[:, idx].tolist()]
                width = min(max(max(map(len, values), default=8) + 2, 12), 42)
                fmt = number_format if column in {"Moran_I_corr", "SVG_Jaccard"} else body_format
                sheet.set_column(idx, idx, width, fmt)
            sheet.hide_gridlines(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sample_size", type=int, default=100000)
    ap.add_argument("--graph_k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args()
    root = Path("/home/gongyuqiao/ur_annotation/NTF")
    feast = root / "mytrain/res/FEAST"
    ntf = root / "mytrain/res/simulation"
    spatialz = Path("/home/gongyuqiao/ur_annotation/SpatialZ/res/simulation")
    datasets = {
        "scale_10w": ("scale", {
            "NTF": str(ntf / "scale_test/10w/test_results.h5ad"),
            "FEAST": str(feast / "scale_test/10w/test_results.h5ad"),
            "SpatialZ": str(spatialz / "10w/test_results.h5ad"),
        }),
        "scale_100w": ("scale", {
            "NTF": str(ntf / "scale_test/100w/test_results.h5ad"),
            "FEAST": str(feast / "scale_test/100w/test_results.h5ad"),
            "SpatialZ": str(spatialz / "100w/test_results.h5ad"),
        }),
        "scale_1000w": ("scale", {
            "NTF": str(ntf / "scale_test/1000w/test_results.h5ad"),
            "FEAST": str(feast / "scale_test/1000w/test_results.h5ad"),
            "SpatialZ": None,
        }),
        "domain_k5": ("domain", {
            "NTF": str(ntf / "domain_interval/k5.h5ad"),
            "FEAST": str(feast / "interval/feast_k5.h5ad"),
            "SpatialZ": None,
        }),
    }
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows, all_details, metadata = [], [], []
    statuses = []
    for dataset, (kind, methods) in datasets.items():
        rows, details, meta = evaluate_dataset(dataset, methods, kind, args.sample_size, args.graph_k, args.seed, args.top_k)
        summary_rows.extend(rows)
        all_details.extend(details)
        metadata.append(meta)
        for method, path in methods.items():
            statuses.append({
                "Dataset": dataset, "Method": method, "Result_path": path or "",
                "Status": "completed" if path and Path(path).exists() else "NA",
                "Note": "Existing prediction reused" if path and Path(path).exists() else "No compatible existing result; not imputed",
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "benchmark_summary.csv", index=False)
    if all_details:
        pd.concat(all_details, ignore_index=True).to_csv(outdir / "gene_moran_values.csv", index=False)
    pd.DataFrame(statuses).to_csv(outdir / "method_status.csv", index=False)
    with open(outdir / "benchmark_metadata.json", "w") as handle:
        json.dump({"sample_size": args.sample_size, "graph_k": args.graph_k, "seed": args.seed, "top_k": args.top_k, "datasets": metadata}, handle, indent=2)
    write_workbook(summary_rows, statuses, metadata, outdir / "svg_benchmark_summary.xlsx")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
