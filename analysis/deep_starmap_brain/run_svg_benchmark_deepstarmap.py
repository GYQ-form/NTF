#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Moran's-I / SVG-Jaccard benchmark for Deep-STARmap mouse brain (NTF vs FEAST vs SpatialZ vs UniST).

Adapted from mytrain/scripts/run_svg_benchmark.py (which was simulation-only and assumed
dense X readable via h5py slicing). Differences:
  - datasets are the 5 Deep-STARmap scenarios; senario3/head/tail are NTF-only
    (SpatialZ/FEAST/UniST cannot extrapolate) and marked NA; UniST only has
    senario1/senario2 results;
  - sparse-aware reads: X in the processed data is CSR (h5ad group), so meta/row reads
    handle both dense datasets and CSR groups; prediction layers are dense;
  - ground truth is the observed X of test cells (real data, no layers['truth']);
  - multiple Top-K values (5/10/20/50) in one pass, long-format output.

A single sampled coordinate graph is shared by all methods within each scenario,
making method comparisons paired (same sampling, same kNN graph).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

NTF_ROOT = Path("/home/gongyuqiao/ur_annotation/NTF")
SPATIALZ_RES = Path("/home/gongyuqiao/ur_annotation/SpatialZ/res")

def build_datasets(res_suffix="", dataset="deep_starmap_brain"):
    return {
        "senario1": {
            "NTF": str(NTF_ROOT / f"mytrain/res/{dataset}{res_suffix}/senario1/test_results.h5ad"),
            "FEAST": str(NTF_ROOT / f"mytrain/res/FEAST/{dataset}{res_suffix}/senario1/test_results.h5ad"),
            "SpatialZ": str(SPATIALZ_RES / f"{dataset}{res_suffix}/senario1/test_results.h5ad"),
            "UniST": str(NTF_ROOT / f"mytrain/res/UniST/{dataset}{res_suffix}/senario1/test_results.h5ad"),
        },
        "senario2": {
            "NTF": str(NTF_ROOT / f"mytrain/res/{dataset}{res_suffix}/senario2/test_results.h5ad"),
            "FEAST": str(NTF_ROOT / f"mytrain/res/FEAST/{dataset}{res_suffix}/senario2/test_results.h5ad"),
            "SpatialZ": str(SPATIALZ_RES / f"{dataset}{res_suffix}/senario2/test_results.h5ad"),
            "UniST": str(NTF_ROOT / f"mytrain/res/UniST/{dataset}{res_suffix}/senario2/test_results.h5ad"),
        },
        "senario3": {
            "NTF": str(NTF_ROOT / f"mytrain/res/{dataset}{res_suffix}/senario3/test_results.h5ad"),
            "FEAST": None,
            "SpatialZ": None,
            "UniST": None,
        },
        "head": {
            "NTF": str(NTF_ROOT / f"mytrain/res/{dataset}{res_suffix}/head/test_results.h5ad"),
            "FEAST": None,
            "SpatialZ": None,
            "UniST": None,
        },
        "tail": {
            "NTF": str(NTF_ROOT / f"mytrain/res/{dataset}{res_suffix}/tail/test_results.h5ad"),
            "FEAST": None,
            "SpatialZ": None,
            "UniST": None,
        },
    }


def decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def read_meta(path):
    with h5py.File(path, "r") as f:
        idx_key = f["var"].attrs.get("_index", "_index")
        if isinstance(idx_key, bytes):
            idx_key = idx_key.decode()
        names = np.asarray([decode(x) for x in f[f"var/{idx_key}"][:]], dtype=object)
        x_node = f["X"]
        if isinstance(x_node, h5py.Group):  # CSR/CSC sparse
            n_obs, n_vars = (int(v) for v in x_node.attrs["shape"])
        else:
            n_obs, n_vars = int(x_node.shape[0]), int(x_node.shape[1])
        meta = {
            "n_obs": n_obs,
            "n_vars": n_vars,
            "var_names": names,
            "spatial": np.asarray(f["obsm/spatial"][:], dtype=np.float32),
        }
        for key in ("slice_id", "slice_id_v2", "domain"):
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
    """Read selected rows of a dense dataset or a CSR group as a dense float32 array."""
    indices = np.asarray(indices, dtype=np.int64)
    with h5py.File(path, "r") as f:
        node = f[dataset]
        if isinstance(node, h5py.Group):  # sparse CSR
            indptr = np.asarray(node["indptr"][:])
            n_vars = int(node.attrs["shape"][1])
            out = np.zeros((len(indices), n_vars), dtype=np.float32)
            # load only the needed span of data/indices
            lo = int(indptr[indices.min()])
            hi = int(indptr[indices.max() + 1]) if indices.max() + 1 < len(indptr) else int(indptr[-1])
            data = np.asarray(node["data"][lo:hi], dtype=np.float32)
            colidx = np.asarray(node["indices"][lo:hi])
            for row_out, r in enumerate(indices):
                s, e = int(indptr[r]), int(indptr[r + 1])
                if e > s:
                    out[row_out, colidx[s - lo:e - lo]] = data[s - lo:e - lo]
            return out
        # dense dataset: block-wise row gather
        order = np.argsort(indices)
        sorted_idx = indices[order]
        blocks = []
        pos = 0
        while pos < len(sorted_idx):
            first = int(sorted_idx[pos])
            end = pos + 1
            while end < len(sorted_idx) and sorted_idx[end] < first + 100000:
                end += 1
            stop = min(first + 100000, node.shape[0])
            block = np.asarray(node[first:stop], dtype=np.float32)
            blocks.append(block[sorted_idx[pos:end] - first])
            pos = end
        sorted_rows = np.concatenate(blocks, axis=0) if blocks else np.empty((0, node.shape[1]), dtype=np.float32)
        rows = np.empty_like(sorted_rows)
        rows[order] = sorted_rows
        return rows


def stratified_indices(labels, sample_size, seed=42):
    labels = np.asarray(labels).astype(str)
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


def metric_pair_multi_k(truth, prediction, graph, var_names, top_ks):
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
    jaccards = {}
    for k in top_ks:
        kk = min(k, len(valid_names))
        truth_top = set(valid_names[truth_order[:kk]])
        pred_top = set(valid_names[pred_order[:kk]])
        union = truth_top | pred_top
        jaccards[k] = len(truth_top & pred_top) / len(union) if union else np.nan
    return corr, jaccards, int(valid.sum()), truth_i, pred_i


def evaluate_dataset(dataset, methods, sample_size, graph_k, seed, top_ks, strata_col="slice_id"):
    available = {name: path for name, path in methods.items() if path and Path(path).exists()}
    if not available:
        raise FileNotFoundError(f"No method result exists for {dataset}")
    reference_name = next(iter(available))
    ref_path = available[reference_name]
    ref_meta = read_meta(ref_path)
    if "prediction" not in ref_meta["layers"]:
        raise ValueError(f"Prediction layer is missing: {ref_path}")
    # real data: ground truth is the observed X of test cells
    truth_dataset = "layers/truth" if "truth" in ref_meta["layers"] else "X"
    sample_idx = stratified_indices(ref_meta.get(strata_col, ref_meta["slice_id"]), sample_size, seed)
    coords = ref_meta["spatial"][sample_idx]
    graph = make_knn_graph(coords, graph_k)
    keep = np.asarray([not str(name).endswith("_HK") for name in ref_meta["var_names"]])
    reference_truth = read_rows(ref_path, sample_idx, truth_dataset)[:, keep]
    names = ref_meta["var_names"][keep]

    rows = []
    gene_details = []
    for method, path in methods.items():
        if not path or not Path(path).exists():
            rows.append({
                "dataset": dataset, "method": method, "Moran_I_corr": np.nan,
                **{f"SVG_Jaccard@{k}": np.nan for k in top_ks},
                "n_genes": int(keep.sum()), "n_points": int(len(sample_idx)),
                "status": "NA: no result (method cannot run this scenario)",
            })
            continue
        meta = read_meta(path)
        if "prediction" not in meta["layers"]:
            raise ValueError(f"Prediction layer is missing: {path}")
        if meta["n_obs"] != ref_meta["n_obs"] or not np.array_equal(meta["var_names"], ref_meta["var_names"]):
            raise ValueError(f"Method result is not aligned with reference for {dataset}: {path}")
        if not np.array_equal(meta["spatial"], ref_meta["spatial"]):
            raise ValueError(f"Spatial coordinates are not aligned with reference for {dataset}: {path}")
        prediction = read_rows(path, sample_idx, "layers/prediction")[:, keep]
        corr, jaccards, n_valid, gt_i, pred_i = metric_pair_multi_k(
            reference_truth, prediction, graph, names, top_ks)
        rows.append({
            "dataset": dataset, "method": method, "Moran_I_corr": corr,
            **{f"SVG_Jaccard@{k}": float(v) for k, v in jaccards.items()},
            "n_genes": int(keep.sum()), "n_points": int(len(sample_idx)),
            "n_valid_genes": n_valid, "status": "completed",
        })
        detail = pd.DataFrame({"gene": names, "I_gt": gt_i, "I_pred": pred_i})
        detail.insert(0, "method", method)
        detail.insert(0, "dataset", dataset)
        gene_details.append(detail)
        print(f"  {dataset}/{method}: Moran_I_corr={corr:.4f}, " +
              ", ".join(f"J@{k}={v:.4f}" for k, v in jaccards.items()), flush=True)
    return rows, gene_details, {
        "dataset": dataset, "reference": str(Path(ref_path).resolve()),
        "truth_source": truth_dataset,
        "sample_size": int(len(sample_idx)), "graph_k": graph_k, "graph_edges": int(graph.nnz),
        "seed": seed, "n_total": int(ref_meta["n_obs"]), "n_genes_total": int(ref_meta["n_vars"]),
        "n_genes_evaluated": int(keep.sum()), "HK_genes_excluded": int((~keep).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sample_size", type=int, default=100000)
    ap.add_argument("--graph_k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top_ks", type=int, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--res_suffix", default="", help="Result-dir suffix, e.g. '_v2' for the v2 benchmark")
    ap.add_argument("--dataset", default="deep_starmap_brain", help="Dataset result-dir base name (e.g. wemerfish_C6s_E2)")
    ap.add_argument("--strata_col", default="slice_id", help="obs column for stratified sampling")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows, all_details, metadata = [], [], []
    datasets = build_datasets(args.res_suffix, dataset=args.dataset)
    for dataset, methods in datasets.items():
        print(f"[{dataset}]", flush=True)
        rows, details, meta = evaluate_dataset(
            dataset, methods, args.sample_size, args.graph_k, args.seed, args.top_ks,
            strata_col=args.strata_col)
        summary_rows.extend(rows)
        all_details.extend(details)
        metadata.append(meta)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "svg_benchmark_summary.csv", index=False)
    # long format for plotting
    long_rows = []
    for row in summary_rows:
        long_rows.append({"dataset": row["dataset"], "method": row["method"],
                          "metric": "Moran_I_corr", "value": row["Moran_I_corr"], "status": row["status"]})
        for k in args.top_ks:
            long_rows.append({"dataset": row["dataset"], "method": row["method"],
                              "metric": f"SVG_Jaccard@{k}", "value": row[f"SVG_Jaccard@{k}"], "status": row["status"]})
    pd.DataFrame(long_rows).to_csv(outdir / "svg_benchmark_long.csv", index=False)
    if all_details:
        pd.concat(all_details, ignore_index=True).to_csv(outdir / "gene_moran_values.csv", index=False)
    with open(outdir / "svg_benchmark_metadata.json", "w") as handle:
        json.dump({"sample_size": args.sample_size, "graph_k": args.graph_k, "seed": args.seed,
                   "top_ks": args.top_ks, "res_suffix": args.res_suffix, "strata_col": args.strata_col,
                   "datasets": metadata}, handle, indent=2)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
