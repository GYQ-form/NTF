#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate SVG/spatial-pattern conservation on existing simulation results.

The implementation deliberately uses h5py-backed row reads instead of loading
large h5ad files through AnnData.  It writes only compact CSV/JSON/PNG outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr


def _decode(x):
    if isinstance(x, bytes):
        return x.decode()
    return str(x)


def read_h5_metadata(path):
    with h5py.File(path, "r") as f:
        names = np.array([_decode(x) for x in f["var/_index"][:]], dtype=object)
        obs_names = np.array([_decode(x) for x in f["obs/_index"][:]], dtype=object)
        meta = {"n_obs": int(f["X"].shape[0]), "n_vars": int(f["X"].shape[1]),
                "var_names": names, "obs_names": obs_names}
        for col in ("slice_id", "domain", "cluster_id", "group"):
            p = f"obs/{col}"
            if f"{p}/codes" in f and f"{p}/categories" in f:
                cats = np.array([_decode(x) for x in f[f"{p}/categories"][:]], dtype=object)
                codes = np.asarray(f[f"{p}/codes"][:])
                meta[col] = cats[codes]
            elif p in f:
                meta[col] = np.asarray(f[p][:])
        meta["spatial"] = np.asarray(f["obsm/spatial"][:], dtype=np.float32)
        meta["layers"] = list(f.get("layers", {}).keys()) if "layers" in f else []
    return meta


def read_rows(path, indices, dataset="X"):
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_idx = indices[order]
    with h5py.File(path, "r") as f:
        d = f[dataset]
        # HDF5 fancy indexing of many random rows can decompress the same
        # gzip chunks repeatedly.  Read moderate contiguous windows once,
        # then select rows in memory.
        pieces = []
        pos = 0
        while pos < len(sorted_idx):
            first = int(sorted_idx[pos]); endpos = pos + 1
            while endpos < len(sorted_idx) and sorted_idx[endpos] < first + 100000:
                endpos += 1
            block_stop = min(first + 100000, d.shape[0])
            block = np.asarray(d[first:block_stop], dtype=np.float32)
            pieces.append(block[sorted_idx[pos:endpos] - first])
            pos = endpos
        arr_sorted = np.concatenate(pieces, axis=0) if pieces else np.empty((0, d.shape[1]), dtype=np.float32)
    arr = np.empty_like(arr_sorted)
    arr[order] = arr_sorted
    out = arr
    return out


def stratified_indices(labels, n, seed=42, labels2=None):
    labels = np.asarray(labels).astype(str)
    if labels2 is not None:
        labels = np.char.add(np.char.add(labels, "||"), np.asarray(labels2).astype(str))
    n = min(int(n), labels.size)
    rng = np.random.default_rng(seed)
    groups = {}
    for i, g in enumerate(labels):
        groups.setdefault(g, []).append(i)
    # proportional allocation, with at least one per non-empty stratum
    keys = list(groups)
    alloc = {k: max(1, int(round(n * len(groups[k]) / labels.size))) for k in keys}
    while sum(alloc.values()) > n:
        k = max((k for k in keys if alloc[k] > 1), key=lambda x: alloc[x], default=None)
        if k is None: break
        alloc[k] -= 1
    while sum(alloc.values()) < n:
        k = max(keys, key=lambda x: len(groups[x]) - alloc[x])
        if alloc[k] < len(groups[k]): alloc[k] += 1
        else: break
    out = []
    for k in keys:
        g = np.asarray(groups[k], dtype=np.int64)
        take = min(alloc[k], len(g))
        out.append(rng.choice(g, size=take, replace=False))
    return np.sort(np.concatenate(out))


def knn_graph(coords, k=15):
    n = coords.shape[0]
    if n < 3:
        return sparse.csr_matrix((n, n), dtype=np.float32)
    kk = min(k + 1, n)
    tree = cKDTree(np.asarray(coords, dtype=np.float64))
    _, idx = tree.query(coords, k=kk, workers=-1)
    idx = np.atleast_2d(idx)
    src = np.repeat(np.arange(n), kk - 1)
    dst = idx[:, 1:].reshape(-1)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    # Symmetric binary union.  A directed graph is also valid for Moran's I,
    # but the symmetric form is easier to audit and compare across matrices.
    rows = np.concatenate([src, dst]); cols = np.concatenate([dst, src])
    w = sparse.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    w.data[:] = 1.0
    w.eliminate_zeros()
    return w


def moran_i(values, w):
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[0]
    if n < 3 or w.nnz == 0:
        return np.nan
    z = x - np.nanmean(x)
    den = np.nansum(z * z)
    if not np.isfinite(den) or den <= 1e-14:
        return np.nan
    # Inputs are finite in the simulation files; replacing any accidental NaN
    # with zero after centering keeps the sparse product well-defined.
    z[~np.isfinite(z)] = 0.0
    return float((n / w.sum()) * np.dot(z, w.dot(z)) / den)


def graph_metrics(gt, pred, w, observed=None, names=None, labels=None, mode="sample_knn"):
    n_genes = gt.shape[1]
    if names is None: names = [f"gene_{i}" for i in range(n_genes)]
    I_gt = np.array([moran_i(gt[:, j], w) for j in range(n_genes)])
    I_pred = np.array([moran_i(pred[:, j], w) for j in range(n_genes)])
    I_obs = np.array([moran_i(observed[:, j], w) for j in range(n_genes)]) if observed is not None else np.full(n_genes, np.nan)
    valid = np.isfinite(I_gt) & np.isfinite(I_pred)
    rg = np.full(n_genes, np.nan); rp = np.full(n_genes, np.nan)
    if valid.sum() > 2:
        rg[valid] = pd.Series(I_gt[valid]).rank(method="average", ascending=False).to_numpy()
        rp[valid] = pd.Series(I_pred[valid]).rank(method="average", ascending=False).to_numpy()
    rows = []
    for j, name in enumerate(names):
        rows.append({"gene": name, "I_gt": I_gt[j], "I_obs": I_obs[j], "I_pred": I_pred[j],
                     "delta_I_pred_gt": I_pred[j] - I_gt[j],
                     "abs_delta_I": abs(I_pred[j] - I_gt[j]),
                     "relative_abs_delta_I": abs(I_pred[j] - I_gt[j]) / max(abs(I_gt[j]), 1e-8),
                     "rank_gt": rg[j], "rank_pred": rp[j], "rank_shift": rp[j] - rg[j],
                     "label": labels[j] if labels is not None else "NA", "graph_mode": mode})
    return pd.DataFrame(rows)


def topk_metrics(df, topks=(5, 10, 20, 50), truth_positive=None):
    d = df.dropna(subset=["I_gt", "I_pred"]).copy()
    d = d.sort_values("I_gt", ascending=False)
    gt_order = d.gene.tolist()
    pred_order = d.sort_values("I_pred", ascending=False).gene.tolist()
    out = []
    for k in topks:
        a, b = set(gt_order[:k]), set(pred_order[:k])
        inter = len(a & b)
        row = {"topk": k, "intersection": inter, "jaccard": inter / max(len(a | b), 1),
               "precision_vs_gt": inter / max(k, 1), "recall_vs_gt": inter / max(k, 1)}
        if truth_positive is not None:
            t = set(truth_positive)
            p = set(pred_order[:k]); g = set(gt_order[:k])
            row.update({"label_precision_pred": len(p & t) / max(len(p), 1),
                        "label_recall_pred": len(p & t) / max(len(t), 1),
                        "label_precision_gt": len(g & t) / max(len(g), 1)})
        out.append(row)
    rho = spearmanr(d.I_gt, d.I_pred).statistic if len(d) > 2 else np.nan
    try: pr = pearsonr(d.I_gt, d.I_pred).statistic if len(d) > 2 else np.nan
    except Exception: pr = np.nan
    return pd.DataFrame(out), {"spearman_I": float(rho), "pearson_I": float(pr),
                               "median_abs_delta_I": float(d.abs_delta_I.median()),
                               "mean_abs_delta_I": float(d.abs_delta_I.mean())}


def binary_label_metrics(scores, labels):
    """AUROC/AUPRC without requiring sklearn (handles one-class labels)."""
    y = np.asarray(labels, dtype=int); s = np.asarray(scores, dtype=float)
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    if y.size == 0 or y.min() == y.max():
        return {"n": int(y.size), "n_positive": int(y.sum()), "auroc": np.nan, "auprc": np.nan}
    order = np.argsort(-s, kind="mergesort"); yy = y[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    rec = tp / max(tp[-1], 1); prec = tp / np.maximum(tp + fp, 1)
    auprc = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]) + rec[0] * prec[0])
    # Mann-Whitney AUROC with ties averaged via rankdata.
    from scipy.stats import rankdata
    ranks = rankdata(s)
    pos = y == 1; neg = ~pos
    auroc = float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))
    return {"n": int(y.size), "n_positive": int(y.sum()), "auroc": auroc, "auprc": auprc}


def grid_aggregate(path, x_dataset, target_bins=100000, chunk=100000):
    """Aggregate a h5ad expression dataset to a fixed 3-D grid."""
    with h5py.File(path, "r") as f:
        coords = np.asarray(f["obsm/spatial"][:], dtype=np.float32)
        n, g = f[x_dataset].shape
        lo = coords.min(axis=0); hi = coords.max(axis=0)
        nside = max(2, int(round(float(target_bins) ** (1 / 3))))
        span = np.maximum(hi - lo, 1e-8)
        ids = np.floor((coords - lo) / span * nside).astype(np.int64)
        ids = np.clip(ids, 0, nside - 1)
        flat = ids[:, 0] * nside * nside + ids[:, 1] * nside + ids[:, 2]
        uniq, inv = np.unique(flat, return_inverse=True)
        sums = np.zeros((len(uniq), g), dtype=np.float64); counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            mat = np.asarray(f[x_dataset][start:stop], dtype=np.float32)
            ii = inv[start:stop]
            # Row-wise accumulation avoids one Python-level bincount call per
            # gene and is substantially faster for million-cell matrices.
            np.add.at(sums, ii, mat)
    means = (sums / np.maximum(counts[:, None], 1)).astype(np.float32)
    xyz = np.column_stack((uniq // (nside*nside), (uniq // nside) % nside, uniq % nside)).astype(np.int32)
    return xyz, means, counts


def grid_graph(xyz, connectivity=6):
    lookup = {tuple(x): i for i, x in enumerate(xyz)}
    offs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    if connectivity == 26:
        offs = [(a,b,c) for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1) if (a,b,c)!=(0,0,0)]
    r=[]; c=[]
    for i, x in enumerate(xyz):
        for o in offs:
            j = lookup.get((int(x[0]+o[0]), int(x[1]+o[1]), int(x[2]+o[2])))
            if j is not None: r.append(i); c.append(j)
    w = sparse.csr_matrix((np.ones(len(r), dtype=np.float32), (r,c)), shape=(len(xyz),len(xyz)))
    w.data[:] = 1
    return w


def save_scatter(df, out_png, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2,4.4))
        for lab, sub in df.groupby("label", dropna=False):
            ax.scatter(sub.I_gt, sub.I_pred, s=18, alpha=.8, label=str(lab))
        lim = np.nanmax(np.abs(np.r_[df.I_gt.values, df.I_pred.values]))
        if np.isfinite(lim): ax.plot([-lim,lim],[-lim,lim],"k--",lw=.8)
        ax.set(xlabel="Ground-truth Moran's I", ylabel="Prediction Moran's I", title=title)
        if df.label.nunique() <= 8: ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)
    except Exception as e:
        (Path(out_png).with_suffix(".plot_error.txt")).write_text(str(e))


def evaluate_one(path, outdir, dataset_kind, sample_size, k, seed, topks, grid_target=None, scenario="full", train_rule_k=None):
    os.makedirs(outdir, exist_ok=True); figdir = os.path.join(outdir, "figures"); os.makedirs(figdir, exist_ok=True)
    meta = read_h5_metadata(path)
    n = meta["n_obs"]
    labels = None
    if dataset_kind == "scale":
        labels = np.array([str(x) for x in meta.get("pattern_name", ["NA"]*meta["n_vars"])]) if "pattern_name" in meta else None
        # var pattern_name is categorical in h5ad; read_metadata currently only obs.
        with h5py.File(path, "r") as f:
            if "var/pattern_name/codes" in f:
                cats=np.array([_decode(x) for x in f["var/pattern_name/categories"][:]],dtype=object); codes=np.asarray(f["var/pattern_name/codes"][:]); labels=cats[codes]
        idx = np.arange(n) if sample_size >= n else stratified_indices(meta.get("slice_id", np.zeros(n)), sample_size, seed)
        gt = read_rows(path, idx, "layers/truth"); pred = read_rows(path, idx, "layers/prediction"); obs = read_rows(path, idx, "X")
    else:
        idx = stratified_indices(meta.get("slice_id"), sample_size, seed, meta.get("domain")) if sample_size < n else np.arange(n)
        gt = read_rows(path, idx, "X"); pred = read_rows(path, idx, "layers/prediction"); obs = None
        labels = np.array(["Diff" if str(x).endswith("_Diff") else "HK" if str(x).endswith("_HK") else "other" for x in meta["var_names"]])
    coords = meta["spatial"][idx]
    if scenario == "heldout" and train_rule_k:
        slice_ids = np.asarray(meta.get("slice_id"))[idx].astype(str)
        train_slices = set(str(x) for x in np.arange(100)[::train_rule_k])
        keep = np.array([s not in train_slices for s in slice_ids])
        coords, gt, pred = coords[keep], gt[keep], pred[keep]
        if obs is not None: obs = obs[keep]
    elif scenario == "hybrid" and train_rule_k:
        slice_ids = np.asarray(meta.get("slice_id"))[idx].astype(str)
        train_slices = set(str(x) for x in np.arange(100)[::train_rule_k])
        train = np.array([s in train_slices for s in slice_ids])
        pred = pred.copy(); pred[train] = gt[train]
    w = knn_graph(coords, k=k)
    label_arr = labels
    df = graph_metrics(gt, pred, w, obs, meta["var_names"], label_arr, "sample_knn")
    df.to_csv(os.path.join(outdir, f"gene_spatial_metrics_{scenario}.csv"), index=False)
    positive = [str(meta["var_names"][i]) for i,x in enumerate(labels) if x not in ("HK","other","NA")] if dataset_kind=="domain" else None
    top, summ = topk_metrics(df, topks, positive)
    top.insert(0, "scenario", scenario); top.to_csv(os.path.join(outdir, f"svg_overlap_{scenario}.csv"), index=False)
    if dataset_kind == "domain":
        lm = binary_label_metrics(df.I_pred.values, labels == "Diff"); lm_gt = binary_label_metrics(df.I_gt.values, labels == "Diff")
        pd.DataFrame([{**lm, "score":"prediction"}, {**lm_gt, "score":"ground_truth"}]).to_csv(os.path.join(outdir, f"truth_label_metrics_{scenario}.csv"), index=False)
    save_scatter(df, os.path.join(figdir, f"moran_gt_vs_pred_{scenario}.png"), f"{dataset_kind} {scenario} (n={len(coords):,})")
    result = {"scenario": scenario, "n_total": n, "n_used": int(len(coords)), "graph":"sample_knn", "k":k, **summ}
    if grid_target is not None and scenario == "full":
        # Grid computation is on the complete file and therefore independent of
        # the kNN sample.  For domain hybrid/heldout, use the sample graph only.
        xyz, ggt, counts = grid_aggregate(path, "layers/truth" if dataset_kind=="scale" else "X", grid_target)
        # prediction is not X for result files; aggregate separately and use same bins.
        with h5py.File(path,"r") as f:
            pred_full = f["layers/prediction"]
            nside = max(2, int(round(float(grid_target)**(1/3))))
            cfull = np.asarray(f["obsm/spatial"][:],dtype=np.float32); lo=cfull.min(0); hi=cfull.max(0); span=np.maximum(hi-lo,1e-8)
            ids=np.floor((cfull-lo)/span*nside).astype(np.int64); ids=np.clip(ids,0,nside-1); flat=ids[:,0]*nside*nside+ids[:,1]*nside+ids[:,2]
            uniq=np.unique(flat); inv=np.searchsorted(uniq,flat); sums=np.zeros((len(uniq), pred_full.shape[1]),dtype=np.float64)
            for st in range(0,n,100000):
                en=min(n,st+100000); mat=np.asarray(pred_full[st:en],dtype=np.float32); ii=inv[st:en]
                np.add.at(sums, ii, mat)
        gpred=(sums/np.maximum(counts[:,None],1)).astype(np.float32); gw=grid_graph(xyz,6)
        gd=graph_metrics(ggt,gpred,gw,None,meta["var_names"],labels,"grid_moran")
        gd.to_csv(os.path.join(outdir,"gene_spatial_metrics_grid.csv"),index=False)
        gt_top, gsumm=topk_metrics(gd,topks,positive); gt_top.insert(0,"scenario","grid"); gt_top.to_csv(os.path.join(outdir,"svg_overlap_grid.csv"),index=False)
        result.update({"grid_n_bins":int(len(xyz)),"grid_target":grid_target,"grid_spearman_I":gsumm["spearman_I"],"grid_pearson_I":gsumm["pearson_I"],"grid_median_abs_delta_I":gsumm["median_abs_delta_I"]})
    with open(os.path.join(outdir,"run_metadata.json"),"w") as f:
        json.dump({"path":os.path.abspath(path),"dataset_kind":dataset_kind,"seed":seed,"sample_size":sample_size,"graph_k":k,"scenario":scenario,"python":sys.version,"platform":platform.platform()},f,indent=2)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["scale","domain"],required=True)
    ap.add_argument("--result",required=True); ap.add_argument("--output_dir",required=True)
    ap.add_argument("--sample_size",type=int,default=100000); ap.add_argument("--k",type=int,default=15)
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--topk",type=int,nargs="+",default=[5,10,20,50])
    ap.add_argument("--grid_bins_target",type=int,default=0); ap.add_argument("--domain_k",type=int,default=0)
    ap.add_argument("--scenarios",nargs="+",default=["full"])
    args=ap.parse_args(); os.makedirs(args.output_dir,exist_ok=True)
    rows=[]
    for sc in args.scenarios:
        rows.append(evaluate_one(args.result,args.output_dir,args.dataset,args.sample_size,args.k,args.seed,args.topk,args.grid_bins_target or None,sc,args.domain_k or None))
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir,"summary.csv"),index=False)


if __name__=="__main__": main()
