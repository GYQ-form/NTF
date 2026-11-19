#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Central re-evaluation of a method's test_results.h5ad against the *real*
simulation ground truth (layers['truth'] of the input dataset).

Rationale: the historical per-method pipelines evaluate against different
targets (NTF train_and_eval.py and FEAST bm_feast.py / SpatialZ train_ur.py
evaluate against observed X; the old train_dropout_simu.py evaluated against
layers['truth']). For the gene-count benchmark all methods are re-evaluated
here against the same target, on the same test cells, with vectorized metrics.

Outputs (in --output_dir):
  gene_metrics_central.csv   per-gene pearson/spearman/rmse vs real truth
  eval_input.h5ad            standardized file for eval_svg_conservation_sim.py:
                             X=observed test expr, layers/truth=real truth,
                             layers/prediction, obs slice_id/cluster_id/group,
                             obsm/spatial, var pattern_name
  central_eval_meta.json     run metadata
"""
from __future__ import annotations

import argparse
import json
import os
import time

import anndata
import h5py
import numpy as np
import pandas as pd


def _decode(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def read_obs_names(path):
    with h5py.File(path, "r") as f:
        return np.array([_decode(x) for x in f["obs/_index"][:]], dtype=object)


def read_rows(path, indices, dataset):
    indices = np.asarray(indices, dtype=np.int64)
    order = np.argsort(indices)
    sorted_idx = indices[order]
    with h5py.File(path, "r") as f:
        d = f[dataset]
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
    return arr


def ranks_cols(x):
    """Column-wise average ranks (ties handled like scipy rankdata)."""
    from scipy.stats import rankdata
    out = np.empty_like(x, dtype=np.float64)
    for j in range(x.shape[1]):
        out[:, j] = rankdata(x[:, j])
    return out


def pearson_cols(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    denom = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(0) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="original simulated h5ad (source of layers/truth)")
    ap.add_argument("--result", required=True, help="method test_results.h5ad (source of layers/prediction)")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    res_obs = read_obs_names(args.result)
    in_obs = read_obs_names(args.input)
    in_index = {name: i for i, name in enumerate(in_obs)}
    try:
        in_rows = np.array([in_index[name] for name in res_obs], dtype=np.int64)
    except KeyError as e:
        raise SystemExit(f"result obs name not found in input: {e}")

    with h5py.File(args.result, "r") as f:
        n_vars = int(f["X"].shape[1])
        var_names = np.array([_decode(x) for x in f["var/_index"][:]], dtype=object)
        obs_cols = {}
        for col in ("slice_id", "cluster_id", "group"):
            p = f"obs/{col}"
            if f"{p}/codes" in f:
                cats = np.array([_decode(x) for x in f[f"{p}/categories"][:]], dtype=object)
                obs_cols[col] = cats[np.asarray(f[f"{p}/codes"][:])]
        spatial = np.asarray(f["obsm/spatial"][:], dtype=np.float32)

    print("reading prediction ...", flush=True)
    pred = read_rows(args.result, np.arange(len(res_obs)), "layers/prediction")
    print("reading observed X ...", flush=True)
    xobs = read_rows(args.result, np.arange(len(res_obs)), "X")
    print("reading real truth from input ...", flush=True)
    truth = read_rows(args.input, in_rows, "layers/truth")

    # var pattern_name from input (result files may not carry it)
    with h5py.File(args.input, "r") as f:
        if "var/pattern_name" in f:
            p = f["var/pattern_name"]
            if "codes" in p and "categories" in p:
                cats = np.array([_decode(x) for x in p["categories"][:]], dtype=object)
                pattern = cats[np.asarray(p["codes"][:])]
            else:
                pattern = np.array([_decode(x) for x in p[:]], dtype=object)
        else:
            pattern = np.array(["NA"] * n_vars, dtype=object)

    print("computing metrics ...", flush=True)
    pear = pearson_cols(truth.astype(np.float64), pred.astype(np.float64))
    rmse = np.sqrt(((truth - pred) ** 2).mean(0))
    # column-wise spearman via ranks (chunked to bound memory)
    spear = np.full(n_vars, np.nan)
    chunk = 500
    for g0 in range(0, n_vars, chunk):
        g1 = min(g0 + chunk, n_vars)
        rt = ranks_cols(truth[:, g0:g1].astype(np.float64))
        rp = ranks_cols(pred[:, g0:g1].astype(np.float64))
        spear[g0:g1] = pearson_cols(rt, rp)

    df = pd.DataFrame({
        "gene": var_names, "pattern_name": pattern,
        "pearson_corr": pear, "spearman_corr": spear, "rmse": rmse,
    })

    # 3D SSIM vs real truth, using the project's standard implementation
    # (voxelize via coords.astype(int), per-gene structural similarity).
    print("computing ssim_3d ...", flush=True)
    from NTF.helper_func import calculate_spatial_metrics
    tmp = anndata.AnnData(
        X=truth,
        obs=pd.DataFrame(index=res_obs),
        var=pd.DataFrame(index=var_names),
        obsm={"spatial": spatial},
        layers={"prediction": pred},
    )
    calculate_spatial_metrics(tmp)
    df["ssim_3d"] = tmp.var["ssim_3d"].values
    df["rmse"] = tmp.var["rmse"].values

    df.to_csv(os.path.join(args.output_dir, "gene_metrics_central.csv"), index=False)

    adata = anndata.AnnData(
        X=xobs,
        obs=pd.DataFrame(obs_cols, index=res_obs),
        var=pd.DataFrame({"pattern_name": pd.Categorical(pattern)}, index=var_names),
        obsm={"spatial": spatial},
        layers={"truth": truth, "prediction": pred},
    )
    out_h5ad = os.path.join(args.output_dir, "eval_input.h5ad")
    print(f"writing {out_h5ad} (gzip) ...", flush=True)
    adata.write_h5ad(out_h5ad, compression="gzip", compression_opts=4)

    meta = {
        "input": os.path.abspath(args.input),
        "result": os.path.abspath(args.result),
        "n_test_cells": int(len(res_obs)),
        "n_vars": n_vars,
        "eval_target": "layers/truth of input (real simulation ground truth)",
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(args.output_dir, "central_eval_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"done in {meta['elapsed_s']:.1f}s; mean pearson={np.nanmean(pear):.4f}, mean spearman={np.nanmean(spear):.4f}")


if __name__ == "__main__":
    main()
