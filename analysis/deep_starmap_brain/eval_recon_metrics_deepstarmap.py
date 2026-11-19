#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified per-gene reconstruction metrics for the Deep-STARmap benchmark.

Computes pearson / spearman / rmse / ssim_3d for every available
method x scenario test_results.h5ad with ONE implementation, so the three
methods are compared apples-to-apples:

- pearson/spearman: same guards as scripts/train_and_eval.py (computed on all
  test values; NaN if <2 non-zero GT values or near-zero variance on the
  GT-nonzero subset);
- rmse / ssim_3d: NTF.helper_func.calculate_spatial_metrics (coordinates
  downscaled to a <=64^3 grid before SSIM, same as NTF's own outputs).

The wrapper-produced gene_metrics.csv of SpatialZ/FEAST is NOT used: their
SSIM implementations differ from NTF's and cannot handle physical-coordinate
grids (both wrappers were patched 2026-09-23 to skip oversized grids, leaving
ssim_3d=NaN for this dataset).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, "/home/gongyuqiao/ur_annotation/NTF")
from NTF.helper_func import calculate_spatial_metrics

NTF_ROOT = Path("/home/gongyuqiao/ur_annotation/NTF")
SPATIALZ_RES = Path("/home/gongyuqiao/ur_annotation/SpatialZ/res")

def build_results(res_suffix="", dataset="deep_starmap_brain"):
    return {
        "senario1": {
            "NTF": NTF_ROOT / f"mytrain/res/{dataset}{res_suffix}/senario1/test_results.h5ad",
            "FEAST": NTF_ROOT / f"mytrain/res/FEAST/{dataset}{res_suffix}/senario1/test_results.h5ad",
            "SpatialZ": SPATIALZ_RES / f"{dataset}{res_suffix}/senario1/test_results.h5ad",
            "UniST": NTF_ROOT / f"mytrain/res/UniST/{dataset}{res_suffix}/senario1/test_results.h5ad",
        },
        "senario2": {
            "NTF": NTF_ROOT / f"mytrain/res/{dataset}{res_suffix}/senario2/test_results.h5ad",
            "FEAST": NTF_ROOT / f"mytrain/res/FEAST/{dataset}{res_suffix}/senario2/test_results.h5ad",
            "SpatialZ": SPATIALZ_RES / f"{dataset}{res_suffix}/senario2/test_results.h5ad",
            "UniST": NTF_ROOT / f"mytrain/res/UniST/{dataset}{res_suffix}/senario2/test_results.h5ad",
        },
        "senario3": {"NTF": NTF_ROOT / f"mytrain/res/{dataset}{res_suffix}/senario3/test_results.h5ad"},
        "head": {"NTF": NTF_ROOT / f"mytrain/res/{dataset}{res_suffix}/head/test_results.h5ad"},
        "tail": {"NTF": NTF_ROOT / f"mytrain/res/{dataset}{res_suffix}/tail/test_results.h5ad"},
    }


def correlation_metrics(true_expr, pred_expr, gene_names):
    """Same guards as scripts/train_and_eval.py per-gene loop."""
    pearson, spearman = {}, {}
    for i, g in enumerate(gene_names):
        t, p = true_expr[:, i], pred_expr[:, i]
        nz = t > 0
        if nz.sum() < 2:
            pearson[g], spearman[g] = np.nan, np.nan
            continue
        if np.var(t[nz]) < 1e-10 or np.var(p[nz]) < 1e-10:
            pearson[g], spearman[g] = np.nan, np.nan
            continue
        try:
            pearson[g] = pearsonr(t, p)[0]
            spearman[g] = spearmanr(t, p)[0]
        except ValueError:
            pearson[g], spearman[g] = np.nan, np.nan
    return pearson, spearman


def evaluate_file(path, scenario, method):
    adata = ad.read_h5ad(path)
    assert "prediction" in adata.layers, f"layers['prediction'] missing: {path}"
    true_expr = adata.X
    if not isinstance(true_expr, np.ndarray):
        true_expr = true_expr.toarray()
    pred_expr = adata.layers["prediction"]
    if not isinstance(pred_expr, np.ndarray):
        pred_expr = pred_expr.toarray()
    assert np.isfinite(pred_expr).all(), f"NaN/Inf in prediction: {path}"

    gene_names = list(adata.var_names)
    pearson, spearman = correlation_metrics(true_expr, pred_expr, gene_names)
    calculate_spatial_metrics(adata)  # adds var['rmse'], var['ssim_3d'] (downscaled grid)

    df = pd.DataFrame({
        "scenario": scenario, "method": method, "gene": gene_names,
        "zero_frac": adata.var["zero_frac"].to_numpy() if "zero_frac" in adata.var else np.nan,
        "pearson": [pearson[g] for g in gene_names],
        "spearman": [spearman[g] for g in gene_names],
        "rmse": adata.var["rmse"].to_numpy(),
        "ssim_3d": adata.var["ssim_3d"].to_numpy(),
    })
    print(f"[{scenario}/{method}] n={adata.n_obs} pearson_mean={df.pearson.mean():.4f} "
          f"spearman_mean={df.spearman.mean():.4f} rmse_mean={df.rmse.mean():.4f} "
          f"ssim_mean={df.ssim_3d.mean():.4f}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--res_suffix", default="", help="Result-dir suffix, e.g. '_v2' for the v2 benchmark")
    ap.add_argument("--dataset", default="deep_starmap_brain", help="Dataset result-dir base name (e.g. wemerfish_C6s_E2)")
    ap.add_argument("--eff_zero_frac", type=float, default=0.95,
                    help="zero_frac threshold for the effective-gene sensitivity summary")
    args = ap.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames, status_rows = [], []
    for scenario, methods in build_results(args.res_suffix, dataset=args.dataset).items():
        for method in ("NTF", "FEAST", "SpatialZ", "UniST"):
            path = methods.get(method)
            if path is None or not Path(path).exists():
                status_rows.append({"scenario": scenario, "method": method,
                                    "status": "NA: no result (method cannot run this scenario)",
                                    "result_path": str(path) if path else ""})
                continue
            frames.append(evaluate_file(path, scenario, method))
            status_rows.append({"scenario": scenario, "method": method,
                                "status": "completed", "result_path": str(path)})

    long_df = pd.concat(frames, ignore_index=True)
    long_df.to_csv(outdir / "recon_metrics_long.csv", index=False)
    summary = (long_df.groupby(["scenario", "method"])
               .agg(n_genes=("gene", "size"),
                    pearson_mean=("pearson", "mean"), pearson_median=("pearson", "median"),
                    pearson_std=("pearson", "std"),
                    spearman_mean=("spearman", "mean"), spearman_median=("spearman", "median"),
                    spearman_std=("spearman", "std"),
                    rmse_mean=("rmse", "mean"), rmse_median=("rmse", "median"),
                    rmse_std=("rmse", "std"),
                    ssim_3d_mean=("ssim_3d", "mean"), ssim_3d_median=("ssim_3d", "median"),
                    ssim_3d_std=("ssim_3d", "std"))
               .reset_index())
    summary.to_csv(outdir / "recon_metrics_summary.csv", index=False)
    # effective-gene subset (zero_frac < 0.95) sensitivity summary
    eff = long_df[long_df.zero_frac < args.eff_zero_frac] if "zero_frac" in long_df else long_df.iloc[0:0]
    summary_eff = (eff.groupby(["scenario", "method"])
                   .agg(n_genes=("gene", "size"),
                        pearson_mean=("pearson", "mean"), pearson_median=("pearson", "median"),
                        pearson_std=("pearson", "std"),
                        spearman_mean=("spearman", "mean"), spearman_median=("spearman", "median"),
                        spearman_std=("spearman", "std"),
                        rmse_mean=("rmse", "mean"), rmse_median=("rmse", "median"),
                        rmse_std=("rmse", "std"),
                        ssim_3d_mean=("ssim_3d", "mean"), ssim_3d_median=("ssim_3d", "median"),
                        ssim_3d_std=("ssim_3d", "std"))
                   .reset_index())
    summary_eff.to_csv(outdir / "recon_metrics_summary_effective.csv", index=False)
    pd.DataFrame(status_rows).to_csv(outdir / "method_status_recon.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
