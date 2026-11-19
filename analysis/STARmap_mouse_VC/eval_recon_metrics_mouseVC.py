#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified per-gene reconstruction metrics for the mouseVC 5-gene benchmark.

Computes pearson_corr / spearman_corr / rmse / ssim_3d per gene for every
available method x scenario test_results.h5ad with ONE implementation (same
approach as scripts/eval_recon_metrics_deepstarmap.py), and writes one CSV per
method into <output_dir>/<scenario>/<METHOD>.csv (index = gene name, columns =
pearson_corr, spearman_corr, rmse, ssim_3d).

Metric definitions (identical to the deep_starmap unified implementation):
- pearson/spearman: same guards as scripts/train_and_eval.py (computed on all
  test values; NaN if <2 non-zero GT values or near-zero variance on the
  GT-nonzero subset);
- rmse / ssim_3d: NTF.helper_func.calculate_spatial_metrics (coordinates
  downscaled to a <=64^3 grid before SSIM).

The wrapper-produced gene_metrics.csv files are NOT used: SpatialZ/FEAST SSIM
implementations differ from NTF's. senario3/head/tail are NTF-only
extrapolation scenarios (UniST interpolates between flanking train slices and
does not extrapolate; FEAST/SpatialZ likewise require both flanks).

Run:  conda activate ntf
      python mytrain/scripts/eval_recon_metrics_mouseVC.py \
          --output_dir mytrain/revision/experiment/cell_typing_benchmark/mouseVC
"""
from __future__ import annotations

import argparse
import warnings
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


def build_results():
    base = "mouseVC/mouseVC_5gene"
    return {
        "senario1": {
            "NTF": NTF_ROOT / f"mytrain/res/{base}/senario1/test_results.h5ad",
            "FEAST": NTF_ROOT / "mytrain/res/FEAST/mouseVC/senario1/test_results.h5ad",
            "SpatialZ": SPATIALZ_RES / "mouseVC/5gene/senario1/test_results.h5ad",
            "UniST": NTF_ROOT / "mytrain/res/UniST/mouseVC_5gene/senario1/test_results.h5ad",
        },
        "senario2": {
            "NTF": NTF_ROOT / f"mytrain/res/{base}/senario2/test_results.h5ad",
            "FEAST": NTF_ROOT / "mytrain/res/FEAST/mouseVC/senario2/test_results.h5ad",
            "SpatialZ": SPATIALZ_RES / "mouseVC/5gene/senario2/test_results.h5ad",
            "UniST": NTF_ROOT / "mytrain/res/UniST/mouseVC_5gene/senario2/test_results.h5ad",
        },
        "senario3": {"NTF": NTF_ROOT / f"mytrain/res/{base}/senario3/test_results.h5ad"},
        "head": {"NTF": NTF_ROOT / f"mytrain/res/{base}/head/test_results.h5ad"},
        "tail": {"NTF": NTF_ROOT / f"mytrain/res/{base}/tail/test_results.h5ad"},
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


def per_gene_df(path):
    """Per-gene metric DataFrame (index=gene) for one test_results.h5ad."""
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
    calculate_spatial_metrics(adata)  # adds var['rmse'], var['ssim_3d']

    return pd.DataFrame({
        "pearson_corr": [pearson[g] for g in gene_names],
        "spearman_corr": [spearman[g] for g in gene_names],
        "rmse": adata.var["rmse"].to_numpy(),
        "ssim_3d": adata.var["ssim_3d"].to_numpy(),
    }, index=pd.Index(gene_names))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", required=True,
                    help="mouseVC metric dir; <METHOD>.csv written per scenario")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    for scenario, methods in build_results().items():
        for method, path in methods.items():
            if not path.exists():
                warnings.warn(f"Result file missing, skipped: {path}")
                continue
            df = per_gene_df(path)
            out = outdir / scenario / f"{method}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out)
            print(f"[{scenario}/{method}] n_genes={len(df)} "
                  f"pearson_mean={df.pearson_corr.mean():.4f} "
                  f"spearman_mean={df.spearman_corr.mean():.4f} "
                  f"rmse_mean={df.rmse.mean():.4f} "
                  f"ssim_mean={df.ssim_3d.mean():.4f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
