#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize R1 Major7 ablation results into CSVs + one Excel workbook.

Collects:
  - simulation side: per-config gene metrics from central_eval (vs layers['truth'])
  - mouseVC side: per-config x scenario gene metrics from train_and_eval (vs observed X)
  - efficiency: wall time / peak GPU mem / peak RSS / inference time from monitor.json + res.log

Outputs (in experiment/r1m7_model_ablation/results/):
  ablation_summary.xlsx        sheets: sim_summary, mousevc_summary, efficiency, sim_gene_metrics, mousevc_gene_metrics
  sim_summary.csv / mousevc_summary.csv / efficiency_summary.csv
"""
import json
import os
import re

import numpy as np
import pandas as pd

NTF_ROOT = "/home/gongyuqiao/ur_annotation/NTF"
SIM_RES = os.path.join(NTF_ROOT, "mytrain/res/simulation/ablation")
VC_RES = os.path.join(NTF_ROOT, "mytrain/res/mouseVC/ablation")
OUT_DIR = "/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/r1m7_model_ablation/results"

CONFIGS = ["M0_full", "M1_no_dropout", "M2_no_psf", "M3_no_smooth", "M4_no_sliceemb",
           "M5_no_pixvar", "M6_no_slicevar", "M7_no_uncertainty", "M8_bare", "M9_bias_on"]
CONFIG_LABEL = {
    "M0_full": "Full model",
    "M1_no_dropout": "- dropout network",
    "M2_no_psf": "- PSF modeling",
    "M3_no_smooth": "- smoothness regularization",
    "M4_no_sliceemb": "- slice embeddings",
    "M5_no_pixvar": "- pixel variance",
    "M6_no_slicevar": "- slice variance",
    "M7_no_uncertainty": "- uncertainty (both variances)",
    "M8_bare": "bare INR (all off)",
    "M9_bias_on": "+ bias correction (default off)",
}
SCENARIOS = ["senario1", "senario3"]


def read_monitor(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def read_inference_time(res_log):
    """Parse 'Inference time (seconds): X' from res.log."""
    if not os.path.exists(res_log):
        return np.nan
    with open(res_log) as f:
        txt = f.read()
    m = re.search(r"Inference time \(seconds\):\s*([\d.eE+-]+)", txt)
    return float(m.group(1)) if m else np.nan


def summarize_gene_metrics(df):
    cols = ["pearson_corr", "spearman_corr", "rmse", "ssim_3d"]
    out = {}
    for c in cols:
        if c in df.columns:
            out[f"{c}_mean"] = np.nanmean(df[c])
            out[f"{c}_median"] = np.nanmedian(df[c])
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- simulation side ----------
    sim_rows, sim_gene_dfs = [], []
    for cfg in CONFIGS:
        csv = os.path.join(SIM_RES, cfg, "central_eval", "gene_metrics_central.csv")
        mon = read_monitor(os.path.join(SIM_RES, cfg, "monitor.json"))
        row = {"config": cfg, "label": CONFIG_LABEL[cfg]}
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            df.insert(0, "config", cfg)
            sim_gene_dfs.append(df)
            row.update(summarize_gene_metrics(df))
        row["wall_time_s"] = mon.get("wall_time_s", np.nan)
        row["peak_gpu_mem_mib"] = mon.get("peak_gpu_mem_mib", np.nan)
        row["peak_rss_gb"] = mon.get("peak_rss_kb", 0) / 1e6 if mon else np.nan
        row["inference_time_s"] = read_inference_time(os.path.join(SIM_RES, cfg, "res.log"))
        row["exit_code"] = mon.get("exit_code", np.nan)
        sim_rows.append(row)
    sim_summary = pd.DataFrame(sim_rows)
    sim_genes = pd.concat(sim_gene_dfs, ignore_index=True) if sim_gene_dfs else pd.DataFrame()

    # ---------- mouseVC side ----------
    vc_rows, vc_gene_dfs = [], []
    for cfg in CONFIGS:
        for sc in SCENARIOS:
            # the dropout-on diagnostic run lives in M0_nan_check/ (senario1 only)
            cfg_dir = "M0_nan_check" if cfg == "M0_full" else cfg
            d = os.path.join(VC_RES, cfg_dir, sc)
            csv = os.path.join(d, "gene_metrics.csv")
            mon = read_monitor(os.path.join(d, "monitor.json"))
            row = {"config": cfg, "label": CONFIG_LABEL[cfg], "scenario": sc}
            if os.path.exists(csv):
                df = pd.read_csv(csv, index_col=0)
                df.insert(0, "config", cfg)
                df.insert(1, "scenario", sc)
                vc_gene_dfs.append(df.reset_index(drop=True))
                row.update(summarize_gene_metrics(df))
            row["wall_time_s"] = mon.get("wall_time_s", np.nan)
            row["peak_gpu_mem_mib"] = mon.get("peak_gpu_mem_mib", np.nan)
            row["peak_rss_gb"] = mon.get("peak_rss_kb", 0) / 1e6 if mon else np.nan
            row["inference_time_s"] = read_inference_time(os.path.join(d, "res.log"))
            row["exit_code"] = mon.get("exit_code", np.nan)
            vc_rows.append(row)
    vc_summary = pd.DataFrame(vc_rows)
    vc_genes = pd.concat(vc_gene_dfs, ignore_index=True) if vc_gene_dfs else pd.DataFrame()

    # ---------- efficiency view ----------
    eff = pd.concat([
        sim_summary[["config", "label", "wall_time_s", "inference_time_s", "peak_gpu_mem_mib", "peak_rss_gb"]].assign(dataset="simulation_1M_100gene"),
        vc_summary[["config", "label", "scenario", "wall_time_s", "inference_time_s", "peak_gpu_mem_mib", "peak_rss_gb"]].assign(dataset="mouseVC_5gene"),
    ], ignore_index=True)

    # ---------- write ----------
    sim_summary.to_csv(os.path.join(OUT_DIR, "sim_summary.csv"), index=False)
    vc_summary.to_csv(os.path.join(OUT_DIR, "mousevc_summary.csv"), index=False)
    eff.to_csv(os.path.join(OUT_DIR, "efficiency_summary.csv"), index=False)
    with pd.ExcelWriter(os.path.join(OUT_DIR, "ablation_summary.xlsx")) as xw:
        sim_summary.to_excel(xw, sheet_name="sim_summary", index=False)
        vc_summary.to_excel(xw, sheet_name="mousevc_summary", index=False)
        eff.to_excel(xw, sheet_name="efficiency", index=False)
        if not sim_genes.empty:
            sim_genes.to_excel(xw, sheet_name="sim_gene_metrics", index=False)
        if not vc_genes.empty:
            vc_genes.to_excel(xw, sheet_name="mousevc_gene_metrics", index=False)

    pd.set_option("display.width", 200)
    print("=== sim_summary ===")
    print(sim_summary.to_string(index=False))
    print("=== mousevc_summary ===")
    print(vc_summary.to_string(index=False))


if __name__ == "__main__":
    main()
