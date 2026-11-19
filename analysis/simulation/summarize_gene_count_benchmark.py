#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize the R2M4 gene-count benchmark (NTF vs SpatialZ vs FEAST).

Collects, for every (method, n_genes):
  - central per-gene metrics (gene_metrics_central.csv, evaluated against the
    real simulation truth by central_eval.py),
  - SVG conservation outputs (svg/{method}/{g}gene/summary.csv + svg_overlap_full.csv),
  - efficiency metadata (monitor.json wall time / peak GPU / peak RSS, plus the
    inference time logged in res.log).

Writes into the experiment directory:
  benchmark_long.csv, method_status.csv, summary_by_genes.csv,
  svg_overlap_summary.csv, efficiency_summary.csv, figures/*.png
"""
from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = "/home/gongyuqiao/ur_annotation/NTF"
EXP = f"{ROOT}/mytrain/revision/experiment/gene_count_benchmark"
GENES = [1, 10, 100, 1000, 10000]

METHOD_DIRS = {
    "NTF": f"{ROOT}/mytrain/res/simulation/gene_count_test",
    "FEAST": f"{ROOT}/mytrain/res/FEAST/gene_count_test",
    "SpatialZ": "/home/gongyuqiao/ur_annotation/SpatialZ/res/simulation/gene_count_test",
}


def result_dir(method, g):
    return os.path.join(METHOD_DIRS[method], f"100wcell_{g}gene")


def read_infer_time(res_dir):
    """Inference/reconstruction seconds reported in res.log, if any."""
    p = os.path.join(res_dir, "res.log")
    if not os.path.exists(p):
        return np.nan
    txt = open(p).read()
    m = re.search(r"Inference [Tt]ime \(seconds\): ([\d.]+)", txt) or re.search(r"Inference Time: ([\d.]+)", txt)
    return float(m.group(1)) if m else np.nan


def read_ntf_train_time(res_dir):
    """NTF training wall time: first log timestamp -> 'Evaluating on the test set'.

    logging writes to stderr, which run_with_monitor.sh redirects to
    train.log.time, so read that file first and fall back to train.log.
    tqdm progress lines carry no timestamp and are ignored.
    """
    from datetime import datetime
    for name in ("train.log.time", "train.log", "eval.log.time", "eval.log"):
        p = os.path.join(res_dir, name)
        if not os.path.exists(p):
            continue
        ts = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - (.*)", open(p, errors="ignore").read())
        if not ts:
            continue
        t0 = datetime.strptime(ts[0][0], "%Y-%m-%d %H:%M:%S")
        for stamp, msg in ts:
            if "Evaluating on the test set" in msg:
                return (datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S") - t0).total_seconds()
    return np.nan


def main():
    os.makedirs(f"{EXP}/figures", exist_ok=True)
    long_rows, eff_rows, status_rows, svg_rows = [], [], [], []

    for method in METHOD_DIRS:
        for g in GENES:
            rd = result_dir(method, g)
            central = os.path.join(rd, "gene_metrics_central.csv")
            status = "completed" if os.path.exists(central) else (
                "NA" if os.path.exists(os.path.join(rd, "STATUS_NA.md")) else "missing")
            status_rows.append({"method": method, "n_genes": g, "status": status})

            if status == "completed":
                df = pd.read_csv(central)
                df["method"] = method
                df["n_genes"] = g
                long_rows.append(df)

            # efficiency
            eff = {"method": method, "n_genes": g, "status": status,
                   "wall_time_s": np.nan, "infer_time_s": np.nan,
                   "peak_gpu_mem_mib": np.nan, "peak_rss_kb": np.nan}
            mp = os.path.join(rd, "monitor.json")
            if os.path.exists(mp):
                m = json.load(open(mp))
                eff["wall_time_s"] = m.get("wall_time_s", np.nan)
                eff["peak_gpu_mem_mib"] = m.get("peak_gpu_mem_mib", np.nan)
                eff["peak_rss_kb"] = m.get("peak_rss_kb", np.nan)
            eff["infer_time_s"] = read_infer_time(rd)
            if method == "NTF":
                eff["train_time_s"] = read_ntf_train_time(rd)
            eff_rows.append(eff)

            # svg
            sp = f"{EXP}/svg/{method.lower()}/{g}gene"
            s = os.path.join(sp, "summary.csv")
            if os.path.exists(s):
                row = pd.read_csv(s).iloc[0].to_dict()
                row.update({"method": method, "n_genes": g})
                svg_rows.append(row)

    svg_overlap = []
    # re-collect overlap tables (kept separate from per-scenario summary)
    for method in METHOD_DIRS:
        for g in GENES:
            ov = f"{EXP}/svg/{method.lower()}/{g}gene/svg_overlap_full.csv"
            if os.path.exists(ov):
                odf = pd.read_csv(ov)
                odf["method"] = method
                odf["n_genes"] = g
                svg_overlap.append(odf)

    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    eff_df = pd.DataFrame(eff_rows)
    status_df = pd.DataFrame(status_rows)
    svg_df = pd.DataFrame(svg_rows)
    svg_ov_df = pd.concat(svg_overlap, ignore_index=True) if svg_overlap else pd.DataFrame()

    long_df.to_csv(f"{EXP}/benchmark_long.csv", index=False)
    status_df.to_csv(f"{EXP}/method_status.csv", index=False)
    eff_df.to_csv(f"{EXP}/efficiency_summary.csv", index=False)
    svg_ov_df.to_csv(f"{EXP}/svg_overlap_summary.csv", index=False)

    # manifest: every run's command, paths and status
    manifest_rows = []
    for method in METHOD_DIRS:
        for g in GENES:
            rd = result_dir(method, g)
            row = {"method": method, "n_genes": g, "result_dir": rd,
                   "input_data": f"{ROOT}/mytrain/data/simulation/gene_count_test/100wcell_{g}gene.h5ad"}
            for logname in ("train.log.env", "run.log.env", "eval.log.env"):
                p = os.path.join(rd, logname)
                if os.path.exists(p):
                    row["cmd"] = open(p).read().strip().replace("CMD: ", "")
                    break
            mp = os.path.join(rd, "monitor.json")
            if os.path.exists(mp):
                m = json.load(open(mp))
                row.update({"conda_env": m.get("conda_env"), "gpu_id": m.get("gpu_id"),
                            "exit_code": m.get("exit_code"), "wall_time_s": m.get("wall_time_s")})
            manifest_rows.append(row)
    pd.DataFrame(manifest_rows).to_csv(f"{EXP}/manifest.csv", index=False)

    # per (method, n_genes) summary
    if not long_df.empty:
        summ = long_df.groupby(["method", "n_genes"]).agg(
            pearson_mean=("pearson_corr", "mean"), pearson_median=("pearson_corr", "median"),
            spearman_mean=("spearman_corr", "mean"), spearman_median=("spearman_corr", "median"),
            rmse_mean=("rmse", "mean"), ssim_3d_mean=("ssim_3d", "mean"),
        ).reset_index()
        if not svg_df.empty:
            summ = summ.merge(
                svg_df[["method", "n_genes", "spearman_I", "pearson_I", "median_abs_delta_I"]],
                on=["method", "n_genes"], how="left")
        summ.to_csv(f"{EXP}/summary_by_genes.csv", index=False)

    # ---- figures ----
    colors = {"NTF": "#d62728", "SpatialZ": "#1f77b4", "FEAST": "#2ca02c"}

    def lineplot(metric, ylabel, fname, logy=False):
        plt.figure(figsize=(5, 4))
        for method in METHOD_DIRS:
            sub = summ[summ.method == method].dropna(subset=[metric]).sort_values("n_genes")
            if len(sub):
                plt.plot(sub.n_genes, sub[metric], "o-", label=method, color=colors[method])
        plt.xscale("log")
        if logy:
            plt.yscale("log")
        plt.xlabel("Number of genes")
        plt.ylabel(ylabel)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(f"{EXP}/figures/{fname}", dpi=200)
        plt.close()

    if not long_df.empty:
        lineplot("pearson_mean", "Mean Pearson (vs truth)", "pearson_vs_ngenes.png")
        lineplot("spearman_mean", "Mean Spearman (vs truth)", "spearman_vs_ngenes.png")
        if "spearman_I" in summ:
            lineplot("spearman_I", "Moran's I Spearman (GT vs pred)", "moran_corr_vs_ngenes.png")

        # time vs genes (total wall time)
        plt.figure(figsize=(5, 4))
        for method in METHOD_DIRS:
            sub = eff_df[(eff_df.method == method) & (eff_df.status == "completed")].sort_values("n_genes")
            col = "train_time_s" if method == "NTF" else "infer_time_s"
            if col not in sub:
                col = "wall_time_s"
            sub = sub.dropna(subset=[col])
            if len(sub):
                plt.plot(sub.n_genes, sub[col], "o-", label=f"{method} ({'train' if method=='NTF' else 'reconstruct'})", color=colors[method])
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("Number of genes"); plt.ylabel("Time (s)")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{EXP}/figures/time_vs_ngenes.png", dpi=200)
        plt.close()

        # Top-K Jaccard at the largest common K (20) vs genes
        if not svg_ov_df.empty:
            plt.figure(figsize=(5, 4))
            for method in METHOD_DIRS:
                sub = svg_ov_df[(svg_ov_df.method == method) & (svg_ov_df.topk == 20)].sort_values("n_genes")
                if len(sub):
                    plt.plot(sub.n_genes, sub.jaccard, "o-", label=method, color=colors[method])
            plt.xscale("log")
            plt.xlabel("Number of genes"); plt.ylabel("Top-20 SVG Jaccard")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(f"{EXP}/figures/svg_jaccard_vs_ngenes.png", dpi=200)
            plt.close()

        # per-gene pearson distribution at 1000/10000
        plt.figure(figsize=(6, 4))
        data, labels, cols = [], [], []
        for method in METHOD_DIRS:
            for g in (1000, 10000):
                sub = long_df[(long_df.method == method) & (long_df.n_genes == g)]
                if len(sub):
                    data.append(sub.pearson_corr.dropna().values)
                    labels.append(f"{method}\n{g} genes")
                    cols.append(colors[method])
        if data:
            vp = plt.violinplot(data, showmedians=True)
            for body, c in zip(vp["bodies"], cols):
                body.set_facecolor(c); body.set_alpha(0.6)
            plt.xticks(range(1, len(labels) + 1), labels, fontsize=8)
            plt.ylabel("Per-gene Pearson (vs truth)")
            plt.tight_layout()
            plt.savefig(f"{EXP}/figures/per_gene_distribution_1000_10000.png", dpi=200)
            plt.close()

    print("status matrix:")
    print(status_df.pivot(index="n_genes", columns="method", values="status"))
    if not long_df.empty:
        print(summ.to_string(index=False))

    # ---- easy-to-read Excel workbook (per revision/AGENTS.md) ----
    xlsx = f"{EXP}/gene_count_benchmark_summary.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        if not long_df.empty:
            cols = [c for c in ["method", "n_genes", "pearson_mean", "pearson_median",
                                "spearman_mean", "spearman_median", "rmse_mean", "ssim_3d_mean",
                                "spearman_I", "pearson_I", "median_abs_delta_I"] if c in summ]
            summary_x = summ[cols].merge(
                eff_df[["method", "n_genes", "train_time_s", "infer_time_s", "wall_time_s",
                        "peak_gpu_mem_mib"]] if "train_time_s" in eff_df else eff_df,
                on=["method", "n_genes"], how="left")
            summary_x.to_excel(xw, sheet_name="Summary", index=False)
        status_df.pivot(index="n_genes", columns="method", values="status").to_excel(xw, sheet_name="Method_Status")
        if not svg_ov_df.empty:
            svg_ov_df.to_excel(xw, sheet_name="SVG_Overlap_TopK", index=False)
        eff_df.to_excel(xw, sheet_name="Efficiency", index=False)
        meta = pd.DataFrame([
            ["experiment", "R2 Major4 gene-count benchmark (NTF vs SpatialZ vs FEAST)"],
            ["cells", "1,000,000 (100 slices x 10,000 cells); test cells 406,897"],
            ["gene_counts", "1, 10, 100, 1000, 10000"],
            ["eval_target", "layers['truth'] of the input simulation (central re-evaluation)"],
            ["svg_eval", "100,000 slice-stratified sample, 3D kNN k=15, seed=42 (same sample for all methods)"],
            ["seed", 42],
            ["record", "README_results.md"],
            ["summary_script", "mytrain/scripts/summarize_gene_count_benchmark.py"],
        ], columns=["key", "value"])
        meta.to_excel(xw, sheet_name="Metadata", index=False)
    print(f"excel -> {xlsx}")


if __name__ == "__main__":
    main()
