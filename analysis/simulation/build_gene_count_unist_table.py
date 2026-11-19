#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the UniST-inr / UniST-fast sheets for refine/gene_count_benchmark.xlsx.

Reads (all products of run_unist_gene_count_eval.sh + serial time policy):
  res/UniST/gene_count_test/{g}gene[_fast]/gene_metrics_central.csv  (central protocol, vs truth)
  revision/experiment/gene_count_benchmark/svg/unist_{mode}/{g}gene/summary.csv          (spearman_I)
  revision/experiment/gene_count_benchmark/svg/unist_{mode}/{g}gene/svg_overlap_full.csv (jaccard@topk20)
  revision/experiment/gene_count_benchmark/unist_gene_count_time.csv (serial-equivalent time)

Appends sheets 'UniST-inr' and 'UniST-fast' to refine/gene_count_benchmark.xlsx
(existing sheets are left untouched).
"""
import os
import numpy as np
import pandas as pd

BASE = "/home/gongyuqiao/ur_annotation/NTF/mytrain"
EXP = f"{BASE}/revision/experiment/gene_count_benchmark"
URES = f"{BASE}/res/UniST/gene_count_test"
XLSX = f"{EXP}/refine/gene_count_benchmark.xlsx"

time_df = pd.read_csv(f"{EXP}/unist_gene_count_time.csv")
time_map = {(r.method, r.n_genes): r.serial_min for r in time_df.itertuples()}


def build_sheet(mode, tiers):
    rows = []
    for g in tiers:
        rd = f"{URES}/{g}gene" if mode == "inr" else f"{URES}/{g}gene_fast"
        gm = pd.read_csv(f"{rd}/gene_metrics_central.csv")
        svg_dir = f"{EXP}/svg/unist_{mode}/{g}gene"
        summ = pd.read_csv(f"{svg_dir}/summary.csv")
        spearman_I = summ["spearman_I"].iloc[0]
        jaccard = np.nan
        ov_path = f"{svg_dir}/svg_overlap_full.csv"
        if os.path.exists(ov_path):
            ov = pd.read_csv(ov_path)
            hit = ov.loc[ov["topk"] == 20, "jaccard"]
            if len(hit):
                jaccard = hit.iloc[0]
        rows.append({
            "n_genes": g,
            "spearman_mean": round(float(np.nanmean(gm["spearman_corr"])), 6),
            "rmse_mean": round(float(np.nanmean(gm["rmse"])), 6),
            "ssim_3d_mean": round(float(np.nanmean(gm["ssim_3d"])), 6),
            "spearman_I": round(float(spearman_I), 6) if pd.notna(spearman_I) else np.nan,
            "jaccard": round(float(jaccard), 6) if pd.notna(jaccard) else np.nan,
            "time(min)": round(float(time_map[(f"UniST-{mode}", g)]), 2),
        })
    return pd.DataFrame(rows)


sheets = {
    "UniST-inr": build_sheet("inr", [10, 100, 1000, 10000]),
    "UniST-fast": build_sheet("fast", [1, 10, 100, 1000, 10000]),
}

existing = pd.read_excel(XLSX, sheet_name=None)
existing.update(sheets)
with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
    for name, df in existing.items():
        df.to_excel(w, sheet_name=name, index=False)
for name, df in sheets.items():
    print(f"--- {name} ---")
    print(df.to_string(index=False))
print("written:", XLSX)
