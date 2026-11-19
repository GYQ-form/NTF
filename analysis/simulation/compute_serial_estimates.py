"""Serial-equivalent cost estimates for UniST runs (per 2026-10-04 policy).

UniST's intrinsic cost unit is one gap (per-gap GAE+INR from-scratch training).
Serial-equivalent total = sum over gaps of the UNCONTENDED single-gap cost, modeled
as  t(n) = 2.5 * n^0.708  (n = flanking cell count), a power-law fit to seven
uncontended reference measurements (fit residuals within ±30%; see README):
  quickstart inr 4k→1308s, mouseVC s1 g0-2 8.4-9k→1150-1342s, 10w smoke 25.6k→3207s,
  interval k50 g0 28.6k→3692s, 1000g typical gap 45.3k→5749s.
inr runs executed with parallel stacking would otherwise overstate serial time
(measured +36% per-gap at 4-way stacking).

fast mode: per-gap cost is dominated by InterpolAI CPU inference + 1-NN mapping
(seconds per gap, scheduling-insensitive); serial estimate = measured sum.

Reads every per_gap_timing.csv / unist_*_per_gap_k*.csv, writes
efficiency_summary.csv with columns:
  serial_estimate_s, measured_sum_per_gap_s, wall_time_s, speedup_strategy, ...
"""
import glob
import json
import os
import re
import numpy as np
import pandas as pd
import anndata

EXP = "/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/r2t5_unist_benchmark"
BASE = "/home/gongyuqiao/ur_annotation/NTF/mytrain"


def model_inr(n):
    return 2.5 * float(n) ** 0.708


_slice_cache = {}

def slice_counts(h5ad, slice_col):
    key = (h5ad, slice_col)
    if key not in _slice_cache:
        ad = anndata.read_h5ad(h5ad, backed="r")
        _slice_cache[key] = ad.obs[slice_col].astype(str).value_counts().to_dict()
    return _slice_cache[key]


def estimate(per_gap_df, h5ad, slice_col, mode):
    """Return (serial_estimate_s, measured_sum_s, n_gaps)."""
    cnt = slice_counts(h5ad, slice_col)
    tot_model = 0.0
    tot_meas = 0.0
    for _, r in per_gap_df.iterrows():
        fl = str(r["flanking"])
        if fl.startswith("proxy:"):
            continue  # boundary proxy: no UniST compute
        a, b = fl.split("|")
        n = cnt.get(a, 0) + cnt.get(b, 0)
        tot_model += model_inr(n) if mode == "inr" else float(r["wall_time_s"])
        tot_meas += float(r["wall_time_s"])
    return tot_model, tot_meas, len(per_gap_df)


rows = []

def add(benchmark, tier, mode, per_gap_csv, h5ad, slice_col, monitor_json, strategy):
    if not os.path.exists(per_gap_csv):
        return
    try:
        pg = pd.read_csv(per_gap_csv)
    except pd.errors.EmptyDataError:  # e.g. k=1: zero gaps by construction
        rows.append({"benchmark": benchmark, "scale": tier, "method": f"UniST-{mode}",
                     "n_gaps": 0, "serial_estimate_s": 0.0,
                     "measured_sum_per_gap_s": 0.0, "speedup_strategy": strategy})
        return
    est, meas, ng = estimate(pg, h5ad, slice_col, mode)
    row = {"benchmark": benchmark, "scale": tier, "method": f"UniST-{mode}",
           "n_gaps": ng, "serial_estimate_s": round(est, 1),
           "measured_sum_per_gap_s": round(meas, 1),
           "speedup_strategy": strategy}
    if os.path.exists(monitor_json):
        j = json.load(open(monitor_json))
        row.update(wall_time_s=j["wall_time_s"], peak_gpu_mem_mib=j["peak_gpu_mem_mib"],
                   peak_rss_kb=j["peak_rss_kb"], gpu_id=j["gpu_id"], exit_code=j["exit_code"])
    rows.append(row)


# --- scale test ---
UNIST_INR_TIERS = {10, 20, 30, 40, 50, 100}
TIERS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for n in TIERS:
    h5 = f"{BASE}/data/simulation/scale_test/{n}w.h5ad"
    if n in UNIST_INR_TIERS:
        d = f"{BASE}/res/UniST/scale_test/{n}w" if n == 10 else f"{BASE}/res/UniST/scale_test/{n}w_inr"
        mon = f"{EXP}/monitor/scale_{n}w_inr.json"
        strat = {10: "GPU6 dedicated (fast serial on GPU7, mild CPU contention)",
                 100: "GPU6 dedicated (GPU7 ran 20-50w inr in parallel)"}.get(
                 n, "2-way parallel across GPUs (shared CPU)")
        add("scale_test", f"{n}w", "inr", f"{d}/per_gap_timing.csv", h5, "slice_id", mon, strat)
    d = f"{BASE}/res/UniST/scale_test/{n}w_fast" if n == 10 else f"{BASE}/res/UniST/scale_test/{n}w"
    mon = f"{EXP}/monitor/scale_{n}w_fast.json"
    add("scale_test", f"{n}w", "fast", f"{d}/per_gap_timing.csv", h5, "slice_id", mon,
        "serial on GPU7 (fast sweep)")

# --- interval ---
GT = f"{BASE}/res/simulation/domain_interval/ground_truth_full.h5ad"
for mode in ["inr", "fast"]:
    for f in glob.glob(f"{BASE}/res/UniST/interval/unist_{mode}_per_gap_k*.csv"):
        k = re.search(r"_k(\d+)\.csv", f).group(1)
        mon = f"{EXP}/monitor/interval_k{k}_{mode}.json"
        if not os.path.exists(mon):
            mon = f"{EXP}/monitor/interval_allk_{mode}.json"
        add("interval", f"k{k}", mode, f, GT, "slice_id", mon,
            "serial per-k on GPU6" if mode == "inr" else "single sweep run on GPU7")

# --- mouseVC ---
MV = f"{BASE}/data/mouse_VC/processed/mouseVC_5gene.h5ad"
for s in ["senario1", "senario2"]:
    add("mouseVC_5gene", s, "inr",
        f"{BASE}/res/UniST/mouseVC_5gene/{s}/per_gap_timing.csv", MV, "slice_id",
        f"{EXP}/monitor/mouseVC_{s}_inr.json", "GPU6 dedicated")

# --- deep_starmap v3 ---
DS = f"{BASE}/data/deep_starmap_brain/processed/deep_starmap_brain_v3.h5ad"
for s in ["senario1", "senario2"]:
    mon = f"{EXP}/monitor/deepstarmap_{s}_inr.json"
    if not os.path.exists(mon):
        mon = f"{EXP}/monitor/deepstarmap_{s}_label_inr.json"
    add("deep_starmap", s, "inr",
        f"{BASE}/res/UniST/deep_starmap_brain_v3/{s}/per_gap_timing.csv", DS,
        "slice_id_v2", mon, "parallel stacked on GPU6 (model-corrected)")

# --- weMERFISH ---
WM = f"{BASE}/data/wemerfish_zebrafish_embryo/processed/wemerfish_C6s_E2.h5ad"
for d, scn, lab in [("senario1", "senario1", "clusters"), ("senario1_niche", "senario1", "type"),
                    ("senario2", "senario2", "clusters"), ("senario2_niche", "senario2", "type")]:
    add("wemerfish", f"{scn}:{lab}", "inr",
        f"{BASE}/res/UniST/wemerfish_C6s_E2/{d}/per_gap_timing.csv", WM, "slice_id",
        f"{EXP}/monitor/wemerfish_{'s1' if scn=='senario1' else 's2'}_{lab}_inr.json",
        "parallel stacked (model-corrected)")

# --- CyCIF ---
CY = f"{BASE}/data/cycif_crc1_3d/processed/cycif_crc1_bm_aligned.h5ad"
for s, col in [("senario1", "group"), ("senario2", "group2")]:
    add("cycif_crc1", s, "fast",
        f"{BASE}/res/UniST/cycif_crc1_aligned/{s}_fast/per_gap_timing.csv", CY,
        "slice_id", f"{EXP}/monitor/cycif_{s}_fast.json", "serial on GPU7")


# --- keep existing non-UniST rows from previous efficiency_summary.csv ---
prev_path = f"{EXP}/efficiency_summary.csv"
if os.path.exists(prev_path):
    prev = pd.read_csv(prev_path)
    prev = prev[~prev["method"].astype(str).str.startswith("UniST")]
    rows.extend(prev.to_dict("records"))

df = pd.DataFrame(rows)
df.to_csv(prev_path, index=False)
pd.set_option("display.width", 200)
print(df[df.method.str.startswith("UniST")].to_string(index=False))
