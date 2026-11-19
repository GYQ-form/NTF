#!/usr/bin/env python
"""Compile manifest.csv + efficiency_summary.csv for the CyCIF CRC1 benchmark.

Adapted from analysis/deep_starmap_brain/build_manifest_efficiency.py:
- NTF only; monitor files are named ntf_<scenario>.json (lowercase).
- Timing methodology (train + inference only, excluding I/O):
  NTF train_time_s: 'Training finished. Time elapsed' in logs/ntf_<scen>.log.time
  NTF infer_time_s: 'Inference time (seconds):' in <ntf_res>/<scenario>/res.log

Usage:
    python build_manifest_efficiency_cycif.py --exp_dir <experiment/cycif_crc1_benchmark>
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

SCENARIOS = ["senario1", "senario2", "senario3", "head", "tail"]


def parse_ntf_train_time(logtime_path):
    txt = Path(logtime_path).read_text(errors="ignore")
    m = re.findall(r"Training finished\. Time elapsed:\s*([0-9.]+) seconds", txt)
    return float(m[-1]) if m else None


def parse_ntf_infer_time(ntf_res, scenario):
    txt = (Path(ntf_res) / scenario / "res.log").read_text(errors="ignore")
    m = re.search(r"Inference time \(seconds\):\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--ntf_res", default="/home/gongyuqiao/ur_annotation/NTF/mytrain/res/cycif_crc1")
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir)

    # --- manifest ---
    rows = []
    for jf in sorted((exp_dir / "monitor").glob("*.json")):
        d = json.loads(jf.read_text())
        rows.append({
            "run": jf.stem,
            "cmd": d["cmd"],
            "conda_env": d["conda_env"],
            "gpu_id": d["gpu_id"],
            "wall_time_s": d["wall_time_s"],
            "exit_code": d["exit_code"],
            "peak_gpu_mem_mib": d["peak_gpu_mem_mib"],
            "peak_rss_kb": d["peak_rss_kb"],
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(exp_dir / "manifest.csv", index=False)
    print(manifest[["run", "wall_time_s", "exit_code"]].to_string(index=False))

    # --- efficiency (NTF only) ---
    eff_rows = []
    for scen in SCENARIOS:
        row = {"method": "NTF", "scenario": scen}
        mon = exp_dir / "monitor" / f"ntf_{scen}.json"
        if not mon.exists():
            row.update(status="missing")
        else:
            d = json.loads(mon.read_text())
            row.update(wall_time_s=d["wall_time_s"], peak_gpu_mem_mib=d["peak_gpu_mem_mib"],
                       peak_rss_kb=d["peak_rss_kb"],
                       status="completed" if d["exit_code"] == 0 else f"exit={d['exit_code']}")
            row["train_time_s"] = parse_ntf_train_time(exp_dir / "logs" / f"ntf_{scen}.log.time")
            row["infer_time_s"] = parse_ntf_infer_time(args.ntf_res, scen)
            tt = row.get("train_time_s")
            it = row.get("infer_time_s")
            row["compute_time_s"] = (tt or 0) + (it or 0) if (tt is not None or it is not None) else None
            row["note"] = "bs=8192, n_iter=20000 (default), fp32, weight_decay=0, no_dropout"
        eff_rows.append(row)
    # FEAST/SpatialZ: infeasible at this scale, see baseline_feasibility/
    for scen in SCENARIOS:
        for method in ["FEAST", "SpatialZ"]:
            eff_rows.append({"method": method, "scenario": scen, "status": "NA",
                             "note": "infeasible at 12.8M cells (OT cost matrix ~1 TB per slice pair); "
                                     "see baseline_feasibility/ for scaling evidence"})
    eff = pd.DataFrame(eff_rows)[["method", "scenario", "train_time_s", "infer_time_s",
                                  "compute_time_s", "wall_time_s", "peak_gpu_mem_mib",
                                  "peak_rss_kb", "status", "note"]]
    eff.to_csv(exp_dir / "efficiency_summary.csv", index=False)
    print(eff.to_string(index=False))


if __name__ == "__main__":
    main()
