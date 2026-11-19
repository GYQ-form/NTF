#!/usr/bin/env python
"""Compile manifest.csv + efficiency_summary.csv from monitor JSONs and run logs.

Timing methodology (train + inference only, excluding I/O):
- NTF train_time_s: tqdm elapsed of the last 'Training NeuralTranscriptomicField'
  line in logs/<run>.log.time
- NTF infer_time_s: 'Inference time (seconds):' in <ntf_res>/<scenario>/res.log
- FEAST infer_time_s: 'Inference Time: X seconds' in logs/<run>.log
- SpatialZ infer_time_s: sum of per-slice phase times
  (coordinate initialization / Ot optimization / Cell type determination /
  Gene expression synthesis)

Expects an experiment directory with monitor/*.json (produced by
run_with_monitor.sh) and logs/.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

SCENARIOS = ["senario1", "senario2", "senario3", "head", "tail"]


def parse_ntf_train_time(logtime_path):
    txt = logtime_path.read_text(errors="ignore").replace("\r", "\n")
    times = re.findall(r"Training NeuralTranscriptomicField:.*?\[(\d+):(\d+)<", txt)
    if not times:
        return None
    m, s = times[-1]
    return int(m) * 60 + int(s)


def parse_ntf_infer_time(ntf_res, scenario):
    txt = (Path(ntf_res) / scenario / "res.log").read_text(errors="ignore")
    m = re.search(r"Inference time \(seconds\):\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None


def parse_feast_infer_time(log_path):
    txt = log_path.read_text(errors="ignore")
    m = re.search(r"Inference Time:\s*([0-9.]+) seconds", txt)
    return float(m.group(1)) if m else None


def parse_spatialz_infer_time(log_path):
    txt = log_path.read_text(errors="ignore")
    vals = re.findall(
        r"(?:coordinate initialization|Ot optimization|Cell type determination|Gene expression synthesis) time:\s*([0-9.]+) seconds",
        txt)
    return sum(float(v) for v in vals) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_dir", default=".",
                    help="Experiment directory containing monitor/ and logs/")
    ap.add_argument("--ntf_res", default="mytrain/res/deep_starmap_brain",
                    help="NTF result root (per-scenario subdirs with res.log)")
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

    # --- efficiency ---
    eff_rows = []
    for scen in SCENARIOS:
        for method in ["NTF", "FEAST", "SpatialZ"]:
            row = {"method": method, "scenario": scen}
            mon = exp_dir / "monitor" / f"{method}_{scen}.json"
            if method in ("FEAST", "SpatialZ") and scen in ("senario3", "head", "tail"):
                row.update(train_time_s=None, infer_time_s=None, wall_time_s=None,
                           peak_gpu_mem_mib=None, peak_rss_kb=None, status="NA",
                           note="mechanically unable to run extrapolation scenarios")
            elif not mon.exists():
                row.update(status="missing")
            else:
                d = json.loads(mon.read_text())
                row.update(wall_time_s=d["wall_time_s"], peak_gpu_mem_mib=d["peak_gpu_mem_mib"],
                           peak_rss_kb=d["peak_rss_kb"],
                           status="completed" if d["exit_code"] == 0 else f"exit={d['exit_code']}")
                if method == "NTF":
                    row["train_time_s"] = parse_ntf_train_time(exp_dir / "logs" / f"NTF_{scen}.log.time")
                    row["infer_time_s"] = parse_ntf_infer_time(args.ntf_res, scen)
                    row["note"] = "bs=8192, n_iter=20000 (default)"
                elif method == "FEAST":
                    row["train_time_s"] = 0.0
                    row["infer_time_s"] = parse_feast_infer_time(exp_dir / "logs" / f"FEAST_{scen}.log")
                    row["note"] = "no training stage; inference time is the full compute time"
                else:
                    row["train_time_s"] = 0.0
                    row["infer_time_s"] = parse_spatialz_infer_time(exp_dir / "logs" / f"SpatialZ_{scen}.log")
                    row["note"] = "no training stage; inference time is the sum of per-slice phase times"
            eff_rows.append(row)
    eff = pd.DataFrame(eff_rows)[["method", "scenario", "train_time_s", "infer_time_s",
                                  "wall_time_s", "peak_gpu_mem_mib", "peak_rss_kb", "status", "note"]]
    eff.to_csv(exp_dir / "efficiency_summary.csv", index=False)
    print(eff.to_string(index=False))


if __name__ == "__main__":
    main()
