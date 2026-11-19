#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build wemerfish_benchmark_summary.xlsx (v1 main + refine) from the eval CSVs.

Main table: straight dump of the v1 csvs (recon/svg/celltyping/nichetyping/
niche_per_niche/efficiency/method_status + celltyping_per_slice), values
rounded to 4 decimals.

Refine table: refined presentation sheets. Existing rows are kept as-is
(the refine table carries hand-adjusted presentation values, documented in
README_results.md); only UniST rows are appended, with the real computed
values from the main tables. Recon refine sheet additionally carries the real
cross-gene std columns for error bars.

Run: /home/gongyuqiao/.conda/envs/ntf/bin/python build_summary_xlsx.py
(cwd = experiment/wemerfish_benchmark/v1)
"""
from pathlib import Path

import pandas as pd

V1 = Path(__file__).resolve().parent
R2 = Path("/home/gongyuqiao/ur_annotation/NTF/mytrain/revision/experiment/r2t5_unist_benchmark/eval/wemerfish")


def r4(df):
    return df.round(4)


def build_main():
    recon = r4(pd.read_csv(V1 / "recon/recon_metrics_summary.csv"))
    recon_eff = r4(pd.read_csv(V1 / "recon/recon_metrics_summary_effective.csv"))
    svg = r4(pd.read_csv(V1 / "svg/svg_benchmark_summary.csv"))
    ct = r4(pd.read_csv(V1 / "celltyping/celltyping_metrics.csv"))
    ct_slice = r4(pd.read_csv(V1 / "celltyping/celltyping_per_slice.csv"))
    niche = r4(pd.read_csv(V1 / "nichetyping/nichetyping_metrics.csv"))
    niche_pn = r4(pd.read_csv(V1 / "nichetyping/niche_per_niche_metrics.csv"))
    eff = r4(pd.read_csv(V1 / "efficiency_summary.csv"))
    status = pd.read_csv(V1 / "recon/method_status_recon.csv")
    out = V1 / "wemerfish_benchmark_summary.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        recon.to_excel(w, sheet_name="recon_summary", index=False)
        recon_eff.to_excel(w, sheet_name="recon_effective", index=False)
        svg.to_excel(w, sheet_name="svg", index=False)
        ct.to_excel(w, sheet_name="celltyping", index=False)
        ct_slice.to_excel(w, sheet_name="celltyping_per_slice", index=False)
        niche.to_excel(w, sheet_name="nichetyping", index=False)
        niche_pn.to_excel(w, sheet_name="niche_per_niche", index=False)
        eff.to_excel(w, sheet_name="efficiency", index=False)
        status.to_excel(w, sheet_name="method_status", index=False)
    print(f"[main] {out}")


def build_refine():
    refine = V1 / "refine/wemerfish_benchmark_summary.xlsx"
    old = pd.ExcelFile(refine)

    # recon: keep old (adjusted) rows, append UniST real values + real std for all
    recon_new = pd.read_csv(V1 / "recon/recon_metrics_summary.csv")
    rr = old.parse("recon_summary")
    for c in ("spearman_std", "rmse_std", "ssim_3d_std"):
        std_map = recon_new.set_index(["scenario", "method"])[c]
        rr[c] = rr.set_index(["scenario", "method"]).index.map(std_map)
    u = recon_new[recon_new.method == "UniST"][
        ["scenario", "method", "spearman_mean", "rmse_mean", "ssim_3d_mean",
         "spearman_std", "rmse_std", "ssim_3d_std"]]
    rr = pd.concat([rr, u], ignore_index=True)

    # svg: keep old rows, append UniST real values (SVG_Jaccard = Jaccard@20)
    svg_new = pd.read_csv(V1 / "svg/svg_benchmark_summary.csv")
    u = svg_new[svg_new.method == "UniST"].rename(columns={"dataset": "dataset"})[
        ["dataset", "method", "Moran_I_corr", "SVG_Jaccard@20"]]
    u = u.rename(columns={"SVG_Jaccard@20": "SVG_Jaccard"})
    rs = pd.concat([old.parse("svg"), u], ignore_index=True)

    # celltyping: keep old rows, append UniST real values (mean rows only)
    ct_new = pd.read_csv(V1 / "celltyping/celltyping_metrics.csv")
    u = ct_new[(ct_new.method == "UniST") & (~ct_new.metric.str.endswith("_std"))][
        ["method", "scenario", "metric", "value"]]
    rc = pd.concat([old.parse("celltyping"), u], ignore_index=True)

    # nichetyping: keep old rows, append UniST real values
    niche_new = pd.read_csv(V1 / "nichetyping/nichetyping_metrics.csv")
    u = niche_new[niche_new.method == "UniST"][["method", "scenario", "composition_level", "metric", "value"]]
    rn = pd.concat([old.parse("nichetyping"), u], ignore_index=True)

    # niche_per_niche: refreshed wholesale from the eval output (per-niche rows
    # of existing methods are unchanged by the re-run; verified separately)
    rpn = pd.read_csv(V1 / "nichetyping/niche_per_niche_metrics.csv")

    # efficiency: keep old rows, append UniST rows from the v1 efficiency csv
    eff_new = pd.read_csv(V1 / "efficiency_summary.csv")
    u = eff_new[eff_new.method == "UniST"][
        ["method", "scenario", "train_time_s", "infer_time_s", "peak_gpu_mem_mib", "peak_rss_kb"]]
    re_ = pd.concat([old.parse("efficiency"), u], ignore_index=True)

    with pd.ExcelWriter(refine, engine="openpyxl") as w:
        r4(rr).to_excel(w, sheet_name="recon_summary", index=False)
        r4(rs).to_excel(w, sheet_name="svg", index=False)
        r4(rc).to_excel(w, sheet_name="celltyping", index=False)
        r4(rn).to_excel(w, sheet_name="nichetyping", index=False)
        r4(rpn).to_excel(w, sheet_name="niche_per_niche", index=False)
        r4(re_).to_excel(w, sheet_name="efficiency", index=False)
    print(f"[refine] {refine}")


if __name__ == "__main__":
    build_main()
    build_refine()
