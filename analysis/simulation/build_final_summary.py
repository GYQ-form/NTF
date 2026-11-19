"""Assemble final benchmark_long.csv + r2t5_unist_benchmark_summary.xlsx.

Sources (all final, centralized-protocol where applicable):
- scale: scale_benchmark_summary_4methods.csv (central, vs truth)
- interval: eval/interval/interval_4methods_summary.csv (per-method official csvs)
- gene_count: gene_count_benchmark/summary_by_genes.csv (NTF/FEAST/SpatialZ, central_eval
  protocol of that experiment) + eval/gene_count/gene_count_unist_summary.csv (UniST, vs truth)
- mouseVC: eval/mousevc/mousevc_4methods_summary.csv (vs X, real data)
- Deep-STARmap: eval/deepstarmap/recon/recon_metrics_summary.csv + celltyping
- weMERFISH: eval/wemerfish/recon + celltyping + nichetyping
- CyCIF: cycif v2 recon_metrics_summary.csv (NTF) + eval/cycif/cycif_unist_summary.csv (UniST-fast)
"""
import pandas as pd
import numpy as np

EXP = "/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/r2t5_unist_benchmark"
GC = "/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/gene_count_benchmark"
CY = "/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/cycif_crc1_benchmark/v2"

# ---------- scale ----------
scale = pd.read_csv(f"{EXP}/scale_benchmark_summary_4methods.csv")
scale = scale.rename(columns={"scale": "scenario"})
# SVG conservation columns (2026-10-14): three-method all-scale SVG benchmark +
# UniST patch (same paired protocol); NA explicit where a method-scale was not run.
SVG_COLS = ["Moran_I_corr_3D", "SVG_Jaccard_3D", "Moran_I_corr_2D_mean",
            "Moran_I_corr_2D_std", "SVG_Jaccard_2D_mean", "SVG_Jaccard_2D_std"]
_svg3 = pd.read_csv(f"{EXP}/../svg_benchmark_allscales/scale_benchmark_summary.csv")
_svgu = pd.read_csv(f"{EXP}/svg_scale/unist_svg_scale_metrics.csv")
_svgu = _svgu[_svgu["method"] != "NTF_recheck"]
_svg = pd.concat([_svg3, _svgu], ignore_index=True)
_svg["scenario"] = _svg["dataset"].str.replace("scale_", "", regex=False)
_svg = _svg[["scenario", "method"] + SVG_COLS]
scale["dataset"] = "scale_test"
scale = scale.merge(_svg, on=["scenario", "method"], how="left")
scale[SVG_COLS] = scale[SVG_COLS].where(scale[SVG_COLS].notna(), "NA")

# ---------- interval ----------
itv = pd.read_csv(f"{EXP}/eval/interval/interval_4methods_summary.csv")
itv = itv.rename(columns={"k": "scenario"})
itv["scenario"] = "k" + itv["scenario"].astype(str)
itv["dataset"] = "interval_100slice"
# n_slices_used = number of input (train) slices, same convention as NTF's
# interval_study_results.csv: slices 0..99 with slice_id % k == 0 -> 99//k + 1.
itv["n_slices_used"] = 99 // itv["scenario"].str[1:].astype(int) + 1

# ---------- gene_count ----------
gc3 = pd.read_csv(f"{GC}/summary_by_genes.csv")
gc3 = gc3[["method", "n_genes", "pearson_mean", "spearman_mean", "rmse_mean", "ssim_3d_mean"]]
gcu = pd.read_csv(f"{EXP}/eval/gene_count/gene_count_unist_summary.csv")
gcu["method"] = "UniST-" + gcu["mode"]
gcu = gcu[["method", "n_genes", "pearson_mean", "spearman_mean", "rmse_mean"]]
gcall = pd.concat([gc3, gcu], ignore_index=True)
gcall = gcall.rename(columns={"n_genes": "scenario"})
gcall["scenario"] = gcall["scenario"].astype(str) + "genes"
gcall["dataset"] = "gene_count_1M"

# ---------- mouseVC ----------
mv = pd.read_csv(f"{EXP}/eval/mousevc/mousevc_4methods_summary.csv")
mv["dataset"] = "mouseVC_5gene"

# ---------- deepstarmap ----------
dsr = pd.read_csv(f"{EXP}/eval/deepstarmap/recon/recon_metrics_summary.csv")
dsr = dsr[["scenario", "method", "pearson_mean", "spearman_mean", "rmse_mean", "ssim_3d_mean"]]
dsr["dataset"] = "deep_starmap_v3"
dsc = pd.read_csv(f"{EXP}/eval/deepstarmap/celltyping/celltyping_metrics.csv")

# ---------- wemerfish ----------
wmr = pd.read_csv(f"{EXP}/eval/wemerfish/recon/recon_metrics_summary.csv")
wmr = wmr[["scenario", "method", "pearson_mean", "spearman_mean", "rmse_mean", "ssim_3d_mean"]]
wmr["dataset"] = "wemerfish"
wmc = pd.read_csv(f"{EXP}/eval/wemerfish/celltyping/celltyping_metrics.csv")
wmn = pd.read_csv(f"{EXP}/eval/wemerfish/nichetyping/nichetyping_metrics.csv")

# ---------- cycif ----------
cyn = pd.read_csv(f"{CY}/recon/recon_metrics_summary.csv")
cyn = cyn[["scenario", "method", "pearson_mean", "spearman_mean", "rmse_mean", "ssim_3d_mean"]]
cyu = pd.read_csv(f"{EXP}/eval/cycif/cycif_unist_summary.csv")
cyu["ssim_3d_mean"] = np.nan
cyall = pd.concat([cyn, cyu], ignore_index=True)
cyall["dataset"] = "cycif_crc1_aligned"

# ---------- efficiency & status ----------
eff = pd.read_csv(f"{EXP}/efficiency_summary.csv")
status = pd.read_csv(f"{EXP}/method_status.csv")

# ---------- benchmark_long ----------
long_frames = []
for df, dsname in [(scale, "scale_test"), (itv, "interval_100slice"), (gcall, "gene_count_1M"),
                   (mv, "mouseVC_5gene"), (dsr, "deep_starmap_v3"), (wmr, "wemerfish"),
                   (cyall, "cycif_crc1_aligned")]:
    metric_cols = [c for c in df.columns if any(
        c.startswith(m) for m in ("pearson", "spearman", "rmse", "ssim", "avg_"))]
    sub = df.melt(id_vars=["dataset", "scenario", "method"], value_vars=metric_cols,
                  var_name="metric", value_name="value")
    long_frames.append(sub)
blong = pd.concat(long_frames, ignore_index=True)
blong.to_csv(f"{EXP}/benchmark_long.csv", index=False)

with pd.ExcelWriter(f"{EXP}/r2t5_unist_benchmark_summary.xlsx", engine="openpyxl") as w:
    scale.to_excel(w, sheet_name="scale_4methods", index=False)
    itv.to_excel(w, sheet_name="interval_4methods", index=False)
    gcall.to_excel(w, sheet_name="gene_count", index=False)
    mv.to_excel(w, sheet_name="mouseVC", index=False)
    dsr.to_excel(w, sheet_name="deepstarmap_recon", index=False)
    dsc.to_excel(w, sheet_name="deepstarmap_celltyping", index=False)
    wmr.to_excel(w, sheet_name="wemerfish_recon", index=False)
    wmc.to_excel(w, sheet_name="wemerfish_celltyping", index=False)
    wmn.to_excel(w, sheet_name="wemerfish_nichetyping", index=False)
    cyall.to_excel(w, sheet_name="cycif", index=False)
    eff.to_excel(w, sheet_name="efficiency", index=False)
    status.to_excel(w, sheet_name="method_status", index=False)
print("xlsx written:", f"{EXP}/r2t5_unist_benchmark_summary.xlsx")
print("benchmark_long rows:", len(blong))
