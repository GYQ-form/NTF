#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build svg_benchmark_allscales_summary_with_unist.xlsx.

Merges the existing three-method all-scale SVG benchmark results
(experiment/svg_benchmark_allscales/scale_benchmark_summary.csv) with the new
UniST-fast/UniST-inr rows (svg_scale/unist_svg_scale_metrics.csv) into a new
workbook.  The original xlsx is untouched.  Layout mirrors
run_svg_benchmark_scales.write_workbook: one sheet per method (rows = scale,
columns = metrics, NA explicit), plus Summary (long), Method_Status and
Metadata sheets.  Also writes allmethods_svg_scale_metrics.csv (long table of
all five methods) for plotting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ALLSCALES = Path("/home/gongyuqiao/ur_annotation/NTF/mytrain/revision/experiment/svg_benchmark_allscales")
SVG_SCALE = Path("/home/gongyuqiao/ur_annotation/NTF/mytrain/revision/experiment/r2t5_unist_benchmark/svg_scale")

METRIC_COLUMNS = [
    "Moran_I_corr_3D", "SVG_Jaccard_3D",
    "Moran_I_corr_2D_mean", "Moran_I_corr_2D_std",
    "SVG_Jaccard_2D_mean", "SVG_Jaccard_2D_std",
]
METHODS = ["NTF", "FEAST", "SpatialZ", "UniST-fast", "UniST-inr"]


def clean(value):
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    base = pd.read_csv(ALLSCALES / "scale_benchmark_summary.csv")
    unist = pd.read_csv(SVG_SCALE / "unist_svg_scale_metrics.csv")
    unist = unist[unist["method"] != "NTF_recheck"]
    summary = pd.concat([base, unist], ignore_index=True)
    summary.to_csv(SVG_SCALE / "allmethods_svg_scale_metrics.csv", index=False)

    status_base = pd.read_csv(ALLSCALES / "method_status.csv")
    status_unist = pd.read_csv(SVG_SCALE / "unist_svg_method_status.csv")
    statuses = pd.concat([status_base, status_unist], ignore_index=True)

    with open(ALLSCALES / "benchmark_metadata.json") as handle:
        metadata = json.load(handle)["datasets"]

    frames = {}
    for method in METHODS:
        sub = summary[summary["method"] == method].sort_values("scale_k")
        frame = pd.DataFrame({"scale (*10^3)": sub["scale_k"].tolist()})
        for column in METRIC_COLUMNS:
            frame[column] = [clean(v) for v in sub[column].tolist()]
        frame["n_slices_evaluated"] = [clean(v) for v in sub["n_slices_evaluated"].tolist()]
        if method.startswith("UniST"):
            frame["n_boundary_slices_skipped"] = [clean(v) for v in sub["n_boundary_slices_skipped"].tolist()]
        frame["n_points"] = [clean(v) for v in sub["n_points"].tolist()]
        frames[method] = frame

    summary_cols = ["dataset", "scale_k", "method"] + METRIC_COLUMNS + [
        "n_slices_evaluated", "TopK", "n_genes", "n_points", "status"]
    summary_long = summary[summary_cols].copy()
    for column in METRIC_COLUMNS:
        summary_long[column] = [clean(v) for v in summary_long[column].tolist()]
    metadata_df = pd.DataFrame(metadata)

    output = SVG_SCALE / "svg_benchmark_allscales_summary_with_unist.xlsx"
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_format = workbook.add_format({"font_name": "Arial", "bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "align": "center", "valign": "vcenter"})
        body_format = workbook.add_format({"font_name": "Arial", "font_size": 10, "valign": "vcenter"})
        number_format = workbook.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.000", "valign": "vcenter"})

        sheets = list(frames.items()) + [("Summary", summary_long), ("Method_Status", statuses), ("Metadata", metadata_df)]
        for name, frame in sheets:
            frame = frame.copy()
            for column in frame.columns:
                frame[column] = [clean(v) for v in frame[column].tolist()]
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes(1, 0)
            sheet.autofilter(0, 0, len(frame), len(frame.columns) - 1)
            sheet.set_default_row(18)
            sheet.set_row(0, 22, header_format)
            for idx, column in enumerate(frame.columns):
                values = [str(column)] + [str(value) for value in frame.iloc[:, idx].tolist()]
                width = min(max(max(map(len, values), default=8) + 2, 12), 60)
                fmt = number_format if column in METRIC_COLUMNS else body_format
                sheet.set_column(idx, idx, width, fmt)
            sheet.hide_gridlines(2)
    print(f"[written] {output}")

    # read-back check
    import openpyxl
    wb = openpyxl.load_workbook(output)
    print("[sheets]", wb.sheetnames)
    for method in METHODS:
        ws = wb[method]
        print(method, "rows:", ws.max_row - 1, "cols:", ws.max_column)


if __name__ == "__main__":
    main()
