#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R1 Major7 ablation (refine) — one bar-chart figure per dataset.

Reads refine/ablation_summary.xlsx (sheets: simulation, mouseVC) and writes one
figure per dataset (ggplot theme_classic style via plotnine):
  refine_ablation_simulation.svg / refine_ablation_mousevc.svg (+ png previews)
X axis: ablation configuration; facets: evaluation metric (Spearman, RMSE,
3D SSIM).
"""
import os

import pandas as pd
from plotnine import (aes, element_blank, element_text, facet_wrap, geom_bar,
                      ggplot, labs, scale_fill_manual, theme, theme_classic)

BASE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(BASE, "ablation_summary.xlsx")

CONFIGS = ["M0_full", "M1_no_dropout", "M2_no_psf", "M3_no_smooth",
           "M4_no_sliceemb", "M5_no_uncertainty", "M6_bias"]
LABEL = {
    "M0_full": "Full",
    "M1_no_dropout": "-dropout",
    "M2_no_psf": "-PSF",
    "M3_no_smooth": "-smooth",
    "M4_no_sliceemb": "-sliceEmb",
    "M5_no_uncertainty": "-uncertainty",
    "M6_bias": "-bias",
}
SHEETS = {"simulation": ("Simulation", "#4883B1"),
          "mouseVC": ("mouseVC (real)", "#C05746")}
METRICS = [("spearman_corr_mean", "Spearman"),
           ("rmse_mean", "RMSE"),
           ("ssim_3d_mean", "3D SSIM")]


def plot_sheet(sheet, group, color):
    df = pd.read_excel(XLSX, sheet_name=sheet)
    # unify the bias-correction ablation config name (M6_no_bias on both sheets)
    df["config"] = df["config"].replace({"M6_no_bias": "M6_bias",
                                         "M6_bias_on": "M6_bias"})
    df["config"] = pd.Categorical(df["config"], categories=CONFIGS, ordered=True)
    df["config_label"] = pd.Categorical(df["config"].map(LABEL),
                                        categories=[LABEL[c] for c in CONFIGS],
                                        ordered=True)

    long = df.melt(id_vars=["config_label"],
                   value_vars=[m for m, _ in METRICS],
                   var_name="metric", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["metric"] = pd.Categorical(long["metric"].map(dict(METRICS)),
                                    categories=[n for _, n in METRICS],
                                    ordered=True)

    p = (ggplot(long, aes("config_label", "value"))
         + geom_bar(stat="identity", width=0.7, fill=color, color="black",
                    size=0.25)
         + facet_wrap("metric", scales="free_y", ncol=3)
         + labs(title=f"NTF component ablation — {group}")
         + theme_classic()
         + theme(axis_title=element_blank(),
                 axis_text_x=element_text(size=8, angle=30, ha="right"),
                 axis_text_y=element_text(size=8),
                 strip_text=element_text(size=10),
                 plot_title=element_text(size=11),
                 figure_size=(8, 3)))
    for ext in ("svg", "png"):
        out = os.path.join(BASE, f"refine_ablation_{sheet}.{ext}")
        p.save(out, verbose=False, dpi=300)
        print("saved", out)


def main():
    for sheet, (group, color) in SHEETS.items():
        plot_sheet(sheet, group, color)


if __name__ == "__main__":
    main()
