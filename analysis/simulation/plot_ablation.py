#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R1 Major7 ablation — figures (ggplot / theme_classic style via plotnine, SVG output).

Reads results/sim_summary.csv, results/mousevc_summary.csv, results/efficiency_summary.csv
(produced by summarize_ablation.py) and writes SVG figures to figures/.
"""
import os

import pandas as pd
from plotnine import (aes, element_text, geom_bar, geom_hline, ggplot, labs,
                      position_dodge, facet_wrap, theme, theme_classic,
                      scale_fill_manual, coord_flip)

BASE = "/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/r1m7_model_ablation"
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

CONFIGS = ["M0_full", "M1_no_dropout", "M2_no_psf", "M3_no_smooth", "M4_no_sliceemb",
           "M5_no_pixvar", "M6_no_slicevar", "M7_no_uncertainty", "M8_bare", "M9_bias_on"]
LABEL = {
    "M0_full": "Full",
    "M1_no_dropout": "-dropout",
    "M2_no_psf": "-PSF",
    "M3_no_smooth": "-smooth",
    "M4_no_sliceemb": "-sliceEmb",
    "M5_no_pixvar": "-pixVar",
    "M6_no_slicevar": "-sliceVar",
    "M7_no_uncertainty": "-uncert",
    "M8_bare": "bare",
    "M9_bias_on": "+bias",
}
FILL = {"Full": "#82659C", "ablation": "#4883B1", "bare": "#C05746", "bias_on": "#5FA24D"}


def prep(df):
    df = df.copy()
    df["config"] = pd.Categorical(df["config"], categories=CONFIGS, ordered=True)
    df["config_label"] = df["config"].map(LABEL)
    df["config_label"] = pd.Categorical(df["config_label"],
                                        categories=[LABEL[c] for c in CONFIGS], ordered=True)
    def kind(c):
        if c == "M0_full":
            return "Full"
        if c == "M8_bare":
            return "bare"
        if c == "M9_bias_on":
            return "bias_on"
        return "ablation"
    df["kind"] = df["config"].map(kind)
    return df


METRICS = [("pearson_corr_mean", "Pearson"),
           ("spearman_corr_mean", "Spearman"),
           ("rmse_mean", "RMSE"),
           ("ssim_3d_mean", "3D SSIM")]


def metric_bars(df, title, fname, extra_facet=None):
    long = df.melt(id_vars=["config_label", "kind"] + ([extra_facet] if extra_facet else []),
                   value_vars=[m for m, _ in METRICS if m in df.columns],
                   var_name="metric", value_name="value")
    long["metric"] = pd.Categorical(long["metric"].map(dict(METRICS)),
                                    categories=[n for _, n in METRICS], ordered=True)
    base = (ggplot(long, aes("config_label", "value", fill="kind"))
            + geom_bar(stat="identity", position=position_dodge(width=0.8), width=0.7)
            + scale_fill_manual(values=FILL, guide=None)
            + coord_flip()
            + labs(title=title, x=None, y="value")
            + theme_classic()
            + theme(axis_text=element_text(size=8), axis_title=element_text(size=9),
                    plot_title=element_text(size=11), strip_text=element_text(size=9),
                    figure_size=(10, 6 if extra_facet is None else 7)))
    if extra_facet is None:
        p = base + facet_wrap("metric", scales="free_y", ncol=4)
    else:
        from plotnine import facet_grid
        p = base + facet_grid(f"{extra_facet} ~ metric", scales="free_y")
    p.save(os.path.join(FIG, fname), verbose=False)
    print("saved", fname)


def main():
    sim = prep(pd.read_csv(os.path.join(RES, "sim_summary.csv")))
    metric_bars(sim, "Ablation on 1M-cell simulation (100 genes), mean over genes, vs ground truth",
                "sim_ablation_metrics.svg")

    vc = prep(pd.read_csv(os.path.join(RES, "mousevc_summary.csv")))
    vc = vc[vc["config"] != "M0_full"]  # NaN check only, no valid metrics
    metric_bars(vc, "Ablation on STARmap mouse VC 5-gene (baseline: -dropout), vs observed X",
                "mousevc_ablation_metrics.svg", extra_facet="scenario")

    eff = pd.read_csv(os.path.join(RES, "efficiency_summary.csv"))
    eff_sim = prep(eff[eff["dataset"] == "simulation_1M_100gene"])
    long = eff_sim.melt(id_vars=["config_label", "kind"],
                        value_vars=["wall_time_s", "peak_gpu_mem_mib"],
                        var_name="metric", value_name="value")
    long["metric"] = long["metric"].map({"wall_time_s": "Wall time (s)",
                                         "peak_gpu_mem_mib": "Peak GPU mem (MiB)"})
    p = (ggplot(long, aes("config_label", "value", fill="kind"))
         + geom_bar(stat="identity", width=0.7)
         + facet_wrap("metric", scales="free_y", ncol=2)
         + scale_fill_manual(values=FILL, guide=None)
         + coord_flip()
         + labs(title="Efficiency cost of components (1M-cell simulation)", x=None, y=None)
         + theme_classic()
         + theme(axis_text=element_text(size=8), strip_text=element_text(size=9),
                 plot_title=element_text(size=11), figure_size=(8, 6)))
    p.save(os.path.join(FIG, "sim_ablation_efficiency.svg"), verbose=False)
    print("saved sim_ablation_efficiency.svg")


if __name__ == "__main__":
    main()
