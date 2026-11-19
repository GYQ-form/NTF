#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone SVG legend for the weMERFISH cell-typing figures.

Reproduces the exact category -> color mapping used by
eval_celltyping_deepstarmap.py: the sorted category list (GT clusters
categories plus "NA" from invalid cells) mapped onto tab20 + tab20b + tab20c
(60 colors); the legend itself covers the GT categories only (non-digit,
non-"NA"), so colors match gt_all_slices.png / spatial_maps_*.png.

Run (cwd = /home/gongyuqiao/ur_annotation/NTF):
    python mytrain/scripts/plot_celltype_legend_wemerfish.py \
        --output <exp>/celltyping/figures/legend_gt_labels.svg
"""
from __future__ import annotations

import argparse
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_data",
                    default="mytrain/data/wemerfish_zebrafish_embryo/processed/wemerfish_C6s_E2.h5ad")
    ap.add_argument("--gt_key", default="clusters")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ncol", type=int, default=3)
    args = ap.parse_args()

    adata = sc.read_h5ad(args.input_data)
    # GT categories sorted get tab20/tab20b/tab20c colors in order — identical
    # to the spatial-map figures (GT-first palette) and the 3D viewer
    cats = sorted(set(adata.obs[args.gt_key].astype(str)))
    colors = (list(plt.get_cmap("tab20").colors)
              + list(plt.get_cmap("tab20b").colors)
              + list(plt.get_cmap("tab20c").colors))
    palette = {c: colors[i % len(colors)] for i, c in enumerate(cats)}
    legend_cats = [c for c in cats if not c.isdigit() and c != "NA"]

    n_rows = int(np.ceil(len(legend_cats) / args.ncol))
    fig = plt.figure(figsize=(3.2 * args.ncol, 0.3 + n_rows * 0.24 + 0.15))
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=4, color=palette[c])
               for c in legend_cats]
    wrapped = [textwrap.fill(str(c), 30) for c in legend_cats]
    fig.legend(handles, wrapped, loc="center", fontsize=8, frameon=False,
               ncol=args.ncol, title="GT label", title_fontsize=9,
               columnspacing=0.8, handletextpad=0.3)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"[saved] {args.output} ({len(legend_cats)} categories, {args.ncol} columns)")


if __name__ == "__main__":
    main()
