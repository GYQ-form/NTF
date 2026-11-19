#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone SVG legend for the Deep-STARmap cell-typing figures.

Reproduces the exact category -> color mapping used by
eval_celltyping_deepstarmap.py: the sorted category list (GT Harmony_labels
categories plus "NA" from invalid cells) mapped onto tab20 + tab20b + tab20c
(60 colors); the legend itself covers the GT categories only (non-digit,
non-"NA"), so colors match gt_all_slices.png / spatial_maps_*.png.
"""
from __future__ import annotations

import argparse
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_data",
                    default="mytrain/data/deep_starmap_brain/processed/deep_starmap_brain_v3.h5ad")
    ap.add_argument("--gt_key", default="Harmony_labels")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ncol", type=int, default=3)
    args = ap.parse_args()

    adata = sc.read_h5ad(args.input_data)
    cats = set(adata.obs[args.gt_key].astype(str))
    cats.add("NA")  # invalid-cell entries are part of the palette category list
    cats = sorted(cats)
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
