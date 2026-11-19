#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""UniST per-gene spatial expression plots for the mouseVC 5-gene benchmark.

Mirrors the plotting cells of scripts/mouseVC.ipynb
(``sc.pl.embedding(basis='spatial', color=var_names, mask_obs=...,
layer='prediction', ncols=3, save=...)``) for the UniST results, writing
``spatialUniST-<slice>.pdf`` into ``scripts/figures/<scenario>/`` next to the
existing GT/NTF/FEAST/SZ panels.

Run from the NTF repo root (/home/gongyuqiao/ur_annotation/NTF):
    conda activate ntf
    python mytrain/scripts/plot_unist_genes_mouseVC.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import scanpy as sc

RES_ROOT = Path("mytrain/res/UniST/mouseVC_5gene")
FIG_ROOT = Path("mytrain/scripts/figures")
TEST_SLICES = {"senario1": [2, 4, 6], "senario2": [3, 4, 5]}


def main():
    for scenario, slices in TEST_SLICES.items():
        adata = sc.read_h5ad(RES_ROOT / scenario / "test_results.h5ad")
        figdir = FIG_ROOT / scenario
        figdir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = figdir
        for n in slices:
            sc.pl.embedding(adata, basis="spatial", color=adata.var_names,
                            mask_obs=adata.obs["slice_id"] == f"slice_{n}",
                            ncols=3, layer="prediction", save=f"UniST-{n}.pdf",
                            show=False)
            print(f"[figure] {figdir / f'spatialUniST-{n}.pdf'}")


if __name__ == "__main__":
    main()
