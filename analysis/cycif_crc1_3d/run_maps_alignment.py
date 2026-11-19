"""Rerun MAPS sequential rigid alignment for cycif_crc1 slices.

Reproduces alignment.ipynb (seed 2026, mode='sequential', epochs=1000,
sample_size=20000, enable_scale=False) on the per-slice exports and saves
aligned 2D coordinates (µm) per slice.
"""
import os
import re
import sys

import anndata as ad
import numpy as np
import torch

from MAPS.align import Multi_slices_rigid_alignment
from MAPS.utils import set_seed

DATA_DIR = "/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/processed/alignment/extracted"
OUT_DIR = "/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/processed/alignment/aligned"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
set_seed(2026)


def slice_num(fname):
    return int(re.search(r"(\d+)", fname).group(1))


files = sorted((f for f in os.listdir(DATA_DIR) if f.endswith(".h5ad")), key=slice_num)
adata_list = []
for f in files:
    a = ad.read_h5ad(os.path.join(DATA_DIR, f))
    print(f"Reading {f} ...", flush=True)
    adata_list.append(a)

aligned_list = Multi_slices_rigid_alignment(
    adata_list, mode="sequential", epochs=1000, sample_size=20000,
    enable_scale=False, device=device,
)

for f, a in zip(files, aligned_list):
    sid = slice_num(f)
    out = os.path.join(OUT_DIR, f"slice{sid}_aligned.npy")
    np.save(out, np.asarray(a.obsm["spatial"], dtype=np.float32))
    print(f"saved {out} shape={a.obsm['spatial'].shape}", flush=True)

print("DONE")
