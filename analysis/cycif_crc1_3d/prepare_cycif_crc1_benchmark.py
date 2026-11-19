"""Prepare the benchmark h5ad for the t-CyCIF CRC1 3D atlas.

Input:  mytrain/data/cycif_crc1_3d/processed/cycif_crc1.h5ad (annotated)
Output: mytrain/data/cycif_crc1_3d/processed/cycif_crc1_bm.h5ad

- Full dataset, NO downsampling (12,758,141 cells x 24 markers).
- X = sc.pp.normalize_total(1e4) + sc.pp.log1p (CSR sparse); layers['counts']
  keeps raw intensities; obsm['spatial'] = spatial_um (all axes in um).
- obs: slice_id (1-25, ordered by z), z (um), cell_type, cell_type_main.
- Five split columns (per-slice pure train/test, asserted):
    group   senario1 interleaved interpolation: test sections 14,39,50,64,78,97
    group2  senario2 middle band interpolation: test sections 49,50,51,52,54
    group4  senario3 two-side extrapolation:    test sections 2,7,102,106
    head    head extrapolation:                 test sections 2,7
    tail    tail extrapolation:                 test sections 102,106

Usage:
    conda activate ntf
    python mytrain/scripts/prepare_cycif_crc1_benchmark.py                 # v1: unregistered coords
    python mytrain/scripts/prepare_cycif_crc1_benchmark.py \
        --coord-key spatial_aligned_um \
        --out-path .../cycif_crc1_bm_aligned.h5ad --qc-dir .../v2          # v2: MAPS-aligned coords
"""

import argparse
import os

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

IN_PATH = '/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/processed/cycif_crc1.h5ad'
OUT_PATH = '/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/processed/cycif_crc1_bm.h5ad'
QC_DIR = '/home/gongyuqiao/ur_annotation/NTF/mytrain/revision/experiment/cycif_crc1_benchmark'

SECTIONS = [2, 7, 14, 20, 25, 29, 34, 39, 44, 49, 50, 51, 52, 54, 59, 64, 69,
            74, 78, 84, 86, 91, 97, 102, 106]  # ascending z

TEST_SECTIONS = {
    'group': {14, 39, 50, 64, 78, 97},   # senario1: every 4th slice
    'group2': {49, 50, 51, 52, 54},      # senario2: middle band z 245-270 um
    'group4': {2, 7, 102, 106},          # senario3: both ends
    'head': {2, 7},
    'tail': {102, 106},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-path', default=IN_PATH)
    ap.add_argument('--out-path', default=OUT_PATH)
    ap.add_argument('--qc-dir', default=QC_DIR)
    ap.add_argument('--coord-key', default='spatial_um',
                    help="obsm key (3 columns, all axes in um) used as obsm['spatial']; "
                         "'spatial_um' = unregistered (v1), 'spatial_aligned_um' = MAPS-registered (v2)")
    args = ap.parse_args()
    in_path, out_path, qc_dir = args.in_path, args.out_path, args.qc_dir

    os.makedirs(qc_dir, exist_ok=True)
    print(f'Reading {in_path} ...')
    adata = anndata.read_h5ad(in_path)
    print(f'{adata.n_obs:,} cells x {adata.n_vars} markers')

    # expression convention: X = normalize_total(1e4) + log1p (CSR), counts = raw
    X_raw = np.asarray(adata.X, dtype=np.float32)
    adata.layers['counts'] = sp.csr_matrix(X_raw)
    adata.X = sp.csr_matrix(X_raw)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # coordinates: all axes in um, from the requested obsm key
    adata.obsm['spatial'] = np.asarray(adata.obsm[args.coord_key], dtype=np.float32)
    adata.uns['bm_coord_key'] = args.coord_key
    for k in ('spatial_um', 'spatial_aligned_um'):
        adata.obsm.pop(k, None)

    # slice_id ordered by z
    sec2sid = {s: i + 1 for i, s in enumerate(SECTIONS)}
    adata.obs['slice_id'] = adata.obs['slice_id'].map(sec2sid).astype(int)
    adata.obs['z'] = np.asarray(adata.obsm['spatial'][:, 2], dtype=np.float32)

    # split columns (slice_id is already the new 1-25 id here)
    for col, test_secs in TEST_SECTIONS.items():
        test_ids = {sec2sid[s] for s in test_secs}
        adata.obs[col] = np.where(adata.obs['slice_id'].isin(test_ids), 'test', 'train')

    # QC: assert pure train/test per slice, and test slices match the design
    for col, test_secs in TEST_SECTIONS.items():
        tab = adata.obs.groupby('slice_id')[col].nunique()
        assert (tab == 1).all(), f'{col}: mixed train/test within a slice'
        got = set(adata.obs.loc[adata.obs[col] == 'test', 'slice_id'].unique())
        want = {sec2sid[s] for s in test_secs}
        assert got == want, f'{col}: test slices {sorted(got)} != design {sorted(want)}'
    zf = pd.DataFrame({
        'marker': adata.var_names,
        'zero_frac': np.asarray((adata.layers['counts'] == 0).mean(axis=0)).flatten(),
    })
    zf.to_csv(os.path.join(qc_dir, 'gene_zero_frac.csv'), index=False)

    split_stats = []
    for col in TEST_SECTIONS:
        n_test = int((adata.obs[col] == 'test').sum())
        split_stats.append({'split_column': col, 'n_test': n_test,
                            'n_train': int((adata.obs[col] == 'train').sum()),
                            'test_sections': ';'.join(map(str, sorted(TEST_SECTIONS[col])))})
    pd.DataFrame(split_stats).to_csv(os.path.join(qc_dir, 'split_stats.csv'), index=False)
    print(pd.DataFrame(split_stats).to_string(index=False))

    adata.write_h5ad(out_path, compression='gzip')
    print(f'Wrote {out_path}: {adata.n_obs:,} cells x {adata.n_vars} markers (coord: {args.coord_key})')


if __name__ == '__main__':
    main()
