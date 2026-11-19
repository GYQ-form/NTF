#!/usr/bin/env python
"""
Validation of the model-side in-silico querying feature (NTF/query.py),
added for the response to Reviewer #2 Major Comment 2.1.

Idea: a trained NTF model needs no pre-estimated coordinates in unmeasured
gaps. The user supplies dense query spots at any desired resolution (the
in-silico analogue of running an assay), and the model itself decides tissue
occupancy from the predicted total expression and dropout probability.

This script (smoke test on STARmap mouse visual cortex, 28 genes):
  1. trains a small NTF model (short run, dropout network enabled);
  2. builds a dense query grid over the whole bounding box (no tissue mask);
  3. runs in_silico_query with automatic occupancy filtering;
  4. checks that occupied spots fall on real tissue: distance of each occupied
     spot to the nearest observed cell should be small, while filtered-out
     spots should mostly lie outside the tissue outline.

Usage (conda env ntf, GPU 6/7 per revision/AGENTS.md):
    CUDA_VISIBLE_DEVICES=6 python analysis/STARmap_mouse_VC/test_in_silico_query.py \
        --output_dir mytrain/revision/experiment/r2m1_insilico_query
"""
import argparse
import logging
import os
import time

import anndata
import numpy as np
import torch
from scipy.spatial import cKDTree

from NTF.config import add_shared_args, process_args
from NTF.query import make_query_grid, in_silico_query
from NTF.train import train

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DATA = '/home/gongyuqiao/ur_annotation/NTF/mytrain/data/mouse_VC/processed/mouseVC_28gene.h5ad'


def get_args():
    shared_parser = add_shared_args()
    parser = argparse.ArgumentParser(parents=[shared_parser])
    # relax required flags so the smoke test can run with built-in defaults
    for action in parser._actions:
        if action.dest in ('input_data', 'slice_id'):
            action.required = False
    parser.set_defaults(input_data=INPUT_DATA, slice_id='slice_id', n_iter=2000)
    args, _ = parser.parse_known_args()
    args.input_data = INPUT_DATA
    args.slice_id = 'slice_id'
    args.n_iter = 2000
    args = process_args(args)
    return args


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    adata = anndata.read_h5ad(args.input_data)
    adata.obsm['spatial'] = np.stack(
        [adata.obs['x'].values, adata.obs['y'].values, adata.obs['z'].values], axis=1
    ).astype(np.float32)
    # STARmap VC is a continuous volume; assign slice ids by z-binning for the slice embedding
    z = adata.obs['z'].values
    adata.obs['slice_id'] = pd_qcut_z(z, n_slices=20)

    raw_coords = adata.obsm['spatial'].copy()
    raw_bbox_size = (raw_coords.max(0) - raw_coords.min(0)).max()
    spatial_scaling = 1.0 / raw_bbox_size
    adata.obsm['spatial'] *= spatial_scaling

    logging.info('Training smoke-test NTF model (dropout network enabled)...')
    trained_NTF = train(adata, args)
    trained_inr = trained_NTF.inr

    # Dense query grid over the full bounding box (no geometric tissue mask),
    # spacing = mean nearest-neighbor distance of the observed cells
    nn_dist = cKDTree(raw_coords).query(raw_coords, k=2)[0][:, 1].mean()
    grid = make_query_grid(raw_coords, resolution=nn_dist, mask=None)
    logging.info(f'Query grid: {len(grid)} spots (spacing {nn_dist:.2f} = mean NN distance).')

    t0 = time.time()
    query_adata = in_silico_query(
        trained_inr, grid,
        spatial_scaling=spatial_scaling,
        occupancy=True,
        min_total_expr='auto',
        var_names=adata.var_names,
        device=args.device,
    )
    logging.info(f'In-silico query + occupancy filtering took {time.time() - t0:.2f} s.')

    # --- Validation: occupied spots should fall on real tissue ---
    tree = cKDTree(raw_coords)
    dist_occupied = tree.query(query_adata.obsm['spatial'])[0]
    keep_mask = query_adata.uns['occupancy_info']
    n_query = keep_mask['n_query_spots']
    n_occ = keep_mask['n_occupied_spots']

    # distance distribution of ALL grid spots for contrast
    all_results = in_silico_query(
        trained_inr, grid, spatial_scaling=spatial_scaling,
        occupancy=False, min_total_expr='auto', device=args.device,
    )
    is_occ = all_results.obs['is_occupied'].values
    dist_all = tree.query(grid)[0]
    tol = nn_dist * 2.0

    frac_occupied_on_tissue = float((dist_occupied < tol).mean())
    frac_filtered_off_tissue = float((dist_all[~is_occ] >= tol).mean()) if (~is_occ).any() else float('nan')
    mean_dropout_in = float(all_results.obs['mean_dropout_prob'].values[is_occ].mean()) if 'mean_dropout_prob' in all_results.obs else float('nan')
    mean_dropout_out = float(all_results.obs['mean_dropout_prob'].values[~is_occ].mean()) if 'mean_dropout_prob' in all_results.obs else float('nan')

    summary = {
        'n_query_spots': int(n_query),
        'n_occupied_spots': int(n_occ),
        'occupied_fraction': n_occ / n_query,
        'frac_occupied_on_tissue(dist<2xNN)': frac_occupied_on_tissue,
        'frac_filtered_off_tissue(dist>=2xNN)': frac_filtered_off_tissue,
        'mean_dropout_prob_occupied': mean_dropout_in,
        'mean_dropout_prob_filtered': mean_dropout_out,
        'min_total_expr_threshold': keep_mask['min_total_expr'],
    }
    for k, v in summary.items():
        logging.info(f'{k}: {v}')

    assert frac_occupied_on_tissue > 0.8, 'occupied spots do not fall on tissue'
    assert mean_dropout_in < mean_dropout_out, 'dropout confidence does not separate tissue from background'

    query_adata.write_h5ad(os.path.join(args.output_dir, 'insilico_query_occupied.h5ad'), compression='gzip')
    all_results.write_h5ad(os.path.join(args.output_dir, 'insilico_query_all_spots.h5ad'), compression='gzip')

    import json
    with open(os.path.join(args.output_dir, 'validation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f'Validation passed. Outputs saved to {args.output_dir}')


def pd_qcut_z(z, n_slices):
    import pandas as pd
    return pd.qcut(pd.Series(z), q=n_slices, labels=False, duplicates='drop').astype(str).values


if __name__ == '__main__':
    main()
