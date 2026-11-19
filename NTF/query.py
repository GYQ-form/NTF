#!/usr/bin/env python
"""
# Author: Yuqiao Gong
# File Name: query.py
# Description: In-silico querying utilities for a trained NTF model.
#
# A trained NTF model does not require pre-estimated cell coordinates in
# unmeasured regions ("gaps"). Instead, the user provides dense query spots
# (or bins) at any desired resolution -- the in-silico analogue of running a
# sequencing assay -- and the model itself decides, from the predicted total
# expression (and, when the dropout network is enabled, the predicted dropout
# probability), whether a query point corresponds to real tissue. Points with
# low confidence are filtered out or flagged.
"""
import logging
from typing import Optional, Union

import anndata
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from .sample import sample_points


def make_query_grid(
    reference_coords: np.ndarray,
    resolution: Union[float, np.ndarray],
    mask: Optional[str] = None,
    dist_tol_multiplier: float = 2.0,
) -> np.ndarray:
    """
    Generate a dense regular grid of query spots covering the reference tissue.

    Parameters:
    - reference_coords: (N, 3) observed coordinates, in raw physical units.
    - resolution: grid spacing in the same units (scalar or per-axis array of 3).
    - mask: None to query the full bounding box, or 'kdtree' to restrict the grid
      to the tissue outline (a point is kept if its distance to the nearest
      observed cell is below dist_tol_multiplier x mean nearest-neighbor distance).
    - dist_tol_multiplier: tolerance multiplier for mask='kdtree'.

    Returns:
    - (M, 3) float32 array of query coordinates (raw units).
    """
    reference_coords = np.asarray(reference_coords, dtype=np.float64)
    spacing = np.broadcast_to(np.asarray(resolution, dtype=np.float64), (3,))

    min_bound = reference_coords.min(axis=0)
    max_bound = reference_coords.max(axis=0)
    axes = [np.arange(lo, hi + 0.5 * s, s) for lo, hi, s in zip(min_bound, max_bound, spacing)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    logging.info(f"Query grid: {len(grid)} spots at spacing {spacing} (bbox {min_bound} ~ {max_bound}).")

    if mask is None:
        return grid.astype(np.float32)
    if mask != 'kdtree':
        raise ValueError(f"Unknown mask mode: {mask}")

    kdtree = cKDTree(reference_coords)
    nn_dist, _ = kdtree.query(reference_coords, k=2)
    threshold = nn_dist[:, 1].mean() * dist_tol_multiplier
    keep = kdtree.query(grid)[0] < threshold
    logging.info(
        f"KD-tree tissue mask (threshold {threshold:.4f}): "
        f"kept {int(keep.sum())}/{len(grid)} grid spots."
    )
    return grid[keep].astype(np.float32)


def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu threshold on 1D values; returns 0 when the distribution is degenerate."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.min() == values.max():
        return 0.0
    hist, edges = np.histogram(values, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight = hist / hist.sum()
    omega = np.cumsum(weight)
    mu = np.cumsum(weight * centers)
    mu_total = mu[-1]
    sigma2 = (mu_total * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return float(centers[int(np.nanargmax(sigma2))])


def in_silico_query(
    model,
    query_coords: np.ndarray,
    spatial_scaling: float = 1.0,
    batch_size: int = 8192,
    occupancy: bool = True,
    min_total_expr: Union[str, float] = 'auto',
    max_mean_dropout: Optional[float] = None,
    var_names=None,
    device: Optional[str] = None,
) -> anndata.AnnData:
    """
    Query a trained NTF model at arbitrary coordinates ("in-silico sequencing")
    and optionally decide tissue occupancy per query spot.

    Parameters:
    - model: trained GeneINR (e.g. trained_NTF.inr).
    - query_coords: (M, 3) raw-unit coordinates, e.g. from make_query_grid.
    - spatial_scaling: the scaling factor applied to the training coordinates
      (query coords are multiplied by the same factor before inference).
    - batch_size: inference batch size.
    - occupancy: if True, compute per-spot confidence metrics and keep only
      occupied spots; if False, return all spots with metrics attached.
    - min_total_expr: minimum predicted total expression for a spot to count as
      tissue; 'auto' uses an Otsu threshold on log1p(total expression).
    - max_mean_dropout: if set (and the dropout network is enabled), spots whose
      mean predicted dropout probability exceeds this value are discarded as
      low-confidence.
    - var_names: gene names for the output AnnData (defaults to gene_0, ...).
    - device: inference device (defaults to model.args.device).

    Returns:
    - AnnData with X = predicted expression, obsm['spatial'] = raw-unit
      coordinates, obs = confidence metrics (total_expression,
      mean_dropout_prob, is_occupied). Thresholds are stored in
      uns['occupancy_info'].
    """
    device = device or getattr(model.args, 'device', 'cpu')
    coords_tensor = torch.from_numpy(
        np.asarray(query_coords, dtype=np.float32) * spatial_scaling
    ).float().to(device)

    logging.info(f"In-silico query on {len(coords_tensor)} spots...")
    results = sample_points(model, coords_tensor, batch_size=batch_size)

    expression = results['expression'].cpu().numpy()
    total_expr = expression.sum(axis=1)

    mean_dropout = None
    if 'dropout_prob' in results:
        mean_dropout = results['dropout_prob'].cpu().numpy().mean(axis=1)

    if min_total_expr == 'auto':
        min_total_expr_value = _otsu_threshold(np.log1p(total_expr))
        min_total_expr_value = float(np.expm1(min_total_expr_value))
    else:
        min_total_expr_value = float(min_total_expr)

    is_occupied = total_expr > min_total_expr_value
    if max_mean_dropout is not None and mean_dropout is not None:
        is_occupied &= mean_dropout <= max_mean_dropout

    logging.info(
        f"Occupancy: threshold total_expression>{min_total_expr_value:.4f}"
        + (f", mean_dropout<={max_mean_dropout}" if max_mean_dropout is not None else "")
        + f" -> {int(is_occupied.sum())}/{len(is_occupied)} occupied spots."
    )

    obs = pd.DataFrame({
        'total_expression': total_expr,
        'is_occupied': is_occupied,
    }, index=[f'query_{i}' for i in range(len(query_coords))])
    if mean_dropout is not None:
        obs['mean_dropout_prob'] = mean_dropout

    if occupancy:
        keep = is_occupied
    else:
        keep = np.ones(len(query_coords), dtype=bool)

    n_genes = expression.shape[1]
    var = pd.DataFrame(index=list(var_names) if var_names is not None else [f'gene_{i}' for i in range(n_genes)])
    adata = anndata.AnnData(X=expression[keep], obs=obs[keep], var=var)
    adata.obsm['spatial'] = np.asarray(query_coords, dtype=np.float32)[keep]
    if 'dropout_prob' in results:
        adata.obsm['dropout_prob'] = results['dropout_prob'].cpu().numpy()[keep]
    adata.uns['occupancy_info'] = {
        'n_query_spots': int(len(query_coords)),
        'n_occupied_spots': int(is_occupied.sum()),
        'min_total_expr': min_total_expr_value,
        'max_mean_dropout': max_mean_dropout,
        'occupancy_filter_applied': bool(occupancy),
    }
    return adata
