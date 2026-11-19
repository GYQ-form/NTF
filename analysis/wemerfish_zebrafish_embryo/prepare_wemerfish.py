"""Build processed h5ad for weMERFISH zebrafish embryo benchmark (Wan et al. 2026).

Input: mytrain/data/wemerfish_zebrafish_embryo/weMERFISH_measured_C_6s_E2_rescaled_z.h5ad
  - 25,858 cells x 495 measured genes, whole-mount native 3D coordinates
    (obsm['spatial'], z already rescaled to real sample thickness)
  - obsm['X_raw'] = raw measured expression (used as the base for preprocessing)
  - annotation hierarchy: germlayer (5) -> type (10) -> tissue (39) -> clusters (51)

Output:
  - processed/wemerfish_C6s_E2.h5ad  (X = normalize_total(1e4)+log1p of X_raw;
      layers['raw'] = X_raw; obsm['spatial']; slice_id from z binning;
      obs split columns for 5 scenarios)
  - processed/gene_zero_frac.csv

Scenario split definitions (slice_id = 1..16 from equal-width z bins, plan
revision/plan/R2_Major2_weMERFISH_benchmark_plan.md section 2):
  group   (senario1): test if slice_id in {2,4,6,8,10,12,14}  -> interleaved holdout
  group2  (senario2): test if slice_id in {8,9,10}            -> consecutive middle band
  group4  (senario3): test if slice_id in {1,2,15,16}         -> both-end extrapolation
  head              : test if slice_id <= 2                    -> head extrapolation
  tail              : test if slice_id >= 15                   -> tail extrapolation
"""

import argparse
import os
import time

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "wemerfish_zebrafish_embryo")
INPUT_H5AD = "weMERFISH_measured_C_6s_E2_rescaled_z.h5ad"

N_SLICES = 16
MIN_CELLS_PER_SLICE = 200
SPLIT_DEFS = {
    "group": lambda s: s.isin([2, 4, 6, 8, 10, 12, 14]),
    "group2": lambda s: s.isin([8, 9, 10]),
    "group4": lambda s: s.isin([1, 2, 15, 16]),
    "head": lambda s: s <= 2,
    "tail": lambda s: s >= 15,
}


def bin_z(z: np.ndarray, n_slices: int) -> np.ndarray:
    # Quantile binning: equal cell counts per slice. The embryo is a curved
    # structure, so z (thickness) is highly non-uniform — equal-width bins
    # leave the top/bottom caps nearly empty. Slice thickness therefore varies
    # along z (recorded in slice_cell_counts.csv via z range per slice).
    edges = np.quantile(z, np.linspace(0, 1, n_slices + 1))
    edges = np.unique(edges)
    assert len(edges) == n_slices + 1, f"degenerate z quantiles: {len(edges)-1} bins"
    sid = np.digitize(z, edges[1:-1]) + 1
    return sid.astype(np.int64)


def main():
    parser = argparse.ArgumentParser(description="Prepare weMERFISH C_6s_E2 h5ad for benchmark.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output", type=str, default=None,
                        help="Output h5ad path (default: <data_dir>/processed/wemerfish_C6s_E2.h5ad)")
    args = parser.parse_args()

    in_path = os.path.join(args.data_dir, INPUT_H5AD)
    out_path = args.output or os.path.join(args.data_dir, "processed", "wemerfish_C6s_E2.h5ad")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    t0 = time.time()
    print(f"[1/6] Reading {in_path} ...", flush=True)
    src = anndata.read_h5ad(in_path)
    print(f"      shape={src.shape}, elapsed={time.time()-t0:.1f}s", flush=True)

    # --- expression: preprocess from X_raw ---
    x_raw = np.asarray(src.obsm["X_raw"], dtype=np.float32)
    assert x_raw.shape == src.shape and np.isfinite(x_raw).all() and (x_raw >= 0).all()
    print(f"[2/6] X_raw: min={x_raw.min():.4f}, max={x_raw.max():.2f}, "
          f"zero_frac={float((x_raw == 0).mean()):.4f}", flush=True)

    # --- z binning into virtual slices (quantile bins, equal cell counts) ---
    print(f"[3/6] Binning z into {N_SLICES} quantile-based virtual slices ...", flush=True)
    spatial = np.asarray(src.obsm["spatial"], dtype=np.float32)
    z = spatial[:, 2].astype(np.float64)
    n_slices = N_SLICES
    sid = bin_z(z, n_slices)
    counts_per_slice = pd.Series(sid).value_counts().sort_index()
    print(f"      cells per slice: min={counts_per_slice.min()}, max={counts_per_slice.max()}", flush=True)
    assert counts_per_slice.min() >= MIN_CELLS_PER_SLICE
    # record actual z thickness per slice for the record
    z_range = pd.DataFrame({"slice_id": sid, "z": z}).groupby("slice_id")["z"].agg(["min", "max"])

    # --- obs ---
    obs = pd.DataFrame({
        "x": spatial[:, 0],
        "y": spatial[:, 1],
        "z": sid.astype(np.float32),  # slice ordinal, used by SpatialZ/FEAST --z_col
        "z_cont": z.astype(np.float32),
        "slice_id": sid,
        "cellvol": src.obs["cellvol"].to_numpy(dtype=np.float32),
        "blank_ct": src.obs["blank_ct"].to_numpy(dtype=np.float32),
        "clusters": pd.Categorical(src.obs["clusters"]),
        "type": pd.Categorical(src.obs["type"]),
        "tissue": pd.Categorical(src.obs["tissue"]),
        "germlayer": pd.Categorical(src.obs["germlayer"]),
    })
    sid_s = pd.Series(sid, index=obs.index)
    for col, fn in SPLIT_DEFS.items():
        obs[col] = pd.Categorical(np.where(fn(sid_s), "test", "train"), categories=["train", "test"])
    obs.index = [f"cell_{i}" for i in range(obs.shape[0])]

    # --- var ---
    zero_frac = 1.0 - (x_raw > 0).sum(axis=0) / x_raw.shape[0]
    var = pd.DataFrame({"zero_frac": zero_frac.astype(np.float32)},
                       index=pd.Index(src.var_names.astype(str), name="gene"))

    # --- normalize_total + log1p (same as Deep-STARmap/mouseVC pipeline) ---
    print("[4/6] sc.pp.normalize_total(target_sum=1e4) + sc.pp.log1p on X_raw ...", flush=True)
    adata = anndata.AnnData(X=sp.csr_matrix(x_raw), obs=obs, var=var)
    adata.obsm["spatial"] = spatial
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = adata.X.tocsr().astype(np.float32)
    adata.layers["raw"] = sp.csr_matrix(x_raw)

    # --- QC ---
    print("[5/6] QC checks ...", flush=True)
    assert adata.shape == (25858, 495), f"Unexpected shape {adata.shape}"
    assert adata.obs["clusters"].nunique() == 51 and adata.obs["type"].nunique() == 10
    for col in SPLIT_DEFS:
        vc = adata.obs[col].value_counts()
        n_test_slices = adata.obs.loc[adata.obs[col] == "test", "slice_id"].nunique()
        print(f"      split {col}: train={vc.get('train', 0)}, test={vc.get('test', 0)}, test_slices={n_test_slices}",
              flush=True)
        n_mix = adata.obs.groupby("slice_id")[col].nunique().gt(1).sum()
        assert n_mix == 0, f"split column {col}: {n_mix} slices mix train/test labels"
    print(f"      X sparse={sp.issparse(adata.X)}, nnz={adata.X.nnz}, "
          f"density={adata.X.nnz / (adata.shape[0] * adata.shape[1]):.4f}", flush=True)
    # sanity: correlation between our preprocessing and the official normalized X
    off = np.asarray(src.X, dtype=np.float32)
    ours = adata.X[:5000].toarray()
    corr = np.corrcoef(ours.ravel(), off[:5000].ravel())[0, 1]
    print(f"      sanity corr(ours, official X) on 5k cells = {corr:.4f}", flush=True)

    # --- write ---
    print(f"[6/6] Writing {out_path} ...", flush=True)
    adata.write_h5ad(out_path, compression="gzip", compression_opts=4)
    zero_csv = os.path.join(os.path.dirname(out_path), "gene_zero_frac.csv")
    var.reset_index().to_csv(zero_csv, index=False)
    slice_csv = os.path.join(os.path.dirname(out_path), "slice_cell_counts.csv")
    slice_tbl = counts_per_slice.rename("n_cells").rename_axis("slice_id").reset_index()
    slice_tbl = slice_tbl.merge(z_range.reset_index(), on="slice_id")
    slice_tbl.to_csv(slice_csv, index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Done. h5ad size={size_mb:.1f} MB, gene_zero_frac -> {zero_csv}, "
          f"slice counts -> {slice_csv}, total elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
