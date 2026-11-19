"""Centralized re-evaluation of scale-test results for 4 methods (NTF/SpatialZ/FEAST/UniST).

Unified protocol (following gene_count central_eval.py and NTF train_dropout_simu.py):
- GT: layers['truth'] of the INPUT {n}w.h5ad (real noise-free truth), reconnected to each
  method's test_results/pred h5ad by obs_names (asserted to match the input's test set).
- Coordinates: GT test coordinates (all methods' predictions already live there: NTF/FEAST/
  SpatialZ natively, UniST via the wrapper's 3D 1-NN mapping).
- Per gene: Pearson + Spearman (full test-cell vector; gene skipped if <2 non-zero truth
  cells or near-zero variance after filtering — NTF's rule), RMSE, 3D SSIM with NTF's
  helper_func.calculate_spatial_metrics implementation (coords scaled to <=64 max dim,
  adaptive odd win_size), truth volume as reference.
- Spearman vectorized: pearson on column-wise ranks.

Legacy caveat (recorded in README): each method's own benchmark_summary.csv used mixed
protocols — NTF evaluated vs layers['truth'], FEAST/SpatialZ vs observed X. This script
recomputes everything vs real truth for a fair 4-method table.

Outputs (under revision/experiment/r2t5_unist_benchmark/):
  scale_eval/gene_metrics/{method}__{tier}.csv   per-gene metrics (central protocol)
  scale_benchmark_summary_unist.csv              UniST tiers (mode column: inr/fast)
  scale_benchmark_summary_4methods.csv           all methods (method column)
"""

import os
import numpy as np
import pandas as pd
import anndata
from scipy.stats import pearsonr
from scipy.stats import rankdata
from skimage.metrics import structural_similarity as ssim

BASE = "/home/gongyuqiao/ur_annotation/NTF/mytrain"
EXP = "/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/r2t5_unist_benchmark"
TIERS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

METHOD_FILES = {
    "NTF": lambda n: f"{BASE}/res/simulation/scale_test/{n}w/test_results.h5ad",
    "FEAST": lambda n: f"{BASE}/res/FEAST/scale_test/{n}w/test_results.h5ad",
    "SpatialZ": lambda n: f"/home/gongyuqiao/ur_annotation/SpatialZ/res/simulation/{n}w/test_results.h5ad",
    "UniST-inr": lambda n: (f"{BASE}/res/UniST/scale_test/{n}w/test_results.h5ad" if n == 10
                            else f"{BASE}/res/UniST/scale_test/{n}w_inr/test_results.h5ad"),
    "UniST-fast": lambda n: (f"{BASE}/res/UniST/scale_test/{n}w_fast/test_results.h5ad" if n == 10
                             else f"{BASE}/res/UniST/scale_test/{n}w/test_results.h5ad"),
}
# UniST inr coverage: 10,20,30,40,50,100; fast: all 19 tiers (10w fast added 2026-10-04).
UNIST_INR_TIERS = {10, 20, 30, 40, 50, 100}


def vec_spearman(T, P):
    """Column-wise spearman = pearson on ranks. T, P: (n_cells, n_genes)."""
    Tr = np.apply_along_axis(rankdata, 0, T)
    Pr = np.apply_along_axis(rankdata, 0, P)
    Tc = Tr - Tr.mean(0)
    Pc = Pr - Pr.mean(0)
    num = (Tc * Pc).sum(0)
    den = np.sqrt((Tc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def vec_pearson(T, P):
    Tc = T - T.mean(0)
    Pc = P - P.mean(0)
    num = (Tc * Pc).sum(0)
    den = np.sqrt((Tc ** 2).sum(0) * (Pc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def ssim_3d_per_gene(coords, T, P):
    """NTF helper_func.calculate_spatial_metrics SSIM part, truth as reference."""
    coords_original = coords
    target_min_dim, target_max_dim = 10, 64
    coords_max = coords_original.max(axis=0)
    coords_min = coords_original.min(axis=0)
    original_dims = coords_max - coords_min + 1
    max_dim = original_dims.max()
    min_dim = original_dims.min()
    if max_dim > target_max_dim:
        scale_factor = target_max_dim / max_dim
    elif min_dim < target_min_dim:
        scale_factor = target_min_dim / min_dim
    else:
        scale_factor = 1.0
    c = ((coords_original - coords_min) * scale_factor).astype(int)
    grid_dims = np.maximum(c.max(axis=0) + 1, 3)
    min_gd = grid_dims.min()
    win_size = min_gd - 1 if min_gd % 2 == 0 else min_gd
    if win_size < 3:
        win_size = 3
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    out = np.full(T.shape[1], np.nan)
    for i in range(T.shape[1]):
        tv = np.zeros(grid_dims)
        pv = np.zeros(grid_dims)
        tv[x, y, z] = T[:, i]
        pv[x, y, z] = P[:, i]
        dr = max(tv.max(), pv.max()) - min(tv.min(), pv.min())
        out[i] = 1.0 if dr == 0 else ssim(tv, pv, data_range=dr, win_size=win_size)
    return out


def main():
    os.makedirs(f"{EXP}/scale_eval/gene_metrics", exist_ok=True)
    summary_rows = []

    for n in TIERS:
        inp = anndata.read_h5ad(f"{BASE}/data/simulation/scale_test/{n}w.h5ad")
        test = inp[inp.obs["group"] == "test"]
        truth = np.asarray(test.layers["truth"], dtype=np.float32)
        coords = np.asarray(test.obsm["spatial"], dtype=np.float64)
        names = test.obs_names.values
        # NTF skip rule: genes with <2 non-zero truth cells or ~zero variance among them
        valid = (truth > 0).sum(0) >= 2
        for g in np.where(valid)[0]:
            t = truth[truth[:, g] > 0, g]
            if t.var() < 1e-10:
                valid[g] = False

        for method, getf in METHOD_FILES.items():
            if method == "UniST-inr" and n not in UNIST_INR_TIERS:
                continue
            f = getf(n)
            gm_path = f"{EXP}/scale_eval/gene_metrics/{method}__{n}w.csv"
            if os.path.exists(gm_path):
                gm = pd.read_csv(gm_path, index_col=0)
                row = {"scale": f"{n}w", "scale_num": n, "method": method}
                for mname in ["pearson_corr", "spearman_corr", "rmse", "ssim_3d"]:
                    row[f"{mname}_mean"] = gm[mname].mean()
                    row[f"{mname}_std"] = gm[mname].std()
                summary_rows.append(row)
                continue
            if not os.path.exists(f):
                print(f"{method} {n}w: MISSING {f}")
                continue
            ad = anndata.read_h5ad(f)
            assert set(ad.obs_names) == set(names), f"{method} {n}w: test cell mismatch"
            pred = np.asarray(ad.layers["prediction"], dtype=np.float32)[
                pd.Index(ad.obs_names).get_indexer(names)]
            assert np.allclose(np.asarray(ad.obsm["spatial"])[
                pd.Index(ad.obs_names).get_indexer(names)], coords), f"{method} {n}w coords"

            pr = vec_pearson(truth, pred)
            sp = vec_spearman(truth, pred)
            rm = np.sqrt(((truth - pred) ** 2).mean(0))
            ss = ssim_3d_per_gene(coords, truth, pred)
            for arr in (pr, sp):
                arr[~valid] = np.nan
            gm = pd.DataFrame({"pearson_corr": pr, "spearman_corr": sp,
                               "rmse": rm, "ssim_3d": ss}, index=test.var_names)
            gm.to_csv(f"{EXP}/scale_eval/gene_metrics/{method}__{n}w.csv")
            row = {"scale": f"{n}w", "scale_num": n, "method": method}
            for mname, arr in [("pearson_corr", pr), ("spearman_corr", sp),
                               ("rmse", rm), ("ssim_3d", ss)]:
                row[f"{mname}_mean"] = np.nanmean(arr)
                row[f"{mname}_std"] = np.nanstd(arr)
            summary_rows.append(row)
            print(f"{method} {n}w: pearson={row['pearson_corr_mean']:.4f} "
                  f"ssim={row['ssim_3d_mean']:.4f}", flush=True)
            del ad, pred

    df = pd.DataFrame(summary_rows).sort_values(["method", "scale_num"])
    df.to_csv(f"{EXP}/scale_benchmark_summary_4methods.csv", index=False)

    u = df[df["method"].str.startswith("UniST")].copy()
    u["mode"] = u["method"].str.split("-").str[1]
    u = u.drop(columns="method")
    u.to_csv(f"{EXP}/scale_benchmark_summary_unist.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
