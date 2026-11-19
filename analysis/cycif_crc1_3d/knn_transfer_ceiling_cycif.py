"""kNN(15) direct-transfer ceiling for the CyCIF CRC1 benchmark (senario1 test set).

Reference frame for interpreting NTF reconstruction / annotation accuracy:
- expression ceiling: per-gene Pearson between GT (normalized log1p X) and the
  mean expression of the 15 nearest train-set neighbours in 3D coordinate space;
- label ceiling: ARI of majority-vote cell_type_main transfer.

Run with --input .../cycif_crc1_bm_aligned.h5ad for the v2 (MAPS-aligned) benchmark.

Usage:
    conda activate ntf
    python knn_transfer_ceiling_cycif.py --input <bm h5ad> --output <csv>
"""

import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score
from scipy.stats import pearsonr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--split-column', default='group', help='senario1 split column')
    ap.add_argument('--k', type=int, default=15)
    args = ap.parse_args()

    adata = sc.read_h5ad(args.input)
    test = (adata.obs[args.split_column].astype(str) == 'test').to_numpy()
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    print(f'train={int((~test).sum()):,} test={int(test.sum()):,} k={args.k}')

    tree = cKDTree(coords[~test])
    _, nn = tree.query(coords[test], k=args.k, workers=-1)  # (n_test, k) train indices

    # expression ceiling: sparse weight matrix (test x train) @ X_train
    n_test, n_train = int(test.sum()), int((~test).sum())
    W = sp.csr_matrix(
        (np.full(n_test * args.k, 1.0 / args.k, dtype=np.float32),
         (np.repeat(np.arange(n_test), args.k), nn.ravel())),
        shape=(n_test, n_train))
    X_train = adata.X[~test]
    if not sp.issparse(X_train):
        X_train = sp.csr_matrix(X_train)
    pred = np.asarray((W @ X_train).todense())
    true = adata.X[test]
    true = true.toarray() if sp.issparse(true) else np.asarray(true)

    pears = []
    for i in range(adata.n_vars):
        t, p = true[:, i], pred[:, i]
        nz = t > 0
        if nz.sum() < 2 or np.var(t[nz]) < 1e-10 or np.var(p[nz]) < 1e-10:
            pears.append(np.nan)
            continue
        pears.append(pearsonr(t, p)[0])
    pears = np.array(pears)
    print(f'mean Pearson (expr transfer) = {np.nanmean(pears):.4f}')

    # label ceiling: majority vote of neighbour cell_type_main
    labels = adata.obs['cell_type_main'].astype(str).to_numpy()
    train_lab = labels[~test]
    votes = train_lab[nn]  # (n_test, k) strings
    pred_lab = np.array([pd.Series(row).mode()[0] for row in votes])
    ari = adjusted_rand_score(labels[test], pred_lab)
    print(f'label transfer ARI (cell_type_main) = {ari:.4f}')

    pd.DataFrame({
        'gene': list(adata.var_names),
        'pearson_knn_transfer': pears,
    }).to_csv(args.output, index=False)
    pd.DataFrame({
        'metric': ['expr_pearson_mean', 'label_ari_cell_type_main'],
        'value': [float(np.nanmean(pears)), float(ari)],
        'setting': [f'kNN({args.k}) on {args.split_column} test, input={args.input}'] * 2,
    }).to_csv(args.output.replace('.csv', '_summary.csv'), index=False)
    print('wrote', args.output)


if __name__ == '__main__':
    main()
