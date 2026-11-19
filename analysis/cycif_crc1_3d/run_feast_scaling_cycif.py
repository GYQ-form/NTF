"""FEAST scaling test on CRC1 adjacent slices (baseline feasibility evidence).

Measures FEAST wall time / peak RSS / peak GPU memory at increasing per-slice
cell counts, for extrapolation to the full ~510k cells/slice of CRC1.

Design: a mini 3-slice dataset is built from CRC1 sections 49 (train),
50 (test), 51 (train), each subsampled to n cells (uniform random, seed=42).
The existing bm_feast.py pipeline is run on it via run_with_monitor.sh
(env feast, GPU 7). n in {2000, 5000, 10000, 20000}; any single run exceeding
2 h wall time aborts the series.

Outputs under experiment/cycif_crc1_benchmark/baseline_feasibility/:
  monitor/feast_n<n>.json, logs/feast_n<n>.log, feast_scaling.csv

Usage:
    conda activate ntf
    python mytrain/scripts/run_feast_scaling_cycif.py
"""

import json
import os
import subprocess

import anndata
import numpy as np

H5AD = '/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/processed/cycif_crc1.h5ad'
OUT_BASE = '/home/gongyuqiao/ur_annotation/NTF/mytrain/revision/experiment/cycif_crc1_benchmark/baseline_feasibility'
BM_FEAST = '/home/gongyuqiao/ur_annotation/NTF/mytrain/scripts/FEAST/bm_feast.py'
MONITOR = '/home/gongyuqiao/ur_annotation/NTF/mytrain/scripts/gene_count_benchmark/run_with_monitor.sh'
WORKDIR = '/home/gongyuqiao/ur_annotation/NTF'

SIZES = [2000, 5000, 10000, 20000]
SEED = 42
TIMEOUT_S = 7200  # 2 h cap per run

# sections 49/50/51 -> slice ids 10/11/12 (z order); middle slice is the test
TRAIN_SEC, TEST_SEC = [49, 51], [50]


def build_mini(adata, n, path):
    rng = np.random.default_rng(SEED)
    parts = []
    for sec in TRAIN_SEC + TEST_SEC:
        sub = adata[adata.obs['slice_id'].to_numpy() == sec]
        take = rng.choice(sub.n_obs, size=min(n, sub.n_obs), replace=False)
        sub = sub[take].copy()
        sub.obs['group'] = 'test' if sec in TEST_SEC else 'train'
        parts.append(sub)
    mini = anndata.concat(parts, join='outer')
    keep_obs = ['slice_id', 'group']
    mini.obs = mini.obs[keep_obs].copy()
    mini.obs['slice_id'] = mini.obs['slice_id'].astype(int).astype(str)
    mini.obs['z'] = np.asarray(mini.obsm['spatial_um'][:, 2], dtype=np.float32)
    mini.obsm['spatial'] = np.asarray(mini.obsm['spatial_um'], dtype=np.float32)
    del mini.obsm['spatial_um']
    mini.X = np.asarray(mini.X, dtype=np.float32)  # raw intensities; bm_feast normalizes itself
    mini.write_h5ad(path, compression='gzip')
    return mini.n_obs


def main():
    os.makedirs(os.path.join(OUT_BASE, 'monitor'), exist_ok=True)
    os.makedirs(os.path.join(OUT_BASE, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(OUT_BASE, 'tmp'), exist_ok=True)

    print(f'Reading {H5AD} ...')
    adata = anndata.read_h5ad(H5AD)

    rows = []
    for n in SIZES:
        mini_path = os.path.join(OUT_BASE, 'tmp', f'mini_n{n}.h5ad')
        n_cells = build_mini(adata, n, mini_path)
        json_path = os.path.join(OUT_BASE, 'monitor', f'feast_n{n}.json')
        log_path = os.path.join(OUT_BASE, 'logs', f'feast_n{n}.log')
        out_dir = os.path.join(OUT_BASE, 'tmp', f'feast_out_n{n}')
        cmd = ['bash', MONITOR, 'feast', '7', WORKDIR, json_path, log_path,
               'python', BM_FEAST, '-i', mini_path, '-o', out_dir,
               '--slice_id', 'slice_id', '--group_key', 'group', '--z_col', 'z',
               '--ot_method', 'sinkhorn', '--ot_reg', '0.05', '--no_pca']
        print(f'=== n={n}/slice (total {n_cells}) ===')
        try:
            subprocess.run(cmd, timeout=TIMEOUT_S)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        if os.path.exists(json_path):
            meta = json.load(open(json_path))
            rows.append({'n_per_slice': n, 'n_cells_total': n_cells,
                         'wall_time_s': meta['wall_time_s'],
                         'peak_rss_kb': meta['peak_rss_kb'],
                         'peak_gpu_mem_mib': meta['peak_gpu_mem_mib'],
                         'exit_code': meta['exit_code'], 'timed_out': timed_out})
        else:
            rows.append({'n_per_slice': n, 'n_cells_total': n_cells,
                         'wall_time_s': np.nan, 'peak_rss_kb': np.nan,
                         'peak_gpu_mem_mib': np.nan, 'exit_code': -1, 'timed_out': timed_out})
        print(rows[-1])
        if timed_out:
            print(f'n={n} exceeded {TIMEOUT_S}s, aborting series.')
            break
        if rows[-1]['exit_code'] != 0:
            print(f'n={n} failed (exit {rows[-1]["exit_code"]}), aborting series.')
            break

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_BASE, 'feast_scaling.csv'), index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
