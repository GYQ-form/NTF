"""Fit FEAST scaling (time/memory vs cells per slice) and extrapolate to full CRC1 scale.

Reads experiment/cycif_crc1_benchmark/baseline_feasibility/feast_scaling.csv,
fits log-log power laws, extrapolates to 510,000 cells/slice (CRC1 average),
and writes feasibility_extrapolation.csv.

Usage:
    conda activate ntf
    python mytrain/scripts/feast_scaling_extrapolate.py
"""

import numpy as np
import pandas as pd

CSV = '/home/gongyuqiao/ur_annotation/NTF/mytrain/revision/experiment/cycif_crc1_benchmark/baseline_feasibility/feast_scaling.csv'
OUT = '/home/gongyuqiao/ur_annotation/NTF/mytrain/revision/experiment/cycif_crc1_benchmark/baseline_feasibility/feasibility_extrapolation.csv'

FULL_N = 510_000  # average cells per CRC1 slice
GPU_MEM_GB = 80   # A100-class GPU memory ceiling
RAM_TB = 1.9      # machine RAM

df = pd.read_csv(CSV)
df = df[df['exit_code'] == 0]
n = df['n_per_slice'].to_numpy(float)

rows = []
for col, unit in [('wall_time_s', 'seconds'), ('peak_rss_kb', 'GB'), ('peak_gpu_mem_mib', 'GB')]:
    y = df[col].to_numpy(float)
    if col == 'peak_rss_kb':
        y = y / 1e6  # -> GB
    elif col == 'peak_gpu_mem_mib':
        y = y / 1024  # -> GB
    # naive log-log power law (dominated by fixed overheads at small n)
    slope, intercept = np.polyfit(np.log(n), np.log(y), 1)
    pred_naive = float(np.exp(intercept + slope * np.log(FULL_N)))
    # quadratic model y = a + b*n + c*n^2: captures the OT cost-matrix term,
    # which is negligible at <=20k cells but dominates at full scale
    A = np.column_stack([np.ones_like(n), n, n ** 2])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred_quad = float(coef @ np.array([1.0, FULL_N, FULL_N ** 2]))
    rows.append({'quantity': col, 'unit': unit,
                 'loglog_slope': slope, f'naive_powerlaw_at_n{FULL_N}': pred_naive,
                 'quad_coef_c': coef[2], f'quadratic_at_n{FULL_N}': pred_quad})
    print(f'{col}: loglog slope={slope:.2f} -> naive {pred_naive:,.0f} {unit}; '
          f'quadratic fit -> {pred_quad:,.0f} {unit} at {FULL_N:,} cells/slice')

# analytic OT cost-matrix memory at full scale (float32 + float64)
for dtype_bytes, name in [(4, 'float32'), (8, 'float64')]:
    mem_gb = FULL_N ** 2 * dtype_bytes / 1e9
    print(f'OT cost matrix ({name}) at {FULL_N:,} x {FULL_N:,}: {mem_gb:,.0f} GB '
          f'(GPU ceiling {GPU_MEM_GB} GB, RAM {RAM_TB} TB)')

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(f'wrote {OUT}')
