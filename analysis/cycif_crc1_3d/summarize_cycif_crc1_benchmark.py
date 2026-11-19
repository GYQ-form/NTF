"""Build the Excel summary workbook for the CyCIF CRC1 benchmark.

Collects the per-section CSVs of an experiment directory (v1 or v2 layout under
experiment/cycif_crc1_benchmark/<version>/) into one xlsx with 15 sheets.

Usage:
    conda activate ntf
    python summarize_cycif_crc1_benchmark.py \
        --exp-dir mytrain/revision/experiment/cycif_crc1_benchmark/v2
"""

import argparse
from pathlib import Path

import pandas as pd

SHEETS = {
    'recon_summary': 'recon/recon_metrics_summary.csv',
    'recon_long': 'recon/recon_metrics_long.csv',
    'svg_summary': 'svg/svg_benchmark_summary.csv',
    'svg_gene_moran': 'svg/gene_moran_values.csv',
    'celltyping': 'celltyping/celltyping_metrics.csv',
    'nichetyping': 'nichetyping/nichetyping_metrics.csv',
    'niche_per_niche': 'nichetyping/niche_per_niche_metrics.csv',
    'baseline_feasibility': 'baseline_feasibility/feast_scaling.csv',
    'baseline_extrapolation': 'baseline_feasibility/feasibility_extrapolation.csv',
    'efficiency': 'efficiency_summary.csv',
    'method_status': 'recon/method_status_recon.csv',
    'annotation_counts': 'annotation/celltype_counts.csv',
    'gating_thresholds': 'annotation/gating_thresholds.csv',
    'tuning_log': 'tuning_log.csv',
    'split_stats': 'split_stats.csv',
    'knn_ceiling': 'knn_transfer_ceiling_summary.csv',
    'knn_ceiling_per_gene': 'knn_transfer_ceiling.csv',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp-dir', required=True)
    ap.add_argument('--out', default=None, help='default: <exp-dir>/cycif_crc1_benchmark_summary.xlsx')
    args = ap.parse_args()
    exp = Path(args.exp_dir)
    out = Path(args.out) if args.out else exp / 'cycif_crc1_benchmark_summary.xlsx'

    with pd.ExcelWriter(out, engine='openpyxl') as xw:
        for sheet, rel in SHEETS.items():
            p = exp / rel
            if not p.exists():
                print(f'[skip] {sheet}: {p} missing')
                continue
            df = pd.read_csv(p)
            df.to_excel(xw, sheet_name=sheet, index=False)
            print(f'[sheet] {sheet}: {len(df)} rows')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
