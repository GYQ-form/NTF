#!/bin/bash
# R1 Major7 ablation — simulation side (1M cells, 100 genes, seed=42)
# Usage: bash run_ablation_sim.sh [config ...]   (default: all M0..M9)
# Outputs: mytrain/res/simulation/ablation/<config>/{best_model.pth,test_results.h5ad,res.log,...}
#          + monitor.json / train.log next to each run dir
set -u

NTF_ROOT=/home/gongyuqiao/ur_annotation/NTF
DATA=$NTF_ROOT/mytrain/data/simulation/gene_count_test/100wcell_100gene.h5ad
RES_ROOT=$NTF_ROOT/mytrain/res/simulation/ablation
MONITOR=$NTF_ROOT/mytrain/scripts/gene_count_benchmark/run_with_monitor.sh
GPU=${GPU:-6}
ENV=ntf

declare -A FLAGS=(
  [M0_full]=""
  [M1_no_dropout]="--no_dropout"
  [M2_no_psf]="--n_samples 1"
  [M3_no_smooth]="--weight_expr 0"
  [M4_no_sliceemb]="--n_features_slice 0"
  [M5_no_pixvar]="--no_pixel_variance"
  [M6_no_slicevar]="--no_slice_variance"
  [M7_no_uncertainty]="--no_pixel_variance --no_slice_variance"
  [M8_bare]="--no_dropout --n_samples 1 --weight_expr 0 --n_features_slice 0 --no_pixel_variance --no_slice_variance"
  [M9_bias_on]="--n_levels_bias 4"
)

if [ $# -gt 0 ]; then CONFIGS=("$@"); else
  CONFIGS=(M0_full M1_no_dropout M2_no_psf M3_no_smooth M4_no_sliceemb M5_no_pixvar M6_no_slicevar M7_no_uncertainty M8_bare M9_bias_on)
fi

for cfg in "${CONFIGS[@]}"; do
  out=$RES_ROOT/$cfg
  if [ -f "$out/test_results.h5ad" ]; then
    echo "[skip] $cfg already done"; continue
  fi
  echo "=== $cfg : ${FLAGS[$cfg]} ==="
  bash "$MONITOR" "$ENV" "$GPU" "$NTF_ROOT" "$out/monitor.json" "$out/train.log" \
    python scripts/train_and_eval.py \
      -i "$DATA" \
      -o "$out" \
      --slice_id slice_id --split_mode label --split_column group \
      --train_label train --test_label test \
      ${FLAGS[$cfg]}
  rc=$?
  echo "=== $cfg exit=$rc ==="
done
