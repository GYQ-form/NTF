#!/bin/bash
# R1 Major7 ablation — STARmap mouse VC 5-gene real data side.
# Baseline here is M1 (--no_dropout) because enabling dropout on this dataset
# produces NaN in training (pre-existing, confirmed bug).
# Usage: bash run_ablation_mousevc.sh [config ...]  (default: M0_nan_check once on senario1 + M1..M9 x 2 scenarios)
# Outputs: mytrain/res/mouseVC/ablation/<config>/<scenario>/
set -u

NTF_ROOT=/home/gongyuqiao/ur_annotation/NTF
DATA=$NTF_ROOT/mytrain/data/mouse_VC/processed/mouseVC_5gene.h5ad
RES_ROOT=$NTF_ROOT/mytrain/res/mouseVC/ablation
MONITOR=$NTF_ROOT/mytrain/scripts/gene_count_benchmark/run_with_monitor.sh
GPU=${GPU:-7}
ENV=ntf

# all configs include --no_dropout (see header); M0_nan_check is the only one with dropout on
declare -A FLAGS=(
  [M0_nan_check]=""
  [M1_no_dropout]="--no_dropout"
  [M2_no_psf]="--no_dropout --n_samples 1"
  [M3_no_smooth]="--no_dropout --weight_expr 0"
  [M4_no_sliceemb]="--no_dropout --n_features_slice 0"
  [M5_no_pixvar]="--no_dropout --no_pixel_variance"
  [M6_no_slicevar]="--no_dropout --no_slice_variance"
  [M7_no_uncertainty]="--no_dropout --no_pixel_variance --no_slice_variance"
  [M8_bare]="--no_dropout --n_samples 1 --weight_expr 0 --n_features_slice 0 --no_pixel_variance --no_slice_variance"
  [M9_bias_on]="--no_dropout --n_levels_bias 4"
)
declare -A SPLIT=( [senario1]=group [senario3]=group4 )

run_one () {
  local cfg=$1 sc=$2
  local out=$RES_ROOT/$cfg/$sc
  if [ -f "$out/test_results.h5ad" ]; then
    echo "[skip] $cfg/$sc already done"; return 0
  fi
  echo "=== $cfg / $sc (split col ${SPLIT[$sc]}) : ${FLAGS[$cfg]} ==="
  bash "$MONITOR" "$ENV" "$GPU" "$NTF_ROOT" "$out/monitor.json" "$out/train.log" \
    python scripts/train_and_eval.py \
      -i "$DATA" \
      -o "$out" \
      --slice_id slice_id --split_mode label --split_column "${SPLIT[$sc]}" \
      --train_label train --test_label test \
      ${FLAGS[$cfg]}
  echo "=== $cfg / $sc exit=$? ==="
}

if [ $# -gt 0 ]; then
  for cfg in "$@"; do
    if [ "$cfg" = "M0_nan_check" ]; then run_one M0_nan_check senario1
    else run_one "$cfg" senario1; run_one "$cfg" senario3; fi
  done
else
  run_one M0_nan_check senario1   # expected NaN; evidence for the record
  for cfg in M1_no_dropout M2_no_psf M3_no_smooth M4_no_sliceemb M5_no_pixvar M6_no_slicevar M7_no_uncertainty M8_bare M9_bias_on; do
    run_one "$cfg" senario1
    run_one "$cfg" senario3
  done
fi
