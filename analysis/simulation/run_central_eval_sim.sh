#!/bin/bash
# R1 Major7 ablation — central re-evaluation of simulation-side runs against layers['truth'].
# Usage: bash run_central_eval_sim.sh [config ...]  (default: all M0..M9)
# Output per config: mytrain/res/simulation/ablation/<config>/central_eval/{gene_metrics_central.csv,eval_input.h5ad,central_eval_meta.json}
NTF_ROOT=/home/gongyuqiao/ur_annotation/NTF
DATA=$NTF_ROOT/mytrain/data/simulation/gene_count_test/100wcell_100gene.h5ad
RES_ROOT=$NTF_ROOT/mytrain/res/simulation/ablation

source /home/gongyuqiao/miniconda3/etc/profile.d/conda.sh
conda activate ntf
set -u
cd "$NTF_ROOT" || exit 1

if [ $# -gt 0 ]; then CONFIGS=("$@"); else
  CONFIGS=(M0_full M1_no_dropout M2_no_psf M3_no_smooth M4_no_sliceemb M5_no_pixvar M6_no_slicevar M7_no_uncertainty M8_bare M9_bias_on)
fi

for cfg in "${CONFIGS[@]}"; do
  res=$RES_ROOT/$cfg/test_results.h5ad
  out=$RES_ROOT/$cfg/central_eval
  if [ ! -f "$res" ]; then echo "[missing] $res"; continue; fi
  if [ -f "$out/gene_metrics_central.csv" ]; then echo "[skip] $cfg"; continue; fi
  echo "=== central_eval $cfg ==="
  python mytrain/scripts/gene_count_benchmark/central_eval.py \
    --input "$DATA" --result "$res" --output_dir "$out"
done
