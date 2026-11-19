#!/bin/bash
# Run NTF on the 5 CRC1 benchmark scenarios.
# Final config for this dataset (see experiment/cycif_crc1_benchmark README):
#   --single_precision  (fp16 mixed precision yields constant predictions here)
#   --weight_decay 0    (default Adam wd=1e-2 collapses the field to a coarse blob)
#   --no_dropout        (zero fraction of all 24 markers is 0.0000)
# Usage: bash run_ntf_cycif.sh <n_iter> [scenarios...]
# Env overrides: GPU (default 6), H5AD, RES_BASE, EXP — e.g. the v2 (MAPS-aligned
# coordinates) rerun used:
#   GPU=6 H5AD=.../cycif_crc1_bm_aligned.h5ad RES_BASE=mytrain/res/cycif_crc1_aligned \
#   EXP=.../experiment/cycif_crc1_benchmark/v2 bash run_ntf_cycif.sh 20000
set -u
cd /home/gongyuqiao/ur_annotation/NTF

GPU=${GPU:-6}
N_ITER=${1:-20000}; shift || true
SCENARIOS=${@:-"senario1 senario2 senario3 head tail"}

H5AD=${H5AD:-mytrain/data/cycif_crc1_3d/processed/cycif_crc1_bm.h5ad}
MONITOR=mytrain/scripts/gene_count_benchmark/run_with_monitor.sh
EXP=${EXP:-mytrain/revision/experiment/cycif_crc1_benchmark}
RES_BASE=${RES_BASE:-mytrain/res/cycif_crc1}

declare -A SCEN=( [senario1]=group [senario2]=group2 [senario3]=group4 [head]=head [tail]=tail )

mkdir -p $EXP/monitor $EXP/logs
for s in $SCENARIOS; do
  col=${SCEN[$s]}
  out=${RES_BASE}/${s}
  tag=${s}
  if [ "$N_ITER" != "20000" ]; then out=${RES_BASE}/${s}_it${N_ITER}; tag=${s}_it${N_ITER}; fi
  bash $MONITOR ntf $GPU /home/gongyuqiao/ur_annotation/NTF \
    $EXP/monitor/ntf_${tag}.json $EXP/logs/ntf_${tag}.log \
    python scripts/train_and_eval.py \
      -i $H5AD \
      -o $out \
      --slice_id slice_id --split_mode label --split_column ${col} \
      --train_label train --test_label test \
      --no_dropout --single_precision --weight_decay 0 --n_iter $N_ITER \
      --cat_keys cell_type cell_type_main
  echo "scenario ${s} exit=$?"
done
