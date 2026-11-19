#!/bin/bash
# NTF benchmark runs: 5 scenarios, sequential on one GPU.
# Run from the repository root with the `ntf` conda env activated.
set -u
GPU=${GPU:-0}
H5AD=mytrain/data/deep_starmap_brain/processed/deep_starmap_brain.h5ad

for item in senario1:group senario2:group2 senario3:group4 head:head tail:tail; do
  scen=${item%%:*}; col=${item##*:}
  CUDA_VISIBLE_DEVICES=$GPU python scripts/train_and_eval.py \
    -i "$H5AD" -o "mytrain/res/deep_starmap_brain/${scen}" \
    --slice_id slice_id --split_mode label --split_column "$col" \
    --train_label train --test_label test --cat_keys Harmony_labels
  echo "NTF_${scen} exit=$?"
done
