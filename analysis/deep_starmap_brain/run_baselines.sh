#!/bin/bash
# Baseline runs: SpatialZ + FEAST on senario1/senario2 (interpolation scenarios;
# both methods cannot extrapolate, so senario3/head/tail are NTF-only).
# SpatialZ runs from its own repo with the `spatialz` env; FEAST uses the `feast` env.
set -u
GPU=${GPU:-0}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
H5AD="$REPO_ROOT/mytrain/data/deep_starmap_brain/processed/deep_starmap_brain.h5ad"
SPATIALZ_REPO=${SPATIALZ_REPO:-/home/gongyuqiao/ur_annotation/SpatialZ}

source /home/gongyuqiao/miniconda3/etc/profile.d/conda.sh

conda activate spatialz
for item in senario1:group senario2:group2; do
  scen=${item%%:*}; col=${item##*:}
  (cd "$SPATIALZ_REPO" && CUDA_VISIBLE_DEVICES=$GPU python train_ur.py \
    -i "$H5AD" -o "res/deep_starmap_brain/${scen}" \
    --slice_id slice_id --group_key "$col" --z_col z --syn_mode fast --cell_type_key Harmony_labels)
  echo "SpatialZ_${scen} exit=$?"
done

conda activate feast
for item in senario1:group senario2:group2; do
  scen=${item%%:*}; col=${item##*:}
  (cd "$REPO_ROOT" && CUDA_VISIBLE_DEVICES=$GPU python mytrain/scripts/FEAST/bm_feast.py \
    -i "$H5AD" -o "mytrain/res/FEAST/deep_starmap_brain/${scen}" \
    --slice_id slice_id --group_key "$col" --z_col z --ot_method sinkhorn --ot_reg 0.05)
  echo "FEAST_${scen} exit=$?"
done
