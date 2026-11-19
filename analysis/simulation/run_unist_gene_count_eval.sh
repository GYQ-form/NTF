#!/bin/bash
# UniST gene_count 集中重评估 + SVG 保守性评估驱动脚本
# 协议与 gene_count_benchmark 三方法（NTF/SpatialZ/FEAST）完全一致：
#   1) central_eval.py：vs layers['truth'] 的 per-gene pearson/spearman/rmse + 3D SSIM，
#      并写出标准化 eval_input.h5ad；
#   2) eval_svg_conservation_sim.py：100k 抽样、3D kNN(k=15)、Moran's I + TopK SVG overlap，
#      输出到 revision/experiment/gene_count_benchmark/svg/unist_{inr,fast}/{g}gene/。
# 用法: bash run_unist_gene_count_eval.sh [mode g ...]   # 缺省跑全部 9 个 (mode,tier) 组合
BASE=/home/gongyuqiao/ur_annotation/NTF/mytrain
EXP=$BASE/revision/experiment/gene_count_benchmark
DATA=$BASE/data/simulation/gene_count_test
URES=$BASE/res/UniST/gene_count_test
CE=$BASE/scripts/gene_count_benchmark/central_eval.py
SVG=$BASE/scripts/eval_svg_conservation_sim.py

cd "$BASE"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ntf

run_one() {
  local mode=$1 g=$2 rd topk
  if [ "$mode" = inr ]; then rd="$URES/${g}gene"; else rd="$URES/${g}gene_fast"; fi
  echo "=== [$(date -Is)] central_eval UniST-${mode} ${g}gene ==="
  python "$CE" --input "$DATA/100wcell_${g}gene.h5ad" \
    --result "$rd/test_results.h5ad" --output_dir "$rd" || return 1
  case $g in
    1)  topk="1";;
    10) topk="5 10";;
    *)  topk="5 10 20 50";;
  esac
  echo "=== [$(date -Is)] svg eval UniST-${mode} ${g}gene (topk $topk) ==="
  python "$SVG" --dataset scale --result "$rd/eval_input.h5ad" \
    --output_dir "$EXP/svg/unist_${mode}/${g}gene" \
    --sample_size 100000 --k 15 --seed 42 --topk $topk || return 1
}

if [ $# -gt 0 ]; then
  while [ $# -ge 2 ]; do run_one "$1" "$2"; shift 2; done
else
  for spec in "inr 10" "inr 100" "inr 1000" "inr 10000" \
              "fast 1" "fast 10" "fast 100" "fast 1000" "fast 10000"; do
    run_one $spec
  done
fi
echo "=== [$(date -Is)] all done ==="
