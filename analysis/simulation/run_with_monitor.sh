#!/bin/bash
# run_with_monitor.sh <conda_env> <gpu_id> <workdir> <out_json> <log_file> <cmd...>
# Runs <cmd> with a conda env and CUDA_VISIBLE_DEVICES=<gpu_id>, recording:
# wall time, exit code, peak GPU memory (polled every 5s), peak RSS (GNU time).
ENV=$1; GPU=$2; WD=$3; JSON=$4; LOG=$5; shift 5

source /home/gongyuqiao/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$WD" || exit 1

mkdir -p "$(dirname "$JSON")" "$(dirname "$LOG")"
PEAK_FILE=$(mktemp)
(
  PEAK=0
  while true; do
    M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null || echo 0)
    [ "$M" -gt "$PEAK" ] && PEAK=$M
    echo "$PEAK" > "$PEAK_FILE"
    sleep 5
  done
) &
MONPID=$!

START=$(date +%s)
CMD_STR="CUDA_VISIBLE_DEVICES=$GPU $*"
echo "CMD: $CMD_STR" > "$LOG.env"
/usr/bin/time -v env CUDA_VISIBLE_DEVICES="$GPU" "$@" > "$LOG" 2> "$LOG.time"
RC=$?
END=$(date +%s)
kill $MONPID 2>/dev/null; wait $MONPID 2>/dev/null

PEAK_GPU=$(cat "$PEAK_FILE"); rm -f "$PEAK_FILE"
PEAK_RSS=$(grep "Maximum resident set size" "$LOG.time" | awk '{print $6}')
ELAPSED=$(grep "wall clock" "$LOG.time" | awk '{print $8}')

python - "$JSON" <<EOF
import json
meta = {
    "cmd": "$CMD_STR",
    "conda_env": "$ENV",
    "gpu_id": "$GPU",
    "workdir": "$WD",
    "start_epoch": $START,
    "end_epoch": $END,
    "wall_time_s": $END - $START,
    "wall_time_hms": "$ELAPSED",
    "exit_code": $RC,
    "peak_gpu_mem_mib": int("$PEAK_GPU" or 0),
    "peak_rss_kb": int("$PEAK_RSS" or 0),
}
with open("$JSON", "w") as f:
    json.dump(meta, f, indent=2)
EOF
exit $RC
