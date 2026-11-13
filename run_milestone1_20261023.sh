#!/usr/bin/env bash
# Execute in the GPU terminal; safe to resume preflight/unfinished episodes.
set -euo pipefail
cd /media/fulei/fuleivla/ecm_vla
run_dir="$PWD/results/M1_acceptance_20261023"
mkdir -p "$run_dir"
nohup python3 -u scripts/run_milestone1_acceptance.py --run "$run_dir" \
  >>"$run_dir/pipeline.log" 2>&1 </dev/null &
pipeline_pid=$!
printf 'M1 pipeline PID: %s\nLog: %s/pipeline.log\n' "$pipeline_pid" "$run_dir"
printf '查看日志：tail -f %s/pipeline.log\n' "$run_dir"
printf '验收结果：%s/acceptance.json\n' "$run_dir"
