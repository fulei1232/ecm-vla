#!/usr/bin/env bash
# Run in the user's GPU terminal. The Python pipeline locks the run and can resume it.
set -euo pipefail
cd /media/fulei/fuleivla/ecm_vla
run_dir="$PWD/results/V_A_formal_20261023"
test -f "$run_dir/protocol.json"
nohup python3 -u scripts/complete_v_a.py --run "$run_dir" \
  >>"$run_dir/pipeline.log" 2>&1 </dev/null &
pipeline_pid=$!
printf 'V-A pipeline PID: %s\nLog: %s/pipeline.log\n' "$pipeline_pid" "$run_dir"
printf '查看日志：tail -f %s/pipeline.log\n' "$run_dir"
