#!/usr/bin/env bash
# One-off chain: once the RoboMME fine-tuning pipeline (pid $1) has exited with its evaluation completed, merge
# the fruit stage into main, run the fruit training smoke and closed-loop smoke, append their record to the
# protocol (append_fruit_smoke_record.py, which refuses unless every check passed), commit it, seal the formal
# fruit run and start its pipeline. Stops at the first failed step; nothing formal starts after a failure.
set -euo pipefail
WAIT_PID=$1
ECM=/media/fulei/fuleivla/ecm_vla
DATASET=data/finetune_kf/fruit_rm75b_train
SMOKE=runs/fruit_finetune_kf_smoke_20261026
EVAL_SMOKE=results/fruit_finetune_kf_eval_smoke_20261026
RUN=runs/fruit_finetune_kf_20261025
EVAL=results/fruit_finetune_kf_eval_20261025
DOC=reports/fruit_finetune_kf_protocol_20261025.md
GPU_LOG=logs/fruit_eval_smoke_gpu_20261026.csv
step() { echo "[$(date '+%F %T')] $*"; }
trap 'step "chain ended (exit $?)"; [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null; true' EXIT
cd "$ECM"

while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
step "RoboMME pipeline exited"
python3 - <<'EOF'
import json, os, sys
d = "results/finetune_kf_eval_20261025"
p = json.load(open(f"{d}/progress.json"))
print("RoboMME evaluation:", p["status"], p["complete_blocks"], "/", p["planned_blocks"])
sys.exit(0 if p["status"] == "completed" and os.path.exists(f"{d}/REPORT.md") else 1)
EOF
git merge --ff-only fruit-stage
step "merged: $(git log --oneline -1)"

python3 scripts/run_finetune_kf.py prepare --study fruit --smoke --out "$SMOKE" --dataset "$DATASET"
t0=$(date +%s)
python3 scripts/run_finetune_kf.py train --out "$SMOKE"
python3 scripts/run_finetune_kf.py gate --out "$SMOKE"
t1=$(date +%s)
step "training smoke passed T1"

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 10 > "$GPU_LOG" &
SAMPLER=$!
python3 scripts/run_teacher_oracle.py all --smoke --stage fruit --checkpoint-run "$SMOKE" --checkpoint-step 99 \
    --out "$EVAL_SMOKE"
kill "$SAMPLER"; SAMPLER=
t2=$(date +%s)
step "closed-loop smoke finished"

python3 scripts/append_fruit_smoke_record.py --doc "$DOC" --train-run "$SMOKE" --eval-run "$EVAL_SMOKE" \
    --gpu-log "$GPU_LOG" --train-seconds $((t1 - t0)) --eval-seconds $((t2 - t1))
git add "$DOC"
git commit -q -m "Fruit fine-tuning protocol: smoke records 4-6; sealed for the formal run"
step "committed: $(git log --oneline -1)"

python3 scripts/run_finetune_kf.py prepare --study fruit --out "$RUN" --dataset "$DATASET"
step "sealed $RUN"
nohup setsid bash scripts/run_fruit_finetune_pipeline.sh "$RUN" "$EVAL" > logs/fruit_finetune_pipeline_20261026.log 2>&1 &
step "formal fruit pipeline started (pid $!): training ~18-19 h, then the closed-loop evaluation"
