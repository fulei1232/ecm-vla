#!/usr/bin/env bash
# Fruit-picking event-keyframe fine-tuning and its closed-loop evaluation (reports/fruit_finetune_kf_protocol_20261025.md).
# Usage (after `run_finetune_kf.py prepare --study fruit`): bash scripts/run_fruit_finetune_pipeline.sh RUN_DIR EVAL_DIR
# Unfinished arms are retrained once (restart rule); the evaluation starts only if gate T1 passes.
set -u
RUN=$1
EVAL=$2
cd "$(dirname "$0")/.."

python3 scripts/run_finetune_kf.py train --out "$RUN" || python3 scripts/run_finetune_kf.py train --out "$RUN"
if ! python3 scripts/run_finetune_kf.py gate --out "$RUN"; then
    echo "gate T1 failed; no closed-loop evaluation"
    exit 1
fi
python3 scripts/run_teacher_oracle.py all --stage fruit --out "$EVAL"
