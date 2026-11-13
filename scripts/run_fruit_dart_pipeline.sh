#!/usr/bin/env bash
# Fruit DART study (reports/fruit_dart_protocol_20261027.md): k0 and k4 in parallel -> gate T1 on them ->
# development gate D -> k8 -> gate T1 on all arms -> closed-loop evaluation (stage fruit_dart, val 50-99).
# Usage (after `run_finetune_kf.py prepare --study fruit_dart`):
#   bash scripts/run_fruit_dart_pipeline.sh RUN_DIR DEV_DIR EVAL_DIR
# Unfinished arms are retrained once (restart rule). Exit 1: a T1 gate failed; exit 2: gate D failed (k8 not trained).
set -u
RUN=$1
DEV=$2
EVAL=$3
cd "$(dirname "$0")/.."

train() {
    python3 scripts/run_finetune_kf.py train --out "$RUN" --arms "$@" ||
        python3 scripts/run_finetune_kf.py train --out "$RUN" --arms "$@"
}

train k0 k4
if ! python3 scripts/run_finetune_kf.py gate --out "$RUN" --arms k0 k4; then
    echo "gate T1 (k0, k4) failed; stopping"
    exit 1
fi
if ! python3 scripts/run_fruit_dev_gate.py --out "$DEV"; then
    echo "development gate D failed; k8 is not trained and there is no formal evaluation"
    exit 2
fi
train k8
if ! python3 scripts/run_finetune_kf.py gate --out "$RUN"; then
    echo "gate T1 failed; no closed-loop evaluation"
    exit 1
fi
python3 scripts/run_teacher_oracle.py all --stage fruit_dart --out "$EVAL"
