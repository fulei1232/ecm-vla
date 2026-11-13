#!/usr/bin/env bash
# Smoke tests of the fruit DART study before its seal (reports/fruit_dart_protocol_20261027.md, section 9):
# training (100 steps, all three arms, four GPUs per arm) -> T1; closed loop (smoke checkpoints, val episode 50,
# seed 7, K0/K4/K8 + C0 split over two servers); development gate (--smoke, dev episode 200).
# GPU memory is sampled every 5 s per step into logs/fruit_dart_smoke_gpu_<step>.csv; timings in logs/fruit_dart_smoke_times.txt.
#   bash scripts/run_fruit_dart_smoke.sh
set -u
cd "$(dirname "$0")/.."
RUN=runs/fruit_dart_kf_smoke_20261027
EVAL=results/fruit_dart_kf_eval_smoke_20261027
DEV=results/fruit_dart_dev_gate_smoke_20261027
TIMES=logs/fruit_dart_smoke_times.txt

step() {
    local name=$1
    shift
    nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 5 \
        > "logs/fruit_dart_smoke_gpu_$name.csv" &
    local logger=$! start=$(date +%s)
    "$@"
    local code=$?
    kill "$logger"
    echo "$name exit=$code seconds=$(( $(date +%s) - start ))" | tee -a "$TIMES"
    return $code
}

step train bash -c "python3 scripts/run_finetune_kf.py prepare --study fruit_dart --smoke --out $RUN \
    --dataset data/finetune_kf/fruit_dart_rm75b_train && python3 scripts/run_finetune_kf.py train --out $RUN" || exit 1
python3 scripts/run_finetune_kf.py gate --out "$RUN" || exit 1
step closed_loop python3 scripts/run_teacher_oracle.py all --smoke --stage fruit_dart --out "$EVAL" \
    --checkpoint-run "$RUN" --checkpoint-step 99 || exit 1
step dev_gate python3 scripts/run_fruit_dev_gate.py --smoke --out "$DEV" --checkpoint-run "$RUN" --checkpoint-step 99
echo "smoke finished"
