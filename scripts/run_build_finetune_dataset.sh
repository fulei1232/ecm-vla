#!/usr/bin/env bash
# Full fine-tuning dataset: plan, one shard per GPU, finish. Resumable (rerun the same command).
#   nohup bash scripts/run_build_finetune_dataset.sh data/finetune_kf/robomme3_train_gt > logs/build_ft.log 2>&1 &
set -euo pipefail
ECM=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(realpath -m "${1:?output directory}")
WS=$(dirname "$ECM")
POL=$WS/acm_vla/vendor/robomme_policy_learning
PY=$WS/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python
export PYTHONPATH=$POL/src:$POL/packages/openpi-client/src OPENPI_DATA_HOME=$WS/acm_vla/cache/openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
SHARDS=${SHARDS:-4}
cd "$POL"
[ -f "$OUT/meta/episodes.json" ] || "$PY" "$ECM/scripts/build_finetune_dataset.py" plan --out "$OUT"
mkdir -p "$OUT/logs"
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$((i % 4)) "$PY" "$ECM/scripts/build_finetune_dataset.py" shard --out "$OUT" --index "$i" --of "$SHARDS" \
    > "$OUT/logs/shard$i.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
[ $status -eq 0 ] || { echo "a shard failed; see $OUT/logs"; exit 1; }
"$PY" "$ECM/scripts/build_finetune_dataset.py" finish --out "$OUT"
CUDA_VISIBLE_DEVICES=0 "$PY" "$ECM/scripts/build_finetune_dataset.py" check-encoder --out "$OUT" \
  --checkpoint "$WS/adaptive_evidence_acquisition_20261009/models/perceptual-framesamp-modul/79999" --frames 8
