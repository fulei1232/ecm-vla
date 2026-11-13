#!/usr/bin/env bash
# Fruit-picking fine-tuning dataset: plan, one shard per GPU, finish, norm stats, encoder check. Resumable.
#   nohup bash scripts/run_build_fruit_dataset.sh data/finetune_kf/fruit_rm75b_train > logs/build_fruit.log 2>&1 &
# EPISODES="0 2" restricts the train episodes (smoke); GPUS="3" restricts the shards' GPUs.
set -euo pipefail
ECM=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(realpath -m "${1:?output directory}")
WS=$(dirname "$ECM")
POL=$WS/acm_vla/vendor/robomme_policy_learning
PY=$WS/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python
export PYTHONPATH=$POL/src:$POL/packages/openpi-client/src OPENPI_DATA_HOME=$WS/acm_vla/cache/openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
EPISODES=${EPISODES:-0 100}
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3}"
SHARDS=${#GPU_LIST[@]}
BUILD="$ECM/scripts/build_fruit_dataset.py"
cd "$POL"
# shellcheck disable=SC2086
[ -f "$OUT/meta/episodes.json" ] || JAX_PLATFORMS=cpu "$PY" "$BUILD" plan --out "$OUT" --episodes $EPISODES
mkdir -p "$OUT/logs"
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} "$PY" "$BUILD" shard --out "$OUT" --index "$i" --of "$SHARDS" \
    > "$OUT/logs/shard$i.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
[ $status -eq 0 ] || { echo "a shard failed; see $OUT/logs"; exit 1; }
"$PY" "$BUILD" finish --out "$OUT"
[ -f "$OUT/meta/norm_stats.json" ] || JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" "$PY" "$BUILD" norm-stats --out "$OUT"
CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} "$PY" "$BUILD" check-encoder --out "$OUT" \
  --checkpoint "$WS/adaptive_evidence_acquisition_20261009/models/perceptual-framesamp-modul/79999" --frames 8
