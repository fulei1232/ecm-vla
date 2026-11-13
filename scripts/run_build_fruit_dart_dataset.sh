#!/usr/bin/env bash
# Build the clean 0-99 + DART 100-199 fruit dataset for the fruit_dart study.
# Run from a GPU-enabled terminal; resumable at the episode level.
set -euo pipefail
ECM=$(cd "$(dirname "$0")/.." && pwd)
WS=$(dirname "$ECM")
POL="$WS/acm_vla/vendor/robomme_policy_learning"
PY="$WS/adaptive_evidence_acquisition_20261009/vendor/robomme_policy_learning/.venv/bin/python"
OUT="$ECM/data/finetune_kf/fruit_dart_rm75b_train"
BUILD="$ECM/scripts/build_fruit_dataset.py"
CLEAN="$ECM/data/fruit_picking/rm75b_eg2/train"
DART="$ECM/data/fruit_picking_dart/rm75b_eg2/train"
CHECKPOINT="$WS/adaptive_evidence_acquisition_20261009/models/perceptual-framesamp-modul/79999"
test -x "$PY"
test -d "$CHECKPOINT"
mkdir -p "$OUT/logs" "$OUT/meta"
exec 9>"$OUT/meta/build.lock"
flock -n 9 || { echo "Another DART dataset build is running" >&2; exit 1; }
export PYTHONPATH="$POL/src:$POL/packages/openpi-client/src" OPENPI_DATA_HOME="$WS/acm_vla/cache/openpi"
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$POL"

if [ ! -f "$OUT/meta/episodes.json" ]; then
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "$PY" -u "$BUILD" plan --out "$OUT" \
    --part "$CLEAN" 0 100 --part "$DART" 100 200 > "$OUT/logs/plan.log" 2>&1
fi
"$PY" - "$OUT/meta/episodes.json" "$CLEAN" "$DART" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
expected = [(task, source, episode) for task in ('FruitCount', 'FruitOccludedRipe')
            for source, interval in ((sys.argv[2], range(100)), (sys.argv[3], range(100, 200)))
            for episode in interval]
actual = [(r['task'], r['source'], r['train_episode']) for r in rows]
assert actual == expected, 'Existing episode plan does not match clean 0-99 + DART 100-199'
print('Plan verified: 400 episodes, correct sources and order', flush=True)
PY

pids=()
for index in {0..7}; do
  CUDA_VISIBLE_DEVICES="$index" "$PY" -u "$BUILD" shard --out "$OUT" --index "$index" --of 8 \
    > "$OUT/logs/shard${index}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
  echo "Dataset shard failed; inspect $OUT/logs/shard*.log" >&2
  exit 1
fi
"$PY" -u "$BUILD" finish --out "$OUT" | tee "$OUT/logs/finish.log"
if [ ! -f "$OUT/meta/norm_stats.json" ]; then
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "$PY" -u "$BUILD" norm-stats --out "$OUT" \
    > "$OUT/logs/norm_stats.log" 2>&1
fi
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$BUILD" check-encoder --out "$OUT" \
  --checkpoint "$CHECKPOINT" --frames 8 | tee "$OUT/logs/encoder_check.log"
"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
build = json.loads((out / 'meta/build.json').read_text())
assert build['passed'] and build['episodes'] == 400, build.get('problems')
assert (out / 'assets/robomme/norm_stats.json').is_file()
encoder = json.loads((out / 'meta/encoder_check.json').read_text())
assert encoder['frames'] == 8 and encoder['shapes'][0] == encoder['shapes'][1], encoder
print('DART dataset build complete:', build['execution_samples'], 'execution samples; encoder mean absolute diff:', encoder['mean_abs_diff'])
PY
