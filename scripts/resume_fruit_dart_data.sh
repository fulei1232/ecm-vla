#!/usr/bin/env bash
# Resume the agreed DART/dev/new-val ranges. Does not start training.
set -euo pipefail
ECM=$(cd "$(dirname "$0")/.." && pwd)
WS=$(dirname "$ECM")
PY="$WS/robomme_benchmark/.venv/bin/python"
MODE=${1:-run}
case "$MODE" in run|--dry-run|--check-only) ;; *) echo "Usage: bash $0 [--dry-run|--check-only]" >&2; exit 2 ;; esac
cd "$ECM"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ECM:$ECM/src:$WS/robomme_benchmark/src"
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}

verify_data() {
  "$PY" - <<'PY'
import json
from pathlib import Path
ok = True
for dataset, split, start, stop in [
    ('fruit_picking_dart', 'train', 100, 200),
    ('fruit_picking_dev', 'train', 200, 220),
    ('fruit_picking', 'val', 50, 100),
]:
    for task in ['count', 'occluded_ripe']:
        folder = Path('data') / dataset / 'rm75b_eg2' / split / task
        missing = []
        invalid = []
        for episode in range(start, stop):
            manifest = folder / f'ep{episode}_manifest.json'
            if not manifest.is_file():
                missing.append(episode)
                continue
            try:
                row = json.loads(manifest.read_text())
                assert row['episode'] == episode and row['task'] == task and row['split'] == split
                assert row['seed'] is not None and row['verification']['final_success'] is True
                assert row['verification']['n_timesteps'] > 0
                assert (folder / row['h5']).is_file() and (folder / row['gt']).is_file()
            except (ValueError, KeyError, TypeError, OSError, AssertionError):
                invalid.append(episode)
        count = stop - start - len(missing) - len(invalid)
        print(f'{dataset}/{split}/{task}: {count}/{stop-start}; missing={missing}; invalid={invalid}')
        ok = ok and not missing and not invalid
print('PASS: all requested episodes present and manifests valid' if ok else 'INCOMPLETE: see missing/invalid episodes and worker logs')
raise SystemExit(0 if ok else 1)
PY
}

if [[ "$MODE" == --check-only ]]; then
  verify_data
  exit 0
fi

LOG_ROOT="$ECM/data/fruit_picking/logs"
if [[ "$MODE" == run ]]; then
  test -x "$PY"
  test -f "$VK_ICD_FILENAMES"
  mkdir -p "$LOG_ROOT"
  exec 9>"$LOG_ROOT/.resume_fruit_dart_data.lock"
  flock -n 9 || { echo "Another data-resume launcher is running." >&2; exit 1; }
  nvidia-smi --query-gpu=index,name,memory.used --format=csv
  LOG_DIR=$(mktemp -d "$LOG_ROOT/resume_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
  printf 'Worker logs: %s\n' "$LOG_DIR"
fi

pids=()
status=0
launch() {
  local gpu=$1 name=$2
  shift 2
  if [[ "$MODE" == --dry-run ]]; then
    printf 'GPU %s | %s | ' "$gpu" "$name"
    printf '%q ' "$PY" -u "$@"
    printf '\n'
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$@" >"$LOG_DIR/$name.log" 2>&1 &
    pids+=("$!")
    printf 'Started %s: GPU %s, PID %s\n' "$name" "$gpu" "$!"
  fi
}
wait_all() {
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
  done
  pids=()
}

i=0
for task in count occluded_ripe; do
  for episodes in 100-124 125-149 150-174 175-199; do
    launch "$i" "dart_${task}_${episodes}" scripts/generate_fruit_dart.py \
      --task "$task" --episodes "$episodes" --sigma 0.04 --tau 10 --out data/fruit_picking_dart
    i=$((i + 1))
  done
done
wait_all

i=0
for task in count occluded_ripe; do
  launch "$i" "dev_${task}_200-219" -m fruit_picking.generate \
    --task "$task" --split train --episodes 200-219 --out data/fruit_picking_dev
  i=$((i + 1))
  for episodes in 50-74 75-99; do
    launch "$i" "val_${task}_${episodes}" -m fruit_picking.generate \
      --task "$task" --split val --episodes "$episodes" --out data/fruit_picking
    i=$((i + 1))
  done
done
wait_all

if [[ "$MODE" == run ]]; then
  if ! verify_data >"$LOG_DIR/verification.log" 2>&1; then status=1; fi
  cat "$LOG_DIR/verification.log"
  printf '%s\n' "$status" >"$LOG_DIR/exit_code.txt"
  printf 'Finished with exit code %s. Logs: %s\n' "$status" "$LOG_DIR"
fi
exit "$status"
