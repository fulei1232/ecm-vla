#!/usr/bin/env python3
"""Generate RoboMME demonstrations (default: train split) with a per-timestep simulator ground-truth sidecar.

Runs exactly the episodes of generate_val_demos.run_episode (official planner, failure recovery off) and
additionally snapshots the simulator ground truth (acm_copy.diagnostic_logging.snapshot: TCP position,
buttons / cubes / bins, task phase, ...) after every recorded timestep. The sidecar
`gt/<Task>_ep<E>_seed<S>.jsonl` has one row per HDF5 timestep, in order (row i <-> timestep_i), so the
event distance of every training frame can be computed offline (event_memory.gt_distance).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<SIM>/src \
        /media/fulei/fuleivla/robomme_benchmark/.venv/bin/python ecm_vla/scripts/generate_train_demos_gt.py \
        --task BinFill --episodes 0 1 2
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_val_demos as gvd  # noqa: E402

ECM = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "diagnostic_logging", ECM / "src/memory_management/validation/acm_copy/diagnostic_logging.py")
diagnostic_logging = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diagnostic_logging)

GT_ROWS: list[dict] = []
_original_step = gvd.RobommeRecordWrapper.step


def recording_step(self, action):
    before = len(self.buffer)
    result = _original_step(self, action)
    added = len(self.buffer) - before
    if added not in (0, 1):
        raise RuntimeError(f"RecordWrapper buffered {added} timesteps in one step")
    if added:
        GT_ROWS.append({"timestep": before, "gt": diagnostic_logging.snapshot(self.unwrapped, result[4])})
    return result


gvd.RobommeRecordWrapper.step = recording_step


def check_alignment(h5_path: Path, episode: int, rows: list[dict]) -> int:
    with h5py.File(h5_path, "r") as f:
        n = sum(1 for k in f[f"episode_{episode}"].keys() if k.startswith("timestep_"))
    if [r["timestep"] for r in rows] != list(range(n)):
        raise RuntimeError(f"ground-truth rows {len(rows)} do not match the {n} HDF5 timesteps")
    return n


def atomic_write(path: Path, text: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text)
    os.replace(temp, path)


def finished(manifest: Path) -> bool:
    try:
        return json.loads(manifest.read_text()).get("used_seed") is not None
    except (OSError, ValueError):
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--split", default="train", choices=["val", "train"])
    p.add_argument("--max-seed-attempts", type=int, default=5)
    p.add_argument("--out", type=Path, default=ECM / "data/train_demos_gt")
    opts = p.parse_args()
    builder = gvd.BenchmarkEnvBuilder(env_id=opts.task, dataset=opts.split)
    out = opts.out / opts.task
    (out / "gt").mkdir(parents=True, exist_ok=True)
    status = 0
    for episode in opts.episodes:
        manifest = out / f"ep{episode:02d}_manifest.json"
        if finished(manifest):
            continue
        seed0, difficulty = builder.resolve_episode(episode)
        assert seed0 is not None, f"no metadata seed for {opts.task} {opts.split} episode {episode}"
        record = {"task": opts.task, "split": opts.split, "episode": episode, "metadata_seed": seed0,
                  "difficulty": difficulty, "failure_recovery": False, "attempts": []}
        for attempt in range(opts.max_seed_attempts):
            seed = seed0 + attempt
            start = time.monotonic()
            GT_ROWS.clear()
            try:
                ok, error = gvd.run_episode(opts.task, episode, seed, difficulty, out), None
            except Exception:
                ok, error = False, traceback.format_exc()
            record["attempts"].append({"seed": seed, "success": ok, "seconds": round(time.monotonic() - start, 1),
                                       "error": error})
            if ok:
                h5 = out / "hdf5_files" / f"{opts.task}_ep{episode}_seed{seed}.h5"
                gt_path = out / "gt" / f"{opts.task}_ep{episode}_seed{seed}.jsonl"
                n = check_alignment(h5, episode, GT_ROWS)
                atomic_write(gt_path, "".join(json.dumps(r) + "\n" for r in GT_ROWS))
                record.update(used_seed=seed, h5_path=str(h5), h5_sha256=hashlib.sha256(h5.read_bytes()).hexdigest(),
                              gt_path=str(gt_path), gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                              gt_rows=n, verification=gvd.verify_h5(h5, episode))
                break
        else:
            status = 1
        atomic_write(manifest, json.dumps(record, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in record.items() if k not in ("attempts", "verification")}
                         | {"n_attempts": len(record["attempts"])}, ensure_ascii=False), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
