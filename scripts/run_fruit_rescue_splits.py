#!/usr/bin/env python3
"""Resumable stage-0 legacy-generator splits; freeze only after all 180 manifests exist.

Run with nohup setsid. This script never evaluates a policy or summarizes val layouts.
Existing manifests are hash checked and never regenerated. Failed runs remain in status.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from v_a_common import ECM, SIM_PYTHON, atomic_json, timestamp

REGISTRY = ECM / "reports/split_registry_20261102.json"
SPLITS = (("dev_gate", "train", 220, 260, ECM / "data/fruit_picking_dev"),
          ("sealed_val", "val", 100, 150, ECM / "data/fruit_picking"))
TASKS = ("count", "occluded_ripe")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def inventory():
    rows, missing = [], []
    for label, split, first, stop, root in SPLITS:
        for task in TASKS:
            for episode in range(first, stop):
                path = root / "rm75b_eg2" / split / task / f"ep{episode}_manifest.json"
                if not path.is_file():
                    missing.append(str(path.relative_to(ECM)))
                    continue
                man = json.loads(path.read_text())
                if (man["task"], man["split"], man["episode"], man["robot"]) != (task, split, episode, "rm75b_eg2"):
                    raise ValueError(f"manifest identity mismatch: {path}")
                base = 100000 if split == "train" else 200000
                if man["seed"] != base + episode * 16 + man["attempt"] or not 0 <= man["attempt"] < 16:
                    raise ValueError(f"manifest seed mismatch: {path}")
                for key in ("h5", "gt"):
                    if digest(path.parent / man[key]) != man[f"{key}_sha256"]:
                        raise ValueError(f"payload hash mismatch: {path} {key}")
                rows.append({"split": label, "task": task, "episode": episode, "seed": man["seed"],
                             "manifest": str(path.relative_to(ECM)), "sha256": digest(path),
                             "h5_sha256": man["h5_sha256"], "gt_sha256": man["gt_sha256"]})
    return rows, missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    a = p.parse_args()
    if len(set(a.gpus)) != len(a.gpus) or any(g < 0 or g > 7 for g in a.gpus):
        p.error("gpus must be unique indexes in 0-7")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("split generation is already running")
    sources = {str(path.relative_to(ECM)): digest(path) for path in sorted((ECM / "fruit_picking").glob("*.py"))}
    plan = {"phase": 0, "solver": "legacy", "source_sha256": sources, "expected_manifests": 180,
            "sim_python": str(SIM_PYTHON), "gpus": a.gpus}
    plan_path = out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        p.error("generation plan or legacy generator changed; refusing to resume")
    atomic_json(plan_path, plan)
    prior = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    attempts = prior.get("attempts", [])
    initial_attempt_count = len(attempts)
    registry_old = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}

    def save_registry(rows, missing):
        content = {"schema": "fruit-split-registry-v1", "updated_utc": timestamp(),
                   "frozen": not missing, "expected_manifests": 180, "present_manifests": len(rows),
                   "generator": plan, "manifests": rows, "missing": missing,
                   "rules": {"dev_gate": "220-259; once per candidate; no training",
                             "sealed_val": "100-149; only phase 5; do not inspect or evaluate before then"}}
        if not missing:
            content["manifest_list_sha256"] = hashlib.sha256(json.dumps(rows, sort_keys=True,
                                                                         separators=(",", ":")).encode()).hexdigest()
            content["frozen_utc"] = timestamp()
        atomic_json(REGISTRY, content)

    rows, missing = inventory()
    if registry_old.get("frozen"):
        if missing or rows != registry_old["manifests"]:
            p.error("frozen split changed; refusing regeneration")
        print("Frozen registry verified; no generation performed", flush=True)
        return 0
    save_registry(rows, missing)
    missing_set = set(missing)
    jobs = []
    for label, split, first, stop, root in SPLITS:
        for task in TASKS:
            for start in range(first, stop, 5):
                eps = [e for e in range(start, min(start+5, stop)) if
                       str((root / 'rm75b_eg2' / split / task / f'ep{e}_manifest.json').relative_to(ECM)) in missing_set]
                if eps:
                    jobs.append((label, split, task, eps, root))

    def run(job, gpu):
        label, split, task, eps, root = job
        name = f"{label}_{task}_{eps[0]}_{eps[-1]}_attempt{len(attempts)}"
        cmd = [str(SIM_PYTHON), "-u", "-m", "fruit_picking.generate", "--task", task,
               "--split", split, "--episodes", ",".join(map(str, eps)), "--robot", "rm75b_eg2", "--out", str(root)]
        log = out / f"{name}.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
               "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
               "MPLCONFIGDIR": str(ECM / "cache/matplotlib"), "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        with log.open("a") as f:
            code = subprocess.run(cmd, cwd=ECM, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        return {"command": cmd, "gpu": gpu, "exit_code": code, "log": str(log), "finished_utc": timestamp()}

    atomic_json(out / "status.json", {"phase": 0, "state": "running", "attempts": attempts,
                                       "updated_utc": timestamp(), "pid": os.getpid()})
    # Smoke with first pending chunk before launching 8 processes; failed setup stops the batch.
    if jobs:
        smoke = run(jobs.pop(0), a.gpus[0])
        attempts.append(smoke)
        if smoke["exit_code"] == 0:
            def lane(gpu, lane_jobs):
                results = []
                for job in lane_jobs:
                    result = run(job, gpu)
                    results.append(result)
                    if result["exit_code"] != 0:
                        break
                return results
            for split_label in ("dev_gate", "sealed_val"):
                split_jobs = [job for job in jobs if job[0] == split_label]
                with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
                    futures = [pool.submit(lane, gpu, split_jobs[i::len(a.gpus)]) for i, gpu in enumerate(a.gpus)]
                    for future in as_completed(futures):
                        attempts.extend(future.result())
                        atomic_json(out / "status.json", {"phase": 0, "state": "running", "attempts": attempts,
                                                           "updated_utc": timestamp(), "pid": os.getpid()})
                if any(attempt["exit_code"] != 0 for attempt in attempts[initial_attempt_count:]):
                    break
    rows, missing = inventory()
    save_registry(rows, missing)
    state = {"phase": 0, "state": "blocked" if missing else "complete", "updated_utc": timestamp(),
             "present_manifests": len(rows), "expected_manifests": 180, "attempts": attempts,
             "registry": str(REGISTRY)}
    atomic_json(out / "status.json", state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return int(bool(missing))


if __name__ == "__main__":
    sys.exit(main())
