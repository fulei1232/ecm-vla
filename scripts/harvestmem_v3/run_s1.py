#!/usr/bin/env python3
"""Resumable S1 expert replay, relabel, manifest, and dataset build driver."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from v_a_common import (ECM, OPENPI_DATA_HOME, POLICY, POLICY_PYTHON, SIM_PYTHON,
                        atomic_json, read_json, sha256, timestamp)


TASKS = ("count", "occluded_ripe")
SIM_RUNNER = ECM / "scripts/harvestmem_v3/run_episode.py"
RELABEL = ECM / "scripts/harvestmem_v3/relabel_episode.py"
S1 = ECM / "results/harvestmem_v3_S1"
DATA_CLEAN = ECM / "data/harvestmem_v3_clean_20261110/rm75b_eg2/train"
DATA_DART = ECM / "data/harvestmem_v3_dart_20261110/rm75b_eg2/train"


def run_child(command: list[str], out: Path, gpu: int | None = None, cwd: Path = ECM) -> None:
    policy_env = cwd.resolve() == POLICY.resolve()
    env = {
        **os.environ,
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": ":".join((
            str(POLICY / "src"), str(POLICY / "packages/openpi-client/src"),
            str(ECM / "scripts"), str(ECM), str(ECM / "src"),
        ) if policy_env else (str(ECM / "scripts"), str(ECM), str(ECM / "src"))),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if policy_env:
        env["OPENPI_DATA_HOME"] = str(OPENPI_DATA_HOME)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "harvestmem_command.log"
    with log.open("a") as stream:
        stream.write(json.dumps({"command": command, "gpu": gpu, "started_utc": timestamp()}) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"child failed ({result.returncode}): {out}")


def relabel_and_audit(out: Path) -> dict:
    run_child([str(SIM_PYTHON), "-u", str(RELABEL), "--dir", str(out)], out)
    result = read_json(out / "result.json")
    audit = read_json(out / "harvestmem_subgoal_audit.json")
    if not result.get("success") or not audit.get("passed"):
        raise RuntimeError(f"unsuccessful or invalid replay: {out}")
    if audit.get("transition_mismatches") or audit.get("wait_for_green_frames"):
        raise RuntimeError(f"S1 label audit failed: {out}")
    return {**result, "audit": audit, "source": str(out)}


def attempt(task: str, episode: int, purpose: str, attempt_id: int, gpu: int,
            layout_seed: int | None = None, dart: bool = False) -> dict:
    out = S1 / "replays" / purpose / task / f"ep{episode}_attempt{attempt_id}"
    if (out / "result.json").exists() and (out / "harvestmem_subgoal_audit.json").exists():
        return relabel_and_audit(out)
    command = [str(SIM_PYTHON), "-u", str(SIM_RUNNER), "--task", task,
               "--episode", str(episode), "--purpose", purpose,
               "--attempt", str(attempt_id), "--out", str(out),
               "--results-root", str(S1)]
    if layout_seed is not None:
        command += ["--layout-seed", str(layout_seed)]
    if dart:
        command.append("--dart-two-cycles")
    run_child(command, out, gpu=gpu)
    return relabel_and_audit(out)


def run_one(task: str, episode: int, purpose: str, gpu: int, paired_layout: int | None = None,
            dart: bool = False, max_attempts: int = 8) -> dict:
    failures = []
    for attempt_id in range(max_attempts):
        try:
            row = attempt(task, episode, purpose, attempt_id, gpu, paired_layout, dart)
            row["failed_attempts"] = failures
            return row
        except Exception as exc:
            failures.append({"attempt": attempt_id, "error": repr(exc), "utc": timestamp()})
    raise RuntimeError(json.dumps({"task": task, "episode": episode, "purpose": purpose,
                                   "failures": failures}, ensure_ascii=False))


def save_manifest(row: dict, root: Path, episode: int, task: str, purpose: str) -> dict:
    root = root / task
    root.mkdir(parents=True, exist_ok=True)
    out = Path(row["source"])
    label = "FruitCount" if task == "count" else "FruitOccludedRipe"
    h5_link = root / "hdf5_files" / f"{label}_ep{episode}_seed{row['seed']}.h5"
    gt_link = root / "gt" / f"{label}_ep{episode}_seed{row['seed']}.jsonl"
    for link, target in ((h5_link, out / "demo.h5"), (gt_link, out / "demo_gt.jsonl")):
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(os.path.relpath(target, link.parent))
        if sha256(link) != sha256(target):
            raise RuntimeError(f"link hash mismatch: {link}")
    manifest = {
        **row,
        "task": task,
        "purpose": purpose,
        "episode": episode,
        "split": "dev" if purpose == "dev_expert" else "train",
        "h5": str(h5_link.relative_to(root)),
        "gt": str(gt_link.relative_to(root)),
        "h5_sha256": sha256(h5_link),
        "gt_sha256": sha256(gt_link),
        "verification": {
            "passed": True,
            "audit": str(out / "harvestmem_subgoal_audit.json"),
            "audit_sha256": sha256(out / "harvestmem_subgoal_audit.json"),
        },
    }
    atomic_json(root / f"ep{episode}_manifest.json", manifest)
    return manifest


def parallel_jobs(jobs: list[dict], worker_fn, workers: int = 8) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker_fn, job, i % workers) for i, job in enumerate(jobs)]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def run_dev() -> None:
    jobs = [{"task": task, "episode": episode, "purpose": "dev_expert"}
            for task in TASKS for episode in range(200, 250)]

    def one(job, gpu):
        manifest_path = S1 / "dev" / job["task"] / f"ep{job['episode']}_manifest.json"
        if manifest_path.exists():
            return read_json(manifest_path)
        row = run_one(job["task"], job["episode"], job["purpose"], gpu)
        return save_manifest(row, S1 / "dev", job["episode"], job["task"], job["purpose"])

    rows = parallel_jobs(jobs, one)
    summary = {"purpose": "dev_expert", "episodes": len(rows), "by_task": {}}
    for task in TASKS:
        subset = [r for r in rows if r["task"] == task]
        summary["by_task"][task] = {
            "episodes": len(subset),
            "successes": sum(bool(r["success"]) for r in subset),
            "success_rate": sum(bool(r["success"]) for r in subset) / len(subset),
            "failed_attempts": sum(len(r.get("failed_attempts", [])) for r in subset),
        }
    summary["passed_45_of_50_each_task"] = all(v["successes"] >= 45 for v in summary["by_task"].values())
    atomic_json(S1 / "dev_summary.json", summary)
    if not summary["passed_45_of_50_each_task"]:
        raise RuntimeError(f"S1 dev gate failed: {summary}")


def run_train() -> None:
    clean_jobs = [{"task": task, "episode": episode, "purpose": "train_clean"}
                  for task in TASKS for episode in range(100)]

    def clean_one(job, gpu):
        manifest_path = DATA_CLEAN / job["task"] / f"ep{job['episode']}_manifest.json"
        if manifest_path.exists():
            return read_json(manifest_path)
        row = run_one(job["task"], job["episode"], job["purpose"], gpu)
        return save_manifest(row, DATA_CLEAN, job["episode"], job["task"], job["purpose"])

    parallel_jobs(clean_jobs, clean_one)

    dart_jobs = [{"task": task, "episode": 100 + episode, "purpose": "train_dart"}
                 for task in TASKS for episode in range(100)]

    def dart_one(job, gpu):
        manifest_path = DATA_DART / job["task"] / f"ep{job['episode']}_manifest.json"
        if manifest_path.exists():
            return read_json(manifest_path)
        clean_manifest = read_json(DATA_CLEAN / job["task"] / f"ep{job['episode'] - 100}_manifest.json")
        row = run_one(job["task"], job["episode"], job["purpose"], gpu,
                      paired_layout=int(clean_manifest["r12_layout_seed"]), dart=True)
        if row["N"] != clean_manifest["N"] or row["r12_layout_sha256"] != clean_manifest["r12_layout_sha256"]:
            raise RuntimeError(f"DART layout pairing mismatch: {job}")
        return save_manifest(row, DATA_DART, job["episode"], job["task"], job["purpose"])

    parallel_jobs(dart_jobs, dart_one)


def build_dataset() -> None:
    dataset = ECM / "data/harvestmem_v3_dataset_20261110"
    if dataset.exists() and not (dataset / "meta/episodes.json").exists():
        raise RuntimeError(f"incomplete dataset plan at {dataset}")
    builder = ECM / "scripts/build_fruit_dataset.py"
    commands = [
        [str(POLICY_PYTHON), str(builder), "plan", "--out", str(dataset),
         "--part", str(DATA_CLEAN), "0", "100", "--part", str(DATA_DART), "100", "200",
         "--distance-signal", "any_hanging"],
        [str(POLICY_PYTHON), str(builder), "finish", "--out", str(dataset)],
        [str(POLICY_PYTHON), str(builder), "norm-stats", "--out", str(dataset)],
    ]
    if not (dataset / "meta/episodes.json").exists():
        run_child(commands[0], dataset, cwd=POLICY)
    # Feature shards are CPU/GPU-specific in the upstream builder; use eight
    # explicit workers so each shard is resumable and independently logged.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_child, [str(POLICY_PYTHON), "-u", str(builder), "shard",
                                           "--out", str(dataset), "--index", str(g), "--of", "8"],
                                dataset / "shards" / str(g), g, POLICY) for g in range(8)]
        for future in as_completed(futures):
            future.result()
    run_child(commands[1], dataset, cwd=POLICY)
    run_child(commands[2], dataset, cwd=POLICY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("dev", "train", "build", "all"))
    args = parser.parse_args()
    S1.mkdir(parents=True, exist_ok=True)
    atomic_json(S1 / "s1_driver_status.json", {"stage": args.stage, "started_utc": timestamp()})
    if args.stage in ("dev", "all"):
        run_dev()
    if args.stage in ("train", "all"):
        run_train()
    if args.stage in ("build", "all"):
        build_dataset()
    atomic_json(S1 / "s1_driver_status.json", {"stage": args.stage, "completed": True, "finished_utc": timestamp()})


if __name__ == "__main__":
    main()
