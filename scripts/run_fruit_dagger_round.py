#!/usr/bin/env python3
"""One resumable, non-formal DAgger iteration on clean train fruit layouts.

Collects recoverable policy deviations with the scripted simulator expert,
exports only successful expert suffixes, builds an isolated training mixture,
fine-tunes on eight GPUs, and compares on sealed-out development layouts.
The formal validation split is never read by this runner.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from v_a_common import ECM, OPENPI_DATA_HOME, POLICY, POLICY_PYTHON, atomic_json, read_json, timestamp

MODES = ("reach_collect_auto", "grasp_collect_auto", "carry_collect_auto")
OFFSETS = (700, 900, 800)
BASE_INIT = ECM / "runs/fruit_dagger_pilot_8gpu_v2_20261030/ckpt/mme_vla_suite/k0_dagger_pilot/199"
TRAIN_ROOT = ECM / "data/fruit_picking/rm75b_eg2/train"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ECM / "runs/fruit_dagger_round2_20261030")
    p.add_argument("--first", type=int, default=4)
    p.add_argument("--stop", type=int, default=20)
    p.add_argument("--collection-gpu", type=int, default=0)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--base-per-episode", type=int, default=100)
    p.add_argument("--rollout-checkpoint", type=Path, default=BASE_INIT)
    p.add_argument("--extend", action="store_true", help="increase stop/steps in an unfinished round")
    a = p.parse_args()
    if not (0 <= a.first < a.stop <= 100) or a.steps <= 200 or not 0 <= a.collection_gpu <= 7 or \
            not 25 <= a.base_per_episode <= 400:
        p.error("expected clean train episodes 0-99, steps >200, base samples 25-400, GPU 0-7")
    checkpoint = a.rollout_checkpoint.resolve()
    if not (checkpoint / "params/manifest.ocdbt").is_file():
        p.error(f"rollout checkpoint incomplete: {checkpoint}")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = {"formal": False, "train_only": True, "first": a.first, "stop": a.stop,
            "modes": list(MODES), "collection_gpu": a.collection_gpu, "train_gpus": list(range(8)),
            "steps": a.steps, "base_per_episode": a.base_per_episode,
            "rollout_checkpoint": str(checkpoint)}
    plan_path = out / "round_plan.json"
    if plan_path.is_file():
        previous = read_json(plan_path)
        if previous != plan:
            stable = ("formal", "train_only", "first", "modes", "collection_gpu",
                      "train_gpus", "rollout_checkpoint")
            if not (a.extend and all(previous.get(k) == plan[k] for k in stable) and
                    previous["stop"] <= a.stop and previous["steps"] <= a.steps and
                    previous.get("base_per_episode", 25) <= a.base_per_episode and
                    not (out / "stage_collect.json").is_file()):
                p.error(f"existing round plan differs: {plan_path}")
            atomic_json(plan_path, plan)
    else:
        atomic_json(plan_path, plan)

    collection = out / "collection"
    exported = out / "corrections/rm75b_eg2/train"
    processed = out / "processed"
    mixed = out / "mixed"
    training = out / "training"
    dev = out / "dev"
    python = sys.executable
    policy_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(a.collection_gpu),
                  "JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                  "OPENPI_DATA_HOME": str(OPENPI_DATA_HOME), "PYTHONDONTWRITEBYTECODE": "1",
                  "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
                  "PYTHONPATH": str(POLICY / "src")}
    stages = []

    def stage(name, command, cwd=ECM, env=None, check=None):
        marker = out / f"stage_{name}.json"
        if marker.is_file() and read_json(marker).get("state") == "complete":
            if check is None or check():
                return
            raise RuntimeError(f"completed stage output missing: {name}")
        log_path = out / f"stage_{name}.log"
        atomic_json(out / "round_status.json", {"state": "running", "stage": name,
                                                 "updated_utc": timestamp()})
        print(json.dumps({"stage": name, "state": "starting", "time": timestamp()}), flush=True)
        with log_path.open("a", buffering=1) as log:
            code = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        if code != 0 or (check is not None and not check()):
            raise RuntimeError(f"{name} failed (exit {code}); inspect {log_path}")
        atomic_json(marker, {"state": "complete", "exit_code": code, "completed_utc": timestamp(),
                             "command": command, "log": str(log_path)})
        stages.append(name)
        print(json.dumps({"stage": name, "state": "complete", "time": timestamp()}), flush=True)

    episodes = list(range(a.first, a.stop))
    try:
        stage("collect", [python, "scripts/run_fruit_phase_diagnosis.py", "--out", str(collection),
                          "--root", str(TRAIN_ROOT), "--episodes", *map(str, episodes),
                          "--modes", *MODES, "--gpu", str(a.collection_gpu),
                          "--checkpoint", str(checkpoint), "--extend-episodes"],
              check=lambda: (collection / "summary.json").is_file() and
              all(read_json(collection / "summary.json")["by_task"][task][mode]["completed"] == len(episodes)
                  for task in ("FruitCount", "FruitOccludedRipe") for mode in MODES))
        stage("export", [python, "scripts/export_fruit_dagger_corrections.py",
                         "--source", str(collection), "--out", str(exported),
                         "--episodes", *map(str, episodes), "--modes", *MODES,
                         "--eligible-only", "--resume"],
              check=lambda: (exported / "collection_summary.json").is_file() and
              read_json(exported / "collection_summary.json")["records"] >= 8)
        count = read_json(exported / "collection_summary.json")["records"]
        print(json.dumps({"accepted_corrections": count}), flush=True)
        parts = []
        for offset in OFFSETS:
            parts.extend(["--part", str(exported), str(offset + a.first), str(offset + a.stop)])
        stage("preprocess_plan", [str(POLICY_PYTHON), "-u", str(ECM / "scripts/build_fruit_dataset.py"),
                                  "plan", "--out", str(processed), *parts, "--skip-missing"],
              cwd=POLICY, env=policy_env,
              check=lambda: (processed / "meta/episodes.json").is_file() and
              len(read_json(processed / "meta/episodes.json")) == count)
        stage("preprocess_shard", [str(POLICY_PYTHON), "-u", str(ECM / "scripts/build_fruit_dataset.py"),
                                   "shard", "--out", str(processed), "--index", "0", "--of", "1"],
              cwd=POLICY, env=policy_env)
        stage("preprocess_finish", [str(POLICY_PYTHON), "-u", str(ECM / "scripts/build_fruit_dataset.py"),
                                    "finish", "--out", str(processed)],
              cwd=POLICY, env=policy_env,
              check=lambda: read_json(processed / "meta/build.json")["passed"])
        stage("mix", [str(POLICY_PYTHON), "-u", str(ECM / "scripts/build_fruit_dagger_pilot_mix.py"),
                      "--out", str(mixed), "--extra-corrections", str(processed),
                      "--extra-manifests", str(exported), "--base-per-episode", str(a.base_per_episode),
                      "--direct-extra", "1"],
              cwd=ECM, env=policy_env,
              check=lambda: read_json(mixed / "meta/build.json")["correction_episodes"] >= count + 17)
        stage("prepare_train", [python, "scripts/prepare_fruit_dagger_pilot.py", "--out", str(training),
                                "--dataset", str(mixed), "--init", str(checkpoint),
                                "--steps", str(a.steps), "--arm", "k0_dagger_round2", "--gpus",
                                *map(str, range(8))],
              check=lambda: (training / "plan.json").is_file())
        stage("train", [python, "scripts/run_fruit_dagger_pilot.py", "--out", str(training)],
              check=lambda: read_json(training / "status.json")["state"] == "complete")
        trained = Path(read_json(training / "status.json")["checkpoint"])
        stage("dev", [python, "scripts/run_fruit_dagger_pilot_dev.py", "--out", str(dev),
                      "--checkpoint", str(trained), "--gpu", str(a.collection_gpu)],
              check=lambda: (dev / "summary.json").is_file())
        comparison = read_json(dev / "summary.json")
        result = {"state": "complete", "completed_utc": timestamp(),
                  "accepted_corrections": count, "training_checkpoint": str(trained),
                  "baseline_success": comparison["baseline_success"],
                  "round_success": comparison["pilot_success"],
                  "baseline_in_basket": comparison["baseline_in_basket"],
                  "round_in_basket": comparison["pilot_in_basket"],
                  "formal": False, "development": True}
        atomic_json(out / "round_status.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as exc:
        atomic_json(out / "round_status.json", {"state": "failed", "error": str(exc),
                                                 "traceback": traceback.format_exc(),
                                                 "updated_utc": timestamp()})
        raise


if __name__ == "__main__":
    main()
