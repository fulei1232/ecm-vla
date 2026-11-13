#!/usr/bin/env python3
"""Resume-safe full-task train collection, fine-tuning, and 40-episode dev gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from v_a_common import ECM, OPENPI_DATA_HOME, POLICY, POLICY_PYTHON, atomic_json, read_json, timestamp

TRAIN_ROOT = ECM / "data/fruit_picking/rm75b_eg2/train"
PRIOR = ECM / "runs/fruit_dagger_round2_20261030"
INIT = PRIOR / "training/ckpt/mme_vla_suite/k0_dagger_round2/7999"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ECM / "runs/fruit_dagger_full_task_20261031")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--stop", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--steps", type=int, default=12000)
    a = p.parse_args()
    if not 0 <= a.first < a.stop <= 100 or not 0 <= a.gpu <= 7 or a.steps < 8000:
        p.error("expected clean train 0-99, GPU 0-7 and at least 8000 steps")
    if not (INIT / "params/manifest.ocdbt").is_file():
        p.error("prior DAgger checkpoint is incomplete")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = {"formal": False, "train_only_collection": True, "first": a.first, "stop": a.stop,
            "gpu": a.gpu, "train_gpus": list(range(8)), "steps": a.steps,
            "rollout_checkpoint": str(INIT), "tasks": ["FruitCount", "FruitOccludedRipe"],
            "mode": "full_collect", "development_episodes": list(range(200, 220))}
    plan_path = out / "round_plan.json"
    if plan_path.is_file():
        if read_json(plan_path) != plan:
            p.error("existing full-task round plan differs")
    else:
        atomic_json(plan_path, plan)
    collection = out / "collection"
    corrections = out / "corrections/rm75b_eg2/train"
    processed = out / "processed"
    mixed = out / "mixed"
    training = out / "training"
    dev = out / "dev40"
    policy_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu), "JAX_PLATFORMS": "cuda",
                  "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OPENPI_DATA_HOME": str(OPENPI_DATA_HOME),
                  "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
                  "PYTHONPATH": str(POLICY / "src")}

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
        atomic_json(marker, {"state": "complete", "exit_code": code,
                             "completed_utc": timestamp(), "command": command,
                             "log": str(log_path)})
        print(json.dumps({"stage": name, "state": "complete", "time": timestamp()}), flush=True)

    count = a.stop - a.first
    try:
        stage("collect", [sys.executable, "scripts/run_fruit_phase_diagnosis.py", "--out", str(collection),
                          "--root", str(TRAIN_ROOT), "--episodes", *map(str, range(a.first, a.stop)),
                          "--modes", "full_collect", "--gpu", str(a.gpu), "--checkpoint", str(INIT)],
              check=lambda: (collection / "summary.json").is_file() and
              all(read_json(collection / "summary.json")["by_task"][task]["full_collect"]["completed"] == count
                  for task in ("FruitCount", "FruitOccludedRipe")))
        stage("export", [sys.executable, "scripts/export_fruit_full_dagger.py", "--source", str(collection),
                         "--out", str(corrections), "--first", str(a.first), "--stop", str(a.stop)],
              check=lambda: (corrections / "collection_summary.json").is_file() and
              read_json(corrections / "collection_summary.json")["attempts_with_corrections"] >= 32 and
              sum(int(v) for k, v in read_json(corrections / "collection_summary.json")["per_pick_index"].items()
                  if int(k) >= 1) >= 64)
        export = read_json(corrections / "collection_summary.json")
        print(json.dumps({"full_task_attempts_with_corrections": export["attempts_with_corrections"],
                          "correction_segments": export["records"],
                          "segments_by_pick": export["per_pick_index"]}), flush=True)
        first_id = 10000 + a.first * 100
        stop_id = 10000 + a.stop * 100
        stage("preprocess_plan", [str(POLICY_PYTHON), "-u", str(ECM / "scripts/build_fruit_dataset.py"),
                                  "plan", "--out", str(processed), "--part", str(corrections),
                                  str(first_id), str(stop_id), "--skip-missing"],
              cwd=POLICY, env=policy_env,
              check=lambda: len(read_json(processed / "meta/episodes.json")) == export["records"])
        stage("preprocess_shard", [str(POLICY_PYTHON), "-u", str(ECM / "scripts/build_fruit_dataset.py"),
                                   "shard", "--out", str(processed), "--index", "0", "--of", "1"],
              cwd=POLICY, env=policy_env)
        stage("preprocess_finish", [str(POLICY_PYTHON), "-u", str(ECM / "scripts/build_fruit_dataset.py"),
                                    "finish", "--out", str(processed)], cwd=POLICY, env=policy_env,
              check=lambda: read_json(processed / "meta/build.json")["passed"])
        stage("mix", [str(POLICY_PYTHON), "-u", str(ECM / "scripts/build_fruit_dagger_pilot_mix.py"),
                      "--out", str(mixed), "--base-per-episode", "100", "--direct-extra", "1",
                      "--additional-source", "round2", str(PRIOR / "processed"),
                      str(PRIOR / "corrections/rm75b_eg2/train"),
                      "--additional-source", "full_task", str(processed), str(corrections)],
              env=policy_env,
              check=lambda: read_json(mixed / "meta/build.json")["correction_episodes"] >=
              17 + 273 + export["records"])
        stage("prepare_train", [sys.executable, "scripts/prepare_fruit_dagger_pilot.py",
                                "--out", str(training), "--dataset", str(mixed), "--init", str(INIT),
                                "--steps", str(a.steps), "--arm", "k0_dagger_full_task", "--gpus",
                                *map(str, range(8))],
              check=lambda: (training / "plan.json").is_file())
        stage("train", [sys.executable, "scripts/run_fruit_dagger_pilot.py", "--out", str(training)],
              check=lambda: read_json(training / "status.json")["state"] == "complete")
        checkpoint = Path(read_json(training / "status.json")["checkpoint"])
        stage("dev40", [sys.executable, "scripts/run_fruit_dagger_dev40.py", "--out", str(dev),
                        "--checkpoint", str(checkpoint), "--gpu", str(a.gpu),
                        "--first", "200", "--stop", "220"],
              check=lambda: read_json(dev / "summary.json")["episodes"] == 40)
        comparison = read_json(dev / "summary.json")
        status = {"state": "complete", "formal": False, "development": True,
                  "completed_utc": timestamp(), "full_task_attempts_with_corrections":
                  export["attempts_with_corrections"], "correction_segments": export["records"],
                  "training_checkpoint": str(checkpoint),
                  "dev_episodes": comparison["episodes"],
                  "baseline_success": comparison["baseline_success"],
                  "candidate_success": comparison["candidate_success"],
                  "baseline_in_basket": comparison["baseline_in_basket"],
                  "candidate_in_basket": comparison["candidate_in_basket"],
                  "baseline_correct_in_basket": comparison["baseline_correct_in_basket"],
                  "candidate_correct_in_basket": comparison["candidate_correct_in_basket"],
                  "baseline_wrong_in_basket": comparison["baseline_wrong_in_basket"],
                  "candidate_wrong_in_basket": comparison["candidate_wrong_in_basket"]}
        atomic_json(out / "round_status.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except BaseException as exc:
        atomic_json(out / "round_status.json", {"state": "failed", "error": str(exc),
                                                 "traceback": traceback.format_exc(),
                                                 "updated_utc": timestamp()})
        raise


if __name__ == "__main__":
    main()
