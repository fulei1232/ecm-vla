#!/usr/bin/env python3
"""Run and check the isolated eight-GPU, 200-step K0 correction pilot."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

from v_a_common import ECM, OPENPI_DATA_HOME, POLICY, POLICY_PYTHON, atomic_json, read_json, sha256, timestamp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ECM / "runs/fruit_dagger_pilot_8gpu_20261030")
    a = p.parse_args()
    out = a.out.resolve()
    plan = read_json(out / "plan.json")
    spec_path = Path(plan.get("spec_path", out / "arms/k0_dagger_pilot.json"))
    spec = read_json(spec_path)
    if plan["formal"] or sha256(spec_path) != plan["spec_sha256"]:
        p.error("pilot plan/spec differs or is marked formal")
    if len(spec["gpus"]) != 8 or spec["train"]["fsdp_devices"] != 8 or spec["train"]["steps"] < 1:
        p.error("expected exactly eight GPUs and a positive step count")
    dataset = Path(spec["dataset"])
    if sha256(dataset / "meta/build.json") != plan["dataset_build_sha256"] or \
            sha256(dataset / "meta/stats.json") != plan["dataset_stats_sha256"]:
        p.error("pilot mixture changed after preparation")
    init = Path(spec["init_checkpoint"])
    if sha256(init / "params/manifest.ocdbt") != plan["init_params_manifest_sha256"]:
        p.error("initial K0 checkpoint changed")
    checkpoint = Path(plan["output_checkpoint"])
    if checkpoint.exists():
        p.error(f"output already exists: {checkpoint}")
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    if log_path.exists():
        p.error(f"training log already exists: {log_path}")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, spec["gpus"])),
           "JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.92",
           "OPENPI_DATA_HOME": str(OPENPI_DATA_HOME), "WANDB_MODE": "disabled",
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "PYTHONPATH": f"{POLICY}/src:{POLICY}/packages/openpi-client/src"}
    command = [str(POLICY_PYTHON), "-u", str(ECM / "scripts/finetune_event_keyframes.py"),
               "--spec", str(spec_path)]
    atomic_json(out / "status.json", {"state": "running", "started_utc": timestamp(),
                                      "gpus": spec["gpus"], "steps": spec["train"]["steps"], "log": str(log_path)})
    with log_path.open("x", buffering=1) as log:
        code = subprocess.run(command, cwd=POLICY, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    content = log_path.read_text(errors="replace")
    losses = [float(x) for x in re.findall(r"Step \d+: .*?\bloss=([^,\s]+)", content)]
    expected = len(range(0, spec["train"]["steps"], spec["train"]["log_interval"]))
    complete = (checkpoint / "_CHECKPOINT_METADATA").is_file() and \
               (checkpoint / "params/manifest.ocdbt").is_file() and \
               (checkpoint / "assets/robomme/norm_stats.json").is_file()
    stats_match = complete and sha256(checkpoint / "assets/robomme/norm_stats.json") == \
                  sha256(init / "assets/robomme/norm_stats.json")
    passed = code == 0 and complete and stats_match and len(losses) == expected and all(map(math.isfinite, losses))
    status = {"state": "complete" if passed else "failed", "finished_utc": timestamp(),
              "exit_code": code, "checkpoint": str(checkpoint), "checkpoint_complete": complete,
              "norm_stats_match_initial_k0": stats_match, "logged_losses": len(losses),
              "expected_logged_losses": expected, "first_loss": losses[0] if losses else None,
              "last_loss": losses[-1] if losses else None, "finite_losses": all(map(math.isfinite, losses)),
              "log": str(log_path)}
    atomic_json(out / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
