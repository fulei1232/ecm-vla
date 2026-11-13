#!/usr/bin/env python3
"""Run S2 smoke tests or formal training for the three arms."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from v_a_common import ECM, POLICY, POLICY_PYTHON, atomic_json, timestamp


ARMS = {"sg": ("0,1,2,3", 0), "k0": ("4,5,6,7", 1), "nm": ("0,1,2,3", 0)}


def run_arm(arm: str, run_dir: Path, steps: int, gpu_ids: str) -> None:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{arm}_{steps}.log"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu_ids,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.92",
        "OPENPI_DATA_HOME": str(ECM.parent / "acm_vla/cache/openpi"),
        "WANDB_MODE": "disabled",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{POLICY}/src:{POLICY}/packages/openpi-client/src:{ECM}/scripts:{ECM}/src:{ECM}/scripts/harvestmem_v3",
    }
    command = [str(POLICY_PYTHON), "-u", str(ECM / "scripts/harvestmem_v3/s2_train.py"),
               "--arm", arm, "--run-dir", str(run_dir), "--steps", str(steps), "--fsdp-devices", "4"]
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=POLICY, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{arm} failed with exit {result.returncode}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "formal"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=tuple(ARMS))
    args = parser.parse_args()
    steps = 3 if args.stage == "smoke" else 12000
    args.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.run_dir / f"{args.stage}_started.json",
                {"stage": args.stage, "arms": args.arms, "steps": steps, "started_utc": timestamp()})
    errors = []
    with ThreadPoolExecutor(max_workers=len(args.arms)) as pool:
        futures = [pool.submit(run_arm, arm, args.run_dir / arm, steps, ARMS[arm][0]) for arm in args.arms]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append(repr(exc))
    atomic_json(args.run_dir / f"{args.stage}_finished.json",
                {"stage": args.stage, "arms": args.arms, "steps": steps,
                 "passed": not errors, "errors": errors, "finished_utc": timestamp()})
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
