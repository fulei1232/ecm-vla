#!/usr/bin/env python3
"""Run small first-pick phase substitutions on isolated fruit splits.

All runs use the same DART-trained K0 checkpoint. The root/episode validation
keeps development and training layouts separate; neither alters the sealed
formal evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

from v_a_common import ECM, POLICY, SIM, SIM_PYTHON, atomic_json, read_json, timestamp
from diagnose_fruit_phase_episode import MODES
from run_fruit_diagnosis import start_server, wait_ready

FORMAL = ECM / "results/fruit_dart_kf_eval_20261027"
DEV = ECM / "data/fruit_picking_dev/rm75b_eg2/train"
TRAIN = ECM / "data/fruit_picking/rm75b_eg2/train"
DART = ECM / "data/fruit_picking_dart/rm75b_eg2/train"
TASKS = ("FruitCount", "FruitOccludedRipe")


def jobs(out: Path, episodes: list[int], modes: list[str]):
    for task in TASKS:
        for episode in episodes:
            for mode in modes:
                yield task, episode, mode, out / mode / task / f"ep{episode}"


def summarize(out: Path, episodes: list[int], modes: list[str]):
    result = {"formal": False, "diagnostic": True, "generated_utc": timestamp(),
              "episodes": episodes, "modes": modes, "by_task": {}}
    for task in TASKS:
        result["by_task"][task] = {}
        for mode in modes:
            paths = [out / mode / task / f"ep{e}/result.json" for e in episodes]
            rows = [read_json(p) for p in paths if p.is_file()]
            result["by_task"][task][mode] = {
                "planned": len(episodes), "completed": len(rows),
                "errors": sum(r["outcome"] == "error" for r in rows),
                "first_pick_success": sum(bool(r.get("first_pick_success")) for r in rows),
                "target_detached": sum(bool(r.get("target_detached")) for r in rows),
                "target_in_basket": sum(bool(r.get("target_in_basket")) for r in rows),
                "task_success": sum(bool(r.get("task_success")) for r in rows),
                "correct_fruits_in_basket": sum(int(r.get("target_baskets", 0)) for r in rows),
                "wrong_fruits_in_basket": sum(int(r.get("wrong_baskets", 0)) for r in rows),
                "outcomes": dict(Counter(r["outcome"] for r in rows)),
                "reasons": dict(Counter(r.get("reason", "") for r in rows if r.get("reason"))),
            }
    atomic_json(out / "summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ECM / "results/fruit_phase_diagnosis_20261030")
    p.add_argument("--root", type=Path, default=DEV)
    p.add_argument("--episodes", type=int, nargs="+", default=[200, 201, 202, 203])
    p.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES[:8]))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, help="policy checkpoint for iterative collection (default: original K0)")
    p.add_argument("--extend-episodes", action="store_true", help="extend an existing collection with more train episodes")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    out = args.out.resolve()
    root = args.root.resolve()
    episodes = sorted(set(args.episodes))
    modes = list(dict.fromkeys(args.modes))
    if not modes:
        p.error("at least one mode is required")
    valid_ranges = {DEV.resolve(): range(200, 220), TRAIN.resolve(): range(0, 100),
                    DART.resolve(): range(100, 200)}
    if root not in valid_ranges or any(e not in valid_ranges[root] for e in episodes):
        p.error("root/episodes must be dev 200-219, clean train 0-99, or DART train 100-199")
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        print(json.dumps(summarize(out, episodes, modes), ensure_ascii=False, indent=2))
        return 0

    params = read_json(FORMAL / "protocol.json")["params"]
    assert params["conditions"] == ["K0", "K4", "K8", "C0"]
    checkpoint = args.checkpoint.resolve() if args.checkpoint else Path(params["checkpoints"]["K0"])
    assert checkpoint.is_dir(), checkpoint
    if not (checkpoint / "params/manifest.ocdbt").is_file():
        p.error(f"checkpoint is incomplete: {checkpoint}")
    params["checkpoints"]["K0"] = str(checkpoint)
    plan = {"formal": False, "purpose": "first-pick phase substitution", "started_utc": timestamp(),
            "source_root": str(root), "checkpoint": str(checkpoint), "gpu": args.gpu,
            "tasks": list(TASKS), "episodes": episodes, "modes": modes}
    if (out / "plan.json").is_file():
        previous = read_json(out / "plan.json")
        for key in ("source_root", "checkpoint", "gpu", "tasks", "episodes", "modes"):
            if previous[key] != plan[key]:
                if key == "episodes" and args.extend_episodes and previous[key] == episodes[:len(previous[key])]:
                    continue
                p.error(f"existing collection plan differs at {key}: {out}")
        if previous["episodes"] != episodes:
            atomic_json(out / "plan.json", plan)
    else:
        atomic_json(out / "plan.json", plan)

    server = start_server(out, 0, args.gpu, params, conditions=("K0",), base_port=28911)
    try:
        wait_ready([server])
        for task, episode, mode, directory in jobs(out, episodes, modes):
            if (directory / "result.json").is_file():
                if read_json(directory / "result.json").get("outcome") != "error":
                    continue
            directory.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu),
                   "PYTHONDONTWRITEBYTECODE": "1", "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
                   "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
                   "PYTHONPATH": f"{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src"}
            cmd = [str(SIM_PYTHON), "-u", str(ECM / "scripts/diagnose_fruit_phase_episode.py"),
                   "--port", str(server[1]["port"]), "--context", str(out / "server0/context.json"),
                   "--root", str(root), "--task", task, "--episode-id", str(episode),
                   "--mode", mode, "--attempt-dir", str(directory)]
            with (directory / "console.log").open("w") as log:
                proc = subprocess.run(cmd, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            row = read_json(directory / "result.json")
            print(json.dumps({"task": task, "episode": episode, "mode": mode,
                              "outcome": row["outcome"], "first_pick_success": row.get("first_pick_success"),
                              "reason": row.get("reason"), "exit": proc.returncode}, ensure_ascii=False), flush=True)
            summarize(out, episodes, modes)
            if mode == "expert" and not row.get("first_pick_success"):
                raise RuntimeError(f"expert sanity run failed: {directory}")
    finally:
        server[0].terminate()
        server[0].wait(120)
    summary = summarize(out, episodes, modes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
