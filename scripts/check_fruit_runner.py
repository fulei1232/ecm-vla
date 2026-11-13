#!/usr/bin/env python3
"""Replay demonstrations through the closed-loop fruit runner and recorder (memory_management/fruit_eval.py) and compare
every frame with the training data: images, state and the recorded distance against fruit_distance.from_gt_row
on the generator's ground truth. The replay must stop with success at the first step whose ground truth reports
success (the demonstrator keeps recording a short settle phase after that; closed-loop episodes end there).

    robomme_benchmark/.venv/bin/python scripts/check_fruit_runner.py --root data/fruit_picking/rm75b_eg2/val \
        --episodes FruitCount:0 FruitOccludedRipe:0 --out results/fruit_runner_check.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ECM = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ECM), str(ECM / "src")]
from memory_management import fruit_distance as fd  # noqa: E402
from memory_management.fruit_eval import FruitEnvRunner, FruitRawRecorder  # noqa: E402


def check(root, env_id, episode):
    import h5py
    runner = FruitEnvRunner(env_id, root, max_steps=1300)
    man = runner.manifest(episode)
    rows = [json.loads(x) for x in (runner.root / man["gt"]).read_text().splitlines()]
    with tempfile.TemporaryDirectory() as tmp, h5py.File(runner.root / man["h5"], "r") as f:
        ep = f[f"episode_{man['episode']}"]
        ripe = fd.ripe_flags(ep["setup/layout_json"][()])
        recorder = FruitRawRecorder(tmp, env_id)
        runner.make_env(episode)
        recorder.install(runner)
        init = runner.get_init_obs()
        frames = [(init["images"][0], init["states"][0])]
        outcome = stop = None
        for t in range(len(rows)):
            (front, _, state), stop, outcome = runner.step(ep[f"timestep_{t}/action/joint_action"][()])
            if front is not None:
                frames.append((front, state))
            if stop:
                break
        runner.close_env()
        image_diff, state_diff, distance_diff = [], [], []
        for t in range(min(len(rows), len(frames))):
            ts = ep[f"timestep_{t}/obs"]
            h5_state = np.concatenate([ts["joint_state"][()], ts["gripper_state"][()][:1]]).astype(np.float32)
            image_diff.append(int(np.abs(frames[t][0].astype(int) - ts["front_rgb"][()].astype(int)).max()))
            state_diff.append(float(np.abs(frames[t][1] - h5_state).max()))
            distance_diff.append(abs(recorder.rows[t]["gt"]["fruit_distance"] - fd.from_gt_row(rows[t], ripe)))
    first_state_diff = next((t for t, d in enumerate(state_diff) if d >= 1e-4), None)
    first_success = next(t for t, r in enumerate(rows) if r["success"])
    return {"task": env_id, "episode": episode, "seed": man["seed"], "demo_steps": len(rows),
            "replayed_steps": recorder.control_step, "first_success_row": first_success,
            "outcome": outcome, "stopped": bool(stop),
            "max_image_diff": max(image_diff), "frames_with_image_diff": sum(d > 0 for d in image_diff),
            "max_state_diff": max(state_diff), "first_frame_state_diff": first_state_diff,
            "max_distance_diff": max(distance_diff), "task_goal": init["task_goal"],
            "passed": outcome == "success" and recorder.control_step == first_success and max(image_diff) == 0
                      and max(state_diff) < 1e-4 and max(distance_diff) < 1e-4}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True, help="Task:episode")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    results = [check(a.root, task, int(episode)) for task, episode in (x.split(":") for x in a.episodes)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"passed": all(r["passed"] for r in results), "episodes": results}, indent=1))
    print(json.dumps(results, indent=1))
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
