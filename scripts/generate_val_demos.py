#!/usr/bin/env python3
"""Generate held-out planner demonstrations on a RoboMME split for the V-B teacher probe.

Mirrors the official generator in third_party/robomme_benchmark/tests/_shared/dataset_generation.py,
with failure-recovery mode disabled so trajectories are clean expert demonstrations.
Run with the simulator venv, e.g.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<SIM>/src \
        /media/fulei/fuleivla/robomme_benchmark/.venv/bin/python ecm_vla/scripts/generate_val_demos.py \
        --task BinFill --episodes 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
import torch
import gymnasium as gym

from robomme.env_record_wrapper import RobommeRecordWrapper, FailsafeTimeout, BenchmarkEnvBuilder
from robomme.robomme_env import *  # noqa: F401,F403 - env registration
from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError
from robomme.robomme_env.utils.planner_fail_safe import (
    FailAwarePandaArmMotionPlanningSolver,
    FailAwarePandaStickMotionPlanningSolver,
    ScrewPlanFailure,
)

ECM = Path(__file__).resolve().parents[1]
SCREW_MAX_ATTEMPTS = 3
RRT_MAX_ATTEMPTS = 3
STICK_TASKS = ("PatternLock", "RouteStick")


def to_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def patch_screw_to_rrt(planner) -> None:
    screw, rrt = planner.move_to_pose_with_screw, planner.move_to_pose_with_RRTStar

    def move(*args, **kwargs):
        for _ in range(SCREW_MAX_ATTEMPTS):
            try:
                result = screw(*args, **kwargs)
            except ScrewPlanFailure:
                continue
            if not (isinstance(result, int) and result == -1):
                return result
        for _ in range(RRT_MAX_ATTEMPTS):
            try:
                result = rrt(*args, **kwargs)
            except Exception:
                continue
            if not (isinstance(result, int) and result == -1):
                return result
        return -1

    planner.move_to_pose_with_screw = move


def run_episode(task: str, episode: int, seed: int, difficulty, out_dir: Path) -> bool:
    env = gym.make(task, obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos",
                   render_mode="rgb_array", reward_mode="dense", seed=seed, difficulty=difficulty)
    # RecordWrapper only buffers timesteps inside its save_video branch.
    env = RobommeRecordWrapper(env, dataset=str(out_dir), env_id=task, episode=episode, seed=seed,
                               save_video=True)
    success = False
    try:
        env.reset()
        solver = FailAwarePandaStickMotionPlanningSolver if task in STICK_TASKS else FailAwarePandaArmMotionPlanningSolver
        kwargs = dict(debug=False, vis=False, base_pose=env.unwrapped.agent.robot.pose,
                      visualize_target_grasp_pose=False, print_env_info=False)
        if task in STICK_TASKS:
            kwargs["joint_vel_limits"] = 0.3
        planner = solver(env, **kwargs)
        patch_screw_to_rrt(planner)
        for entry in list(getattr(env.unwrapped, "task_list", []) or []):
            solve = entry.get("solve")
            if not callable(solve):
                continue
            env.unwrapped.evaluate(solve_complete_eval=True)
            failed = False
            try:
                result = solve(env, planner)
                failed = isinstance(result, int) and result == -1
            except ScrewPlanFailure:
                failed = True
            except FailsafeTimeout:
                break
            if failed:
                env.unwrapped.failureflag = torch.tensor([True])
                env.unwrapped.successflag = torch.tensor([False])
                env.unwrapped.current_task_failure = True
            evaluation = env.unwrapped.evaluate(solve_complete_eval=True)
            if to_bool(evaluation.get("success", False)):
                success = True
                break
            if failed or to_bool(evaluation.get("fail", False)):
                break
        else:
            success = to_bool(env.unwrapped.evaluate(solve_complete_eval=True).get("success", False))
        success = success or to_bool(getattr(env, "episode_success", False))
    except SceneGenerationError:
        success = False
    finally:
        try:
            env.close()
        except Exception:
            pass
    return success


def verify_h5(path: Path, episode: int) -> dict:
    """Check the fields that the training builder and the V-B probe read."""
    with h5py.File(path, "r") as f:
        ep = f[f"episode_{episode}"]
        steps = sorted(int(k.split("_")[-1]) for k in ep.keys() if k.startswith("timestep_"))
        assert steps == list(range(len(steps))), "timesteps must be contiguous from 0"
        demo = [bool(ep[f"timestep_{t}"]["info"]["is_video_demo"][()]) for t in steps]
        exec_start = demo.index(False)
        assert not any(demo[exec_start:]), "demo frames must precede execution frames"
        ts0 = ep[f"timestep_{exec_start}"]
        shapes = {"front_rgb": ts0["obs"]["front_rgb"].shape, "joint_state": ts0["obs"]["joint_state"].shape,
                  "gripper_state": ts0["obs"]["gripper_state"].shape,
                  "joint_action": ts0["action"]["joint_action"].shape}
        assert shapes == {"front_rgb": (256, 256, 3), "joint_state": (7,), "gripper_state": (2,),
                          "joint_action": (8,)}, shapes
        boundaries = [t for t in steps if bool(ep[f"timestep_{t}"]["info"]["is_subgoal_boundary"][()])]
        actions = np.stack([ep[f"timestep_{t}"]["action"]["joint_action"][()] for t in steps[exec_start:]])
        return {"timesteps": len(steps), "exec_start_idx": exec_start, "exec_frames": len(steps)-exec_start,
                "subgoal_boundaries": boundaries, "task_goal": ep["setup"]["task_goal"][()][0].decode(),
                "shapes": {k: list(v) for k, v in shapes.items()},
                "action_finite": bool(np.isfinite(actions).all()),
                "gripper_values": sorted({float(x) for x in np.unique(actions[:, 7])})[:6]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--max-seed-attempts", type=int, default=5)
    p.add_argument("--out", type=Path, default=ECM / "data/val_demos")
    opts = p.parse_args()
    builder = BenchmarkEnvBuilder(env_id=opts.task, dataset=opts.split)
    out = opts.out / opts.task
    out.mkdir(parents=True, exist_ok=True)
    status = 0
    for episode in opts.episodes:
        seed0, difficulty = builder.resolve_episode(episode)
        assert seed0 is not None, f"no metadata seed for {opts.task} {opts.split} episode {episode}"
        record = {"task": opts.task, "split": opts.split, "episode": episode, "metadata_seed": seed0,
                  "difficulty": difficulty, "failure_recovery": False, "attempts": []}
        for attempt in range(opts.max_seed_attempts):
            seed = seed0 + attempt
            start = time.monotonic()
            try:
                ok, error = run_episode(opts.task, episode, seed, difficulty, out), None
            except Exception:
                ok, error = False, traceback.format_exc()
            record["attempts"].append({"seed": seed, "success": ok, "seconds": round(time.monotonic()-start, 1),
                                       "error": error})
            if ok:
                h5 = out / "hdf5_files" / f"{opts.task}_ep{episode}_seed{seed}.h5"
                record.update(used_seed=seed, h5_path=str(h5),
                              h5_sha256=hashlib.sha256(h5.read_bytes()).hexdigest(),
                              verification=verify_h5(h5, episode))
                break
        else:
            status = 1
        (out / f"ep{episode:02d}_manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in record.items() if k != "attempts"} | {"n_attempts": len(record["attempts"])},
                         ensure_ascii=False), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
