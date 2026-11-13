#!/usr/bin/env python3
"""Fruit-picking demonstrations with injected execution noise (DART, Laskey et al. 2017).

The scripted demonstrator and recorder of fruit_picking.generate run unchanged; only the joint targets the
simulator executes are perturbed by temporally correlated per-joint noise (Ornstein-Uhlenbeck, stationary std
`sigma` rad, correlation time `tau` steps), while the h5 records the demonstrator's clean targets as the action.
Actions are absolute joint targets and every segment is planned from the actual (perturbed) joint state, so the
label at a perturbed state leads back to the demonstrator's path: the data show recoveries from deviations of
the kind the policy makes in closed loop. Noise acts while the arm moves (`PHASE_SCALE`: reach, pull, carry, and
approach at half scale) and is off while it observes, grasps, releases and settles. Each gt row also carries the
executed joint offset and its tcp displacement.

Same output layout, manifest and retry rule as fruit_picking.generate (seed = split_base + 16 * episode + attempt);
the manifest adds the noise parameters and the identity of this script.

    CUDA_VISIBLE_DEVICES=0 robomme_benchmark/.venv/bin/python scripts/generate_fruit_dart.py --task count \
        --episodes 100-199 --sigma 0.02 --out ecm_vla/data/fruit_picking_dart
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ECM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECM))
from fruit_picking.generate import (MAX_ATTEMPTS, SPLIT_BASE, TASK_NAME, Recorder, atomic_write, code_version,  # noqa: E402
                                    finished, parse_range, sha256, verify)

PHASE_SCALE = {"reach": 1.0, "approach": 0.5, "pull": 1.0, "carry": 1.0}


class NoisyExecution:
    """Executes action + scale * OU noise on the arm joints; `scale` is set per step by the recorder."""

    def __init__(self, env, sigma, tau, rng):
        self.env, self.unwrapped = env, env.unwrapped
        self.n = len(env.unwrapped.agent.arm_joint_names)
        self.sigma, self.tau, self.rng = sigma, tau, rng
        self.noise = np.zeros(self.n)
        self.scale = 0.0
        self.offset = np.zeros(self.n)

    def step(self, action):
        if self.scale > 0:
            self.noise += -self.noise / self.tau + self.sigma * np.sqrt(2.0 / self.tau) * self.rng.standard_normal(self.n)
        else:
            self.noise[:] = 0.0
        executed = np.asarray(action, np.float64).copy()
        self.offset = self.scale * self.noise
        executed[:self.n] += self.offset
        return self.env.step(executed)


class NoisyRecorder(Recorder):
    def step(self, action, labels):
        self.env.scale = PHASE_SCALE.get(labels["phase"], 0.0)
        qpos = self.u.agent.robot.get_qpos()[0].cpu().numpy()
        obs, info = super().step(action, labels)
        offset = self.env.offset.copy()
        clean = np.concatenate([action[:self.n], qpos[self.n:]])
        moved = clean.copy()
        moved[:self.n] += offset
        self.gt[-1]["executed_offset"] = offset.tolist()
        self.gt[-1]["noise_tcp_m"] = float(np.linalg.norm(self.fk(moved).p - self.fk(clean).p)) if offset.any() else 0.0
        return obs, info


def run_episode(task, episode, seed, out: Path, robot: str, sigma: float, tau: float):
    import gymnasium as gym
    import h5py
    import fruit_picking.env  # noqa: F401  (registers env and agent)
    from fruit_picking.solver import FruitPickSolver, PlanningFailure

    name = f"{TASK_NAME[task]}_ep{episode}_seed{seed}"
    h5_path = out / "hdf5_files" / f"{name}.h5"
    gt_path = out / "gt" / f"{name}.jsonl"
    tmp = h5_path.with_name(h5_path.name + ".tmp")
    base = gym.make("FruitPick-v1", task=task, robot_uids=robot, obs_mode="rgb+depth+segmentation",
                    control_mode="pd_joint_pos", render_mode="rgb_array", sim_backend="cpu")
    env = NoisyExecution(base, sigma, tau, np.random.default_rng([seed, 1]))
    try:
        obs, _ = base.reset(seed=seed)
        u = base.unwrapped
        with h5py.File(tmp, "w") as f:
            ep = f.create_group(f"episode_{episode}")
            s = ep.create_group("setup")
            s.create_dataset("task_goal", data=[u.layout.instruction])
            s.create_dataset("seed", data=seed)
            s.create_dataset("difficulty", data="easy")
            s.create_dataset("task", data=task)
            s.create_dataset("robot", data=robot)
            s.create_dataset("layout_json", data=u.layout.to_json())
            for cam, key in (("front_camera", "front"), ("hand_camera", "wrist")):
                s.create_dataset(f"{key}_camera_intrinsic",
                                 data=obs["sensor_param"][cam]["intrinsic_cv"][0].cpu().numpy().astype(np.float32))
            solver = None
            rec = NoisyRecorder(env, lambda q: solver.fk(q), ep)
            rec.observe(obs)
            solver = FruitPickSolver(env, rec.step)
            try:
                ok, reason = solver.run(), ""
            except PlanningFailure as e:
                ok, reason = False, str(e)
            if ok and not rec.gt[-1]["success"]:
                ok, reason = False, "success not reached before the last recorded step"
            ep["setup"].create_dataset("n_timesteps", data=rec.t)
        if not ok:
            tmp.unlink(missing_ok=True)
            return {"ok": False, "reason": reason, "steps": rec.t}
        atomic_write(gt_path, "".join(json.dumps(r) + "\n" for r in rec.gt))
        os.replace(tmp, h5_path)
        noisy = [r["noise_tcp_m"] for r in rec.gt if r["phase"] in PHASE_SCALE]
        return {"ok": True, "h5": h5_path, "gt": gt_path, "steps": rec.t, "n_fruits": len(u.layout.fruits),
                "n_pick": u.layout.n_pick, "instruction": u.layout.instruction,
                "noise_tcp_m": {"median": float(np.median(noisy)), "p90": float(np.percentile(noisy, 90)),
                                "noisy_steps": len(noisy)}}
    finally:
        base.close()
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASK_NAME), required=True)
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--episodes", required=True, help="e.g. 100-199")
    parser.add_argument("--robot", choices=["rm75b_eg2"], default="rm75b_eg2")
    parser.add_argument("--sigma", type=float, required=True, help="stationary noise std per arm joint (rad)")
    parser.add_argument("--tau", type=float, default=10.0, help="noise correlation time (control steps)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out / args.robot / args.split / args.task
    for sub in ("hdf5_files", "gt"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    code = {**code_version(), "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve())}
    noise = {"sigma_rad": args.sigma, "tau_steps": args.tau, "phase_scale": PHASE_SCALE, "kind": "ornstein-uhlenbeck"}
    for e in parse_range(args.episodes):
        manifest = out / f"ep{e}_manifest.json"
        if finished(manifest):
            continue
        failures = []
        for attempt in range(MAX_ATTEMPTS):
            seed = SPLIT_BASE[args.split] + 16 * e + attempt
            t0 = time.time()
            res = run_episode(args.task, e, seed, out, args.robot, args.sigma, args.tau)
            if not res["ok"]:
                failures.append({"seed": seed, "reason": res["reason"], "steps": res["steps"]})
                print(f"[{args.task} ep{e}] seed {seed} failed: {res['reason']}", flush=True)
                continue
            record = {"robot": args.robot, "task": args.task, "split": args.split, "episode": e, "seed": seed,
                      "attempt": attempt, "code": code, "noise": noise, "noise_tcp_m": res["noise_tcp_m"],
                      "failures": failures, "steps": res["steps"], "n_fruits": res["n_fruits"], "n_pick": res["n_pick"],
                      "instruction": res["instruction"], "seconds": round(time.time() - t0, 1),
                      "h5": str(res["h5"].relative_to(out)), "h5_sha256": sha256(res["h5"]),
                      "gt": str(res["gt"].relative_to(out)), "gt_sha256": sha256(res["gt"]),
                      "verification": verify(res["h5"], res["gt"], e)}
            atomic_write(manifest, json.dumps(record, indent=1))
            print(f"[{args.task} ep{e}] seed {seed} ok: {res['steps']} steps, {record['seconds']} s, "
                  f"noise tcp median {100 * res['noise_tcp_m']['median']:.2f} cm", flush=True)
            break
        else:
            atomic_write(out / f"ep{e}_FAILED.json", json.dumps({"failures": failures}, indent=1))


if __name__ == "__main__":
    main()
