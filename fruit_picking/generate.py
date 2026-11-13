#!/usr/bin/env python3
"""Generate scripted fruit-picking demonstrations in the RoboMME h5 layout, with a per-step ground-truth sidecar.

Outputs per episode under `<out>/<robot>/<split>/<task>/`:
- `hdf5_files/<Task>_ep<e>_seed<s>.h5`: group `episode_<e>` with `setup/` and `timestep_<t>/{obs,action,info}`
  (same keys as RoboMME; with the default 7-joint RM75-B, `joint_state` is 7-d and `joint_action` 8-d = 7 + gripper,
  as for the Panda).
- `gt/<Task>_ep<e>_seed<s>.jsonl`: one row per timestep (state before the step's action). `d_grasp` is the tcp
  distance to the grasp point (centre) of the first target fruit still on the tree, so it jumps to the next fruit
  once the current one detaches and is null after the last pick; `d_drop` is the distance to the drop pose.
- `ep<e>_manifest.json`: seed, attempts, sha256 of both files, verification. Episodes whose manifest exists are skipped.

A failed attempt (planning failure, missed detach or basket, or task not solved) is discarded and retried with the
next seed: seed = split_base + 16 * episode + attempt.

    CUDA_VISIBLE_DEVICES=0 robomme_benchmark/.venv/bin/python -m fruit_picking.generate --task occluded_ripe \
        --split train --episodes 0-99 --out ecm_vla/data/fruit_picking
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SPLIT_BASE = {"train": 100000, "val": 200000, "test": 300000}
MAX_ATTEMPTS = 16
TASK_NAME = {"occluded_ripe": "FruitOccludedRipe", "count": "FruitCount"}
IMAGE_KW = dict(compression="gzip", compression_opts=4)


def parse_range(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def code_version() -> dict:
    import subprocess
    repo = Path(__file__).resolve().parents[1]
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "fruit_picking"))}


def finished(manifest: Path) -> bool:
    try:
        return json.loads(manifest.read_text()).get("seed") is not None
    except (OSError, ValueError):
        return False


class Recorder:
    """Writes (obs before the step, action, labels) for every control step straight into an h5 group."""

    def __init__(self, env, solver_fk, h5_group):
        import transforms3d
        self.env, self.u, self.fk, self.g = env, env.unwrapped, solver_fk, h5_group
        self.n = len(self.u.agent.arm_joint_names)
        self.euler = lambda q: np.asarray(transforms3d.euler.quat2euler(q, "sxyz"))
        self.t = 0
        self.prev_subgoal = None
        self.gt = []
        self.obs = None

    def observe(self, obs):
        self.obs = obs

    def seg_ids(self):
        return [int(f.per_scene_id[0]) for f in self.u.fruits]

    def step(self, action, labels):
        import torch
        u, obs, n = self.u, self.obs, self.n
        sd, sp = obs["sensor_data"], obs["sensor_param"]
        g = self.g.create_group(f"timestep_{self.t}")
        tcp = u.agent.tcp_pose
        tcp_p, tcp_q = tcp.p[0].cpu().numpy(), tcp.q[0].cpu().numpy()
        qpos = u.agent.robot.get_qpos()[0].cpu().numpy()
        o = g.create_group("obs")
        for cam, key in (("front_camera", "front"), ("hand_camera", "wrist")):
            o.create_dataset(f"{key}_rgb", data=sd[cam]["rgb"][0].cpu().numpy().astype(np.uint8), **IMAGE_KW)
            depth = sd[cam]["depth"][0].cpu().numpy()
            o.create_dataset(f"{key}_depth", data=np.clip(depth, -32768, 32767).astype(np.int16), **IMAGE_KW)
            o.create_dataset(f"{key}_camera_extrinsic", data=sp[cam]["extrinsic_cv"][0].cpu().numpy().astype(np.float32))
        o.create_dataset("joint_state", data=qpos[:n].astype(np.float32))
        o.create_dataset("gripper_state", data=qpos[n:n + 2].astype(np.float64))
        o.create_dataset("eef_state", data=np.concatenate([tcp_p, self.euler(tcp_q)]).astype(np.float32))
        # the horizontal approach sits at the sxyz gimbal lock (pitch = pi/2), so keep the quaternion as well
        o.create_dataset("eef_quat", data=tcp_q.astype(np.float32))
        o.create_dataset("is_gripper_close", data=bool(action[n] < 0))
        target = self.fk(np.concatenate([action[:n], qpos[n:]]))
        a = g.create_group("action")
        a.create_dataset("joint_action", data=np.asarray(action, np.float64))
        a.create_dataset("eef_action", data=np.concatenate([target.p, self.euler(target.q), [action[n]]]))
        info_before = u.evaluate()
        seg = sd["front_camera"]["segmentation"][0, ..., 0].cpu().numpy()
        pixels = [int((seg == i).sum()) for i in self.seg_ids()]
        positions = u.fruit_positions()
        inside = u.in_basket()
        attached = list(u.attached)
        push = list(u.occluder_push)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        i = labels.get("target")
        nxt = next((j for j in u.layout.targets if not attached[j]), None)
        d_grasp = None if nxt is None else float(np.linalg.norm(tcp_p - positions[nxt]))
        d_drop = float(np.linalg.norm(tcp_p - np.asarray(u.layout.basket["drop"])))
        meta = g.create_group("info")
        meta.create_dataset("simple_subgoal", data=labels["subgoal"])
        meta.create_dataset("grounded_subgoal", data=labels["subgoal"])
        meta.create_dataset("simple_subgoal_online", data=labels["subgoal"])
        meta.create_dataset("grounded_subgoal_online", data=labels["subgoal"])
        meta.create_dataset("is_subgoal_boundary", data=labels["subgoal"] != self.prev_subgoal)
        meta.create_dataset("is_video_demo", data=False)
        meta.create_dataset("is_completed", data=bool(info_before["success"][0]))
        self.prev_subgoal = labels["subgoal"]
        self.gt.append({
            "timestep": self.t, "phase": labels["phase"], "subgoal": labels["subgoal"], "target": i,
            "next_target": nxt, "pick_index": labels.get("pick_index"),
            "tcp_pos": tcp_p.tolist(), "tcp_quat": tcp_q.tolist(), "gripper_qpos": qpos[n:n + 2].tolist(),
            "gripper_action": float(action[n]), "d_grasp": d_grasp, "d_drop": d_drop,
            "fruits": [{"pos": positions[k].tolist(), "attached": attached[k], "in_basket": bool(inside[k]),
                        "front_pixels": pixels[k]} for k in range(len(positions))],
            "occluder_alpha": float(np.clip((self.t - u.layout.gust[0]) / (u.layout.gust[1] - u.layout.gust[0]), 0, 1)),
            "occluder_push": push,
            "n_in_basket": int(inside.sum()), "success": bool(info_before["success"][0]),
        })
        self.t += 1
        return obs, info


def run_episode(task, episode, seed, out: Path, robot: str, ik_mode="legacy", ik_seed=0, ik_candidates=64,
                canonical_options=None):
    import gymnasium as gym
    import h5py
    import fruit_picking.env  # noqa: F401  (registers env and agent)
    from fruit_picking.solver import FruitPickSolver, PlanningFailure

    name = f"{TASK_NAME[task]}_ep{episode}_seed{seed}"
    h5_path = out / "hdf5_files" / f"{name}.h5"
    gt_path = out / "gt" / f"{name}.jsonl"
    tmp = h5_path.with_name(h5_path.name + ".tmp")
    env = gym.make("FruitPick-v1", task=task, robot_uids=robot, obs_mode="rgb+depth+segmentation",
                   control_mode="pd_joint_pos", render_mode="rgb_array", sim_backend="cpu")
    try:
        obs, _ = env.reset(seed=seed)
        u = env.unwrapped
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
            rec = Recorder(env, lambda q: solver.fk(q), ep)
            rec.observe(obs)
            solver = FruitPickSolver(env, rec.step, ik_mode=ik_mode, ik_seed=ik_seed, ik_candidates=ik_candidates)
            if ik_mode == "canonical":
                from fruit_picking.canonical_solver import configure
                configure(solver, canonical_options)
            try:
                ok, reason = solver.run(), ""
            except PlanningFailure as e:
                ok, reason = False, str(e)
            if ok and not rec.gt[-1]["success"]:
                ok, reason = False, "success not reached before the last recorded step"
            ep["setup"].create_dataset("n_timesteps", data=rec.t)
        extra = {}
        if ik_mode != "legacy":
            extra = {"solver_version": solver.solver_version, "ik_mode": ik_mode, "ik_seed": ik_seed,
                     "ik_candidates": ik_candidates, "planning_audit": solver.trajectory_audit}
        if ik_mode == "canonical":
            extra.update(settle_steps=solver.canonical_settle_steps, q7_all_attempt_steps=solver.canonical_q7,
                         deferred_basket_checks=solver.canonical_basket_checks)
        if not ok:
            if ik_mode == "canonical" and (canonical_options or {}).get("preserve_failures"):
                failed_h5 = h5_path.with_name(h5_path.stem + ".failed.h5")
                os.replace(tmp, failed_h5)
                atomic_write(gt_path, "".join(json.dumps(r) + "\n" for r in rec.gt))
                extra.update(failed_h5=failed_h5, failed_gt=gt_path)
            tmp.unlink(missing_ok=True)
            return {"ok": False, "reason": reason, "steps": rec.t, **extra}
        atomic_write(gt_path, "".join(json.dumps(r) + "\n" for r in rec.gt))
        os.replace(tmp, h5_path)
        return {"ok": True, "h5": h5_path, "gt": gt_path, "steps": rec.t, "n_fruits": len(u.layout.fruits),
                "n_pick": u.layout.n_pick, "instruction": u.layout.instruction, **extra}
    finally:
        env.close()
        tmp.unlink(missing_ok=True)


def verify(h5_path: Path, gt_path: Path, episode: int) -> dict:
    import h5py
    rows = [json.loads(l) for l in gt_path.read_text().splitlines()]
    with h5py.File(h5_path) as f:
        ep = f[f"episode_{episode}"]
        steps = sorted(int(k.split("_")[1]) for k in ep if k.startswith("timestep_"))
        assert steps == list(range(len(steps))), "timesteps are not contiguous"
        assert [r["timestep"] for r in rows] == steps, "gt rows do not match h5 timesteps"
        last = ep[f"timestep_{steps[-1]}"]
        assert last["obs/front_rgb"].shape == (256, 256, 3) and last["obs/wrist_rgb"].shape == (256, 256, 3)
        n = len(last["obs/joint_state"][()])
        assert last["action/joint_action"].shape == (n + 1,) and last["obs/gripper_state"].shape == (2,)
        assert rows[-1]["success"], "episode did not end in success"
    return {"n_timesteps": len(steps), "final_success": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASK_NAME), required=True)
    parser.add_argument("--split", choices=sorted(SPLIT_BASE), default="train")
    parser.add_argument("--episodes", required=True, help="e.g. 0-99 or 0,5,7-9")
    parser.add_argument("--robot", choices=["rm75b_eg2", "rm65b_eg2"], default="rm75b_eg2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ik-mode", choices=["legacy", "nearest", "canonical"], default="legacy")
    parser.add_argument("--ik-seed", type=int, default=0)
    parser.add_argument("--ik-candidates", type=int, default=64)
    args = parser.parse_args()
    if args.ik_mode in ("nearest", "canonical") and (args.robot != "rm75b_eg2" or args.ik_seed < 0 or args.ik_candidates < 1):
        parser.error("nearest expert requires rm75b_eg2, nonnegative IK seed and positive candidate count")
    out = args.out / args.robot / args.split / args.task
    for sub in ("hdf5_files", "gt"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    code = code_version()
    for e in parse_range(args.episodes):
        manifest = out / f"ep{e}_manifest.json"
        if finished(manifest):
            continue
        failures = []
        for attempt in range(MAX_ATTEMPTS):
            seed = SPLIT_BASE[args.split] + 16 * e + attempt
            t0 = time.time()
            res = run_episode(args.task, e, seed, out, args.robot, args.ik_mode, args.ik_seed, args.ik_candidates)
            if not res["ok"]:
                failure = {"seed": seed, "reason": res["reason"], "steps": res["steps"]}
                if args.ik_mode != "legacy":
                    failure["planning_audit"] = res["planning_audit"]
                failures.append(failure)
                print(f"[{args.task} ep{e}] seed {seed} failed: {res['reason']}", flush=True)
                continue
            record = {"robot": args.robot, "task": args.task, "split": args.split, "episode": e, "seed": seed, "attempt": attempt, "code": code,
                      "failures": failures, "steps": res["steps"], "n_fruits": res["n_fruits"], "n_pick": res["n_pick"],
                      "instruction": res["instruction"], "seconds": round(time.time() - t0, 1),
                      "h5": str(res["h5"].relative_to(out)), "h5_sha256": sha256(res["h5"]),
                      "gt": str(res["gt"].relative_to(out)), "gt_sha256": sha256(res["gt"]),
                      "verification": verify(res["h5"], res["gt"], e)}
            if args.ik_mode != "legacy":
                record.update({key: res[key] for key in ("solver_version", "ik_mode", "ik_seed", "ik_candidates", "planning_audit")})
            atomic_write(manifest, json.dumps(record, indent=1))
            print(f"[{args.task} ep{e}] seed {seed} ok: {res['steps']} steps, {record['seconds']} s", flush=True)
            break
        else:
            atomic_write(out / f"ep{e}_FAILED.json", json.dumps({"failures": failures}, indent=1))


if __name__ == "__main__":
    main()
