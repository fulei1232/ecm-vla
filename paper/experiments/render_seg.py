#!/usr/bin/env python3
"""Replay recorded demonstrations in FruitPick-v1 and save ground-truth instance segmentation (front and wrist
camera) with RGB-D at every `--every`-th frame, for the fruit-segmentation benchmark.

Replaying the recorded joint actions from the recorded seed reproduces the episode (the rendered RGB is compared
with the h5 frame and the maximum absolute difference stored). Output per episode:
data/perception_fruit/seg/<set>/<task>_ep<e>.npz

    VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json CUDA_VISIBLE_DEVICES=0 \
        robomme_benchmark/.venv/bin/python ecm_vla/paper/experiments/render_seg.py --set val_a --shard 0 --of 8
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from common import ECM, episodes

OUT = ECM / "data/perception_fruit/seg"


def render(ep, every, out):
    import gymnasium as gym
    import fruit_picking.env  # noqa: F401
    env = gym.make("FruitPick-v1", task=ep.task, robot_uids="rm75b_eg2", obs_mode="rgb+depth+segmentation",
                   control_mode="pd_joint_pos", render_mode="rgb_array", sim_backend="cpu")
    try:
        obs, _ = env.reset(seed=ep.seed)
        u = env.unwrapped
        ids = {"fruit": [int(a.per_scene_id[0]) for a in u.fruits],
               "occluder": [int(a.per_scene_id[0]) for a in u.occluders],
               "tree": int(u.tree.per_scene_id[0])}
        rec = {k: [] for k in ("frame", "front_rgb", "front_depth", "front_seg", "front_extrinsic",
                               "wrist_rgb", "wrist_depth", "wrist_seg", "wrist_extrinsic", "rgb_max_diff")}
        with h5py.File(ep.h5) as f:
            g = f[f"episode_{ep.episode}"]
            n = int(g["setup/n_timesteps"][()])
            intr = {c: g[f"setup/{c}_camera_intrinsic"][()] for c in ("front", "wrist")}
            for t in range(n):
                if t % every == 0:
                    sd, sp = obs["sensor_data"], obs["sensor_param"]
                    rec["frame"].append(t)
                    diff = 0
                    for cam, key in (("front_camera", "front"), ("hand_camera", "wrist")):
                        rgb = sd[cam]["rgb"][0].cpu().numpy().astype(np.uint8)
                        diff = max(diff, int(np.abs(rgb.astype(int) - g[f"timestep_{t}/obs/{key}_rgb"][()].astype(int)).max()))
                        rec[f"{key}_rgb"].append(rgb)
                        rec[f"{key}_depth"].append(np.clip(sd[cam]["depth"][0, ..., 0].cpu().numpy(), 0, 32767).astype(np.int16))
                        rec[f"{key}_seg"].append(sd[cam]["segmentation"][0, ..., 0].cpu().numpy().astype(np.int16))
                        rec[f"{key}_extrinsic"].append(sp[cam]["extrinsic_cv"][0].cpu().numpy().astype(np.float32))
                    rec["rgb_max_diff"].append(diff)
                obs, *_ = env.step(g[f"timestep_{t}/action/joint_action"][()])
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in rec.items()},
                            front_intrinsic=intr["front"], wrist_intrinsic=intr["wrist"],
                            ids=json.dumps(ids), ripe=np.asarray(ep.ripe),
                            radius=np.asarray([fr["radius"] for fr in ep.layout.fruits]),
                            fruit_pos=np.asarray([[fr["pos"] for fr in ep.rows[t]["fruits"]] for t in rec["frame"]]),
                            occluded=np.asarray([o["fruit"] for o in ep.layout.occluders], dtype=np.int64))
        os.replace(tmp, out)
        return max(rec["rgb_max_diff"]), len(rec["frame"])
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-episode", type=int, default=None, help="only episodes with index below this")
    args = ap.parse_args()
    out_dir = OUT / args.set
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = [ep for ep in episodes(args.set) if args.max_episode is None or ep.episode < args.max_episode]
    eps = eps[args.shard::args.of][:args.limit]
    for ep in eps:
        out = out_dir / f"{ep.task}_ep{ep.episode}.npz"
        if out.exists():
            continue
        diff, n = render(ep, args.every, out)
        print(f"{ep.task} ep{ep.episode}: {n} frames, rgb max diff {diff}", flush=True)


if __name__ == "__main__":
    main()
