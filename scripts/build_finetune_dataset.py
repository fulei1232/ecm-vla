#!/usr/bin/env python3
"""Preprocess the train demonstrations with ground truth (data/train_demos_gt) into the official MME-VLA
training format, plus one distance file per episode for the event-keyframe memory.

Official layout, read unchanged by mme_vla_suite.training.dataset.RoboMMEDataset:
  <out>/data/<sample>.pkl                          execution samples, numbered over all episodes
  <out>/features/episode_<g>/token_emb_<t>.npy     per-frame features (official DatasetProcessor)
  <out>/meta/stats.json
Added:
  <out>/features/episode_<g>/distance.npy   event_memory.gt_distance of every frame (NaN = unknown)
  <out>/meta/episodes.json                  g -> task, train episode, seed, sha256s, timesteps, sample range
  <out>/meta/build.json                     inputs, code identity and the checks of `finish`

Episodes are numbered in a fixed order (tasks as given, train episodes ascending) and their sample ids
are assigned beforehand, so shards on several GPUs give the same output as the official sequential
builder. Every frame goes through DatasetProcessor._process_episode.

    plan    index episodes, write distances (CPU)
    shard   process the episodes g with g % of == index; resumable per episode
    finish  check counts and contiguity, write stats.json and build.json
    check-encoder  stored features vs the served checkpoint's vision encoder on a few frames
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

ECM = Path(__file__).resolve().parents[1]
WORKSPACE = ECM.parent
sys.path.insert(0, str(ECM / "src"))
from memory_management.validation.event_memory import gt_distance  # noqa: E402

TASKS = ("PatternLock", "BinFill", "VideoUnmaskSwap")
SOURCE = ECM / "data/train_demos_gt"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = Path(path).with_name(Path(path).name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def plan(out, tasks, episodes):
    import h5py
    from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step
    if (out / "meta/episodes.json").exists():
        raise SystemExit(f"{out} is already planned")
    for sub in ("data", "features", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows, sample, frame = [], 0, 0
    for task in tasks:
        for e in episodes:
            man = read_json(SOURCE / task / f"ep{e:02d}_manifest.json")
            h5, gt = WORKSPACE / man["h5_path"], WORKSPACE / man["gt_path"]
            if sha256(h5) != man["h5_sha256"] or sha256(gt) != man["gt_sha256"]:
                raise SystemExit(f"{task} ep{e}: source files differ from the manifest")
            with h5py.File(h5, "r") as f:
                ep = f[f"episode_{e}"]
                steps = sum(1 for k in ep.keys() if k.startswith("timestep_"))
                start = first_execution_step(ep)
            rows_gt = [json.loads(x) for x in gt.read_text().splitlines() if x.strip()]
            if [r["timestep"] for r in rows_gt] != list(range(steps)):
                raise SystemExit(f"{task} ep{e}: ground-truth rows do not match the timesteps")
            g = len(rows)
            distance = np.asarray([gt_distance(task, r["gt"]) for r in rows_gt], dtype=np.float64)
            (out / f"features/episode_{g}").mkdir(exist_ok=True)
            np.save(out / f"features/episode_{g}/distance.npy", distance)
            rows.append({"global_episode": g, "task": task, "train_episode": e, "seed": man["used_seed"],
                         "h5": str(h5), "h5_episode": e, "h5_sha256": man["h5_sha256"], "gt_sha256": man["gt_sha256"],
                         "timesteps": steps, "exec_start_idx": start, "first_sample": sample,
                         "exec_samples": steps - start, "first_frame": frame,
                         "distance_known": int(np.isfinite(distance).sum())})
            sample += steps - start
            frame += steps
            print(f"{task} ep{e} -> {g}: {steps} frames, exec from {start}", flush=True)
    atomic_json(out / "meta/episodes.json", rows)
    print(json.dumps({"episodes": len(rows), "execution_samples": sample, "total_samples": frame}))


def processor(out):
    from mme_vla_suite.dataset_builder.build_robomme_dataset import DatasetProcessor
    p = DatasetProcessor.__new__(DatasetProcessor)  # __init__ would wipe the output directory
    p.dataset_path, p.execution_horizon, p.visualize, p.max_episodes = str(out), 16, False, None
    p.feature_path, p.data_path, p.meta_path = str(out / "features"), str(out / "data"), str(out / "meta")
    return p


def shard(out, index, of):
    import h5py
    from mme_vla_suite.shared.mem_buffer import MemoryBuffer
    rows = [r for r in read_json(out / "meta/episodes.json") if r["global_episode"] % of == index]
    p = processor(out)
    buffer = MemoryBuffer(num_views=1, compute_token_drop_score=True, token_drop_stride=p.execution_horizon // 2,
                          prepare_buffer=True)
    for r in rows:
        g = r["global_episode"]
        directory = out / f"features/episode_{g}"
        if (directory / "done.json").exists():
            continue
        for path in directory.glob("token_emb_*.npy"):
            path.unlink()
        for s in range(r["first_sample"], r["first_sample"] + r["exec_samples"]):
            (out / f"data/{s}.pkl").unlink(missing_ok=True)
        start = time.monotonic()
        with h5py.File(r["h5"], "r") as f:
            nxt, buffer, exec_id, total_id = p._process_episode(f, r["h5_episode"], g, buffer, r["first_sample"],
                                                                r["first_frame"])
        if (nxt, exec_id, total_id) != (g + 1, r["first_sample"] + r["exec_samples"], r["first_frame"] + r["timesteps"]):
            raise RuntimeError(f"episode {g}: sample ids differ from the plan")
        atomic_json(directory / "done.json", {"seconds": round(time.monotonic() - start, 1), "shard": index,
                                              "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


def finish(out):
    rows = read_json(out / "meta/episodes.json")
    problems, samples, frames = [], 0, 0
    for r in rows:
        directory = out / f"features/episode_{r['global_episode']}"
        if not (directory / "done.json").exists():
            problems.append(f"episode {r['global_episode']} not processed")
            continue
        n = sum(1 for _ in directory.glob("token_emb_*.npy"))
        d = np.load(directory / "distance.npy")
        if n != r["timesteps"] or len(d) != r["timesteps"]:
            problems.append(f"episode {r['global_episode']}: {n} features, {len(d)} distances, {r['timesteps']} frames")
        if r["first_sample"] != samples or r["first_frame"] != frames:
            problems.append(f"episode {r['global_episode']}: sample ids not contiguous")
        samples += r["exec_samples"]
        frames += r["timesteps"]
    pkls = sum(1 for _ in (out / "data").glob("*.pkl"))
    if pkls != samples or not all((out / f"data/{s}.pkl").exists() for s in (0, samples - 1)):
        problems.append(f"{pkls} samples on disk, {samples} planned")
    code = [Path(__file__).resolve(), ECM / "src/memory_management/validation/event_memory.py"]
    policy = WORKSPACE / "acm_vla/vendor/robomme_policy_learning/src/mme_vla_suite"
    code += [policy / "dataset_builder/build_robomme_dataset.py", policy / "dataset_builder/robomme_h5_utils.py",
             policy / "shared/mem_buffer.py"]
    report = {"passed": not problems, "problems": problems, "episodes": len(rows), "execution_samples": samples,
              "total_samples": frames, "per_task": {t: sum(r["exec_samples"] for r in rows if r["task"] == t)
                                                    for t in dict.fromkeys(r["task"] for r in rows)},
              "episodes_sha256": sha256(out / "meta/episodes.json"),
              "code_identity": [{"path": str(p), "sha256": sha256(p)} for p in code],
              "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not problems:
        atomic_json(out / "meta/stats.json", {"execution_samples": samples, "total_samples": frames})
    atomic_json(out / "meta/build.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "code_identity"}, indent=1))
    return 0 if not problems else 1


def check_encoder(out, checkpoint, frames):
    """Stored 4x4 features of episode 0 vs the served policy's own encoder (policy._vision_encode)."""
    import h5py
    import jax
    import jax.numpy as jnp
    import einops
    from openpi.shared import image_tools
    from mme_vla_suite.shared.data_utils import pool_tokens_to_size
    from mme_vla_suite.policies.policy_config import create_trained_policy
    from mme_vla_suite.training import config as _config
    r = read_json(out / "meta/episodes.json")[0]
    policy = create_trained_policy(_config.get_config("mme_vla_suite"), Path(checkpoint), seed=7)
    with h5py.File(r["h5"], "r") as f:
        ep = f[f"episode_{r['h5_episode']}"]
        images = np.stack([ep[f"timestep_{t}"]["obs"]["front_rgb"][()] for t in range(frames)])
    x = jnp.array(images[:, None].astype(np.float32) / 255.0 * 2.0 - 1.0)
    x = einops.rearrange(image_tools.resize_with_pad(einops.rearrange(x, "t v h w c -> (t v) h w c"), 224, 224),
                         "(t v) h w c -> t v h w c", t=frames, v=1)
    served = np.asarray(jax.device_get(pool_tokens_to_size(policy._vision_encode(x), 16)), dtype=np.float32)
    stored = np.stack([np.load(out / f"features/episode_0/token_emb_{t}.npy", allow_pickle=True).item()["image_emb_4x4"]
                       for t in range(frames)]).astype(np.float32)
    diff = np.abs(served - stored)
    result = {"frames": frames, "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
              "mean_abs_value": float(np.abs(stored).mean()), "checkpoint": str(checkpoint),
              "shapes": [list(served.shape), list(stored.shape)]}
    atomic_json(out / "meta/encoder_check.json", result)
    print(json.dumps(result, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["plan", "shard", "finish", "check-encoder"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--episodes", type=int, nargs=2, default=[0, 100], metavar=("FIRST", "STOP"))
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--frames", type=int, default=8)
    a = p.parse_args()
    out = a.out.resolve()
    if a.command == "plan":
        plan(out, a.tasks, list(range(*a.episodes)))
    elif a.command == "shard":
        shard(out, a.index, a.of)
    elif a.command == "finish":
        return finish(out)
    else:
        check_encoder(out, a.checkpoint, a.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
