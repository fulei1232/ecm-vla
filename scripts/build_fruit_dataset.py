#!/usr/bin/env python3
"""Preprocess the fruit-picking train demonstrations (data/fruit_picking/<robot>/train) into the official MME-VLA
training format, plus the per-frame distance of fruit_distance.py and the robot's own normalisation statistics.

Same layout and processing as build_finetune_dataset.py, whose `shard`, `finish` and `check-encoder` are used
unchanged; only the episode index differs:
    plan        index episodes (tasks as given; per task the --part sources in order, episodes ascending), verify
                sha256s, write distance.npy (CPU)
    shard       official DatasetProcessor._process_episode per episode (GPU)
    finish      checks, stats.json, build.json (+ the identity of this script and fruit_distance.py)
    norm-stats  official compute_norm_stats over the built samples -> <out>/assets/robomme/norm_stats.json
                (the asset id stays "robomme" so the trained checkpoint is served unchanged)
    check-encoder  stored features vs the served checkpoint's vision encoder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ECM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECM / "src"))
sys.path.insert(0, str(ECM / "scripts"))
import build_finetune_dataset as bfd  # noqa: E402
from memory_management import fruit_distance as fd  # noqa: E402

TASKS = {"count": "FruitCount", "occluded_ripe": "FruitOccludedRipe"}
SOURCE = ECM / "data/fruit_picking/rm75b_eg2/train"


def plan(out, tasks, parts, skip_missing=False, distance_signal='mature'):
    """`parts`: [(source split directory, train episodes)], indexed per task in the given order."""
    import h5py
    from mme_vla_suite.dataset_builder.robomme_h5_utils import first_execution_step
    if (out / "meta/episodes.json").exists():
        raise SystemExit(f"{out} is already planned")
    for sub in ("data", "features", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows, sample, frame = [], 0, 0
    for task, (source, episodes) in ((t, part) for t in tasks for part in parts):
        for e in episodes:
            if skip_missing and not (source / task / f"ep{e}_manifest.json").is_file():
                continue
            man = bfd.read_json(source / task / f"ep{e}_manifest.json")
            h5, gt = source / task / man["h5"], source / task / man["gt"]
            if bfd.sha256(h5) != man["h5_sha256"] or bfd.sha256(gt) != man["gt_sha256"]:
                raise SystemExit(f"{task} ep{e}: source files differ from the manifest")
            with h5py.File(h5, "r") as f:
                ep = f[f"episode_{man['episode']}"]
                steps = sum(1 for k in ep.keys() if k.startswith("timestep_"))
                start = first_execution_step(ep)
                ripe = fd.ripe_flags(ep["setup/layout_json"][()])
            rows_gt = [json.loads(x) for x in gt.read_text().splitlines() if x.strip()]
            if [r["timestep"] for r in rows_gt] != list(range(steps)):
                raise SystemExit(f"{task} ep{e}: ground-truth rows do not match the timesteps")
            g = len(rows)
            if distance_signal == 'any_hanging':
                from memory_management.fruit_v2_distance import from_gt_row
                distance = np.asarray([from_gt_row(r) for r in rows_gt], dtype=np.float64)
            elif distance_signal == 'mature':
                distance = np.asarray([fd.from_gt_row(r, ripe) for r in rows_gt], dtype=np.float64)
            else:
                raise ValueError('Unknown distance signal')
            (out / f"features/episode_{g}").mkdir(exist_ok=True)
            np.save(out / f"features/episode_{g}/distance.npy", distance)
            rows.append({"global_episode": g, "task": TASKS[task], "source": str(source), "train_episode": e,
                         "seed": man["seed"], "noise": man.get("noise"),
                         "h5": str(h5), "h5_episode": man["episode"], "h5_sha256": man["h5_sha256"],
                         "gt_sha256": man["gt_sha256"], "timesteps": steps, "exec_start_idx": start,
                         "first_sample": sample, "exec_samples": steps - start, "first_frame": frame,
                         "distance_known": int(np.isfinite(distance).sum()), "n_pick": man["n_pick"] if "n_pick" in man else man["N"]})
            if distance_signal != 'mature':rows[-1]['distance_signal'] = distance_signal
            sample += steps - start
            frame += steps
            print(f"{task} ep{e} -> {g}: {steps} frames, exec from {start}", flush=True)
    bfd.atomic_json(out / "meta/episodes.json", rows)
    print(json.dumps({"episodes": len(rows), "execution_samples": sample, "total_samples": frame}))


def finish(out):
    code = bfd.finish(out)
    build = bfd.read_json(out / "meta/build.json")
    build["code_identity"] += [{"path": str(p), "sha256": bfd.sha256(p)}
                               for p in (Path(__file__).resolve(), ECM / "src/memory_management/fruit_distance.py")]
    if any(r.get('distance_signal') == 'any_hanging' for r in bfd.read_json(out / 'meta/episodes.json')):
        build['distance_signal'] = 'any_hanging'
        build['code_identity'].append({'path':str(ECM/'src/memory_management/fruit_v2_distance.py'),
                                      'sha256':bfd.sha256(ECM/'src/memory_management/fruit_v2_distance.py')})
    bfd.atomic_json(out / "meta/build.json", build)
    return code


def norm_stats(out):
    """Run from the policy repository root (relative config paths)."""
    import openpi.shared.normalize as normalize
    import mme_vla_suite.training.config as _config
    sys.path.insert(0, str(Path.cwd() / "scripts"))  # importable by name in the loader's spawned workers
    import compute_norm_stats as official
    config = _config.get_config("mme_vla_suite")
    data_config = config.data.create(config.assets_dirs, config.model)
    batch_size = min(128, bfd.read_json(out / "meta/stats.json")["execution_samples"])
    loader, batches = official.create_data_loader(dataset_path=str(out), data_config=data_config, history_config=None,
                                                  action_horizon=config.model.action_horizon, batch_size=batch_size,
                                                  num_workers=16, compute_norm_stats=True)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    start = time.monotonic()
    for batch in loader:
        for key in stats:
            stats[key].update(np.asarray(batch[key]))
    values = {key: s.get_statistics() for key, s in stats.items()}
    normalize.save(out / "assets/robomme", values)
    bfd.atomic_json(out / "meta/norm_stats.json", {
        "batches": batches, "batch_size": batch_size, "seconds": round(time.monotonic() - start, 1),
        "official_script_sha256": bfd.sha256(Path.cwd() / "scripts/compute_norm_stats.py"),
        "norm_stats_sha256": bfd.sha256(out / "assets/robomme/norm_stats.json"),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({k: {s: np.round(getattr(v, s), 3).tolist() for s in ("q01", "q99")} for k, v in values.items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["plan", "shard", "finish", "norm-stats", "check-encoder"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--episodes", type=int, nargs=2, default=[0, 100], metavar=("FIRST", "STOP"))
    p.add_argument("--part", nargs=3, action="append", metavar=("SOURCE", "FIRST", "STOP"),
                   help="plan from several sources (repeatable; replaces --source/--episodes)")
    p.add_argument("--skip-missing", action="store_true", help="allow sparse accepted correction manifests")
    p.add_argument('--distance-signal', choices=['mature','any_hanging'], default='mature')
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--frames", type=int, default=8)
    a = p.parse_args()
    out = a.out.resolve()
    if a.command == "plan":
        parts = [(Path(s).resolve(), range(int(b), int(c))) for s, b, c in a.part] if a.part else \
            [(a.source.resolve(), range(*a.episodes))]
        plan(out, a.tasks, parts, a.skip_missing, a.distance_signal)
    elif a.command == "shard":
        bfd.shard(out, a.index, a.of)
    elif a.command == "finish":
        return finish(out)
    elif a.command == "norm-stats":
        norm_stats(out)
    else:
        bfd.check_encoder(out, a.checkpoint, a.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
