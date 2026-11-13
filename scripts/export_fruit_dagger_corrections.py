#!/usr/bin/env python3
"""Export successful K0-deviation/expert-recovery runs to RoboMME training HDF5.

The last 32 pre-intervention observations are history-only (`is_video_demo`).
Every execution sample begins at the first expert correction and carries an
expert action. Development and validation episodes are rejected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from v_a_common import ECM, atomic_json, read_json, sha256, timestamp

TASKS = {"FruitCount": "count", "FruitOccludedRipe": "occluded_ripe"}
MODES = {"reach_collect": ("reach_repair", 400), "carry_collect": ("carry_repair", 500),
         "carry_collect_late": ("carry_repair", 600),
         "reach_collect_auto": ("reach_repair", 700),
         "carry_collect_auto": ("carry_repair", 800),
         "grasp_collect_auto": ("grasp_repair", 900)}
TRAIN = ECM / "data/fruit_picking/rm75b_eg2/train"


def image_state(attempt: Path, frame: dict, cache: dict):
    shard = frame["shard"]
    if shard not in cache:
        with np.load(attempt / shard, allow_pickle=False) as data:
            cache[shard] = (data["front"].copy(), data["wrist"].copy(), data["state"].copy())
    front, wrist, state = (x[frame["shard_offset"]] for x in cache[shard])
    for value, key in ((front, "front_sha256"), (wrist, "wrist_sha256"), (state, "state_sha256")):
        if hashlib.sha256(value.tobytes(order="C")).hexdigest() != frame[key]:
            raise ValueError(f"frame {frame['frame_id']} differs from the raw shard")
    return front, wrist, state


def export_one(source: Path, out: Path, task: str, episode: int, mode: str, history: int):
    phase, offset = MODES[mode]
    attempt = source / mode / task / f"ep{episode}"
    result = read_json(attempt / "result.json")
    if result["outcome"] != "first_pick_complete" or not result["first_pick_success"]:
        raise ValueError(f"{attempt}: recovery did not succeed")
    trace = read_json(attempt / "phase_trace.json")
    frames = [json.loads(x) for x in (attempt / "raw_frames.jsonl").read_text().splitlines()]
    if len(trace) != result["executed_steps"] or len(frames) != len(trace) + 1:
        raise ValueError(f"{attempt}: trace/frame length mismatch")
    if any(r["control_step"] != i + 1 or r["pre_frame_id"] != i for i, r in enumerate(trace)):
        raise ValueError(f"{attempt}: trace step mismatch")
    if any(r["frame_id"] != i for i, r in enumerate(frames)):
        raise ValueError(f"{attempt}: frame ids are not contiguous")
    correction = next((i for i, r in enumerate(trace) if r["phase"] == phase), None)
    if correction is None or any(r["source"] != "expert" for r in trace[correction:]):
        raise ValueError(f"{attempt}: correction boundary or expert-only suffix invalid")
    begin = max(0, correction - history)
    local_exec = correction - begin
    selected = trace[begin:]
    number = offset + episode
    task_dir = out / TASKS[task]
    for sub in ("hdf5_files", "gt"):
        (task_dir / sub).mkdir(parents=True, exist_ok=True)
    stem = f"{task}_ep{number}_seed{result['environment_seed']}"
    h5 = task_dir / "hdf5_files" / f"{stem}.h5"
    gt_path = task_dir / "gt" / f"{stem}.jsonl"
    manifest = task_dir / f"ep{number}_manifest.json"
    if h5.exists() or gt_path.exists() or manifest.exists():
        raise FileExistsError(f"destination already exists: {manifest}")
    original_dir = TRAIN / TASKS[task]
    original = read_json(original_dir / f"ep{episode}_manifest.json")
    if result["environment_seed"] != original["seed"]:
        raise ValueError(f"{attempt}: source seed differs from train manifest")
    original_h5 = original_dir / original["h5"]
    tmp_h5 = h5.with_name(h5.name + ".tmp")
    tmp_gt = gt_path.with_name(gt_path.name + ".tmp")
    sidecar = []
    cache = {}
    try:
        with h5py.File(original_h5) as src, h5py.File(tmp_h5, "w") as dst:
            ep = dst.create_group(f"episode_{number}")
            src[f"episode_{episode}"].copy("setup", ep)
            del ep["setup/n_timesteps"]
            ep["setup"].create_dataset("n_timesteps", data=len(selected))
            ep["setup"].create_dataset("source_episode", data=episode)
            ep["setup"].create_dataset("correction_kind", data=phase)
            ep["setup"].create_dataset("exec_start_idx", data=local_exec)
            last_subgoal = None
            last_phase = None
            for local, row in enumerate(selected):
                raw = frames[row["pre_frame_id"]]
                front, wrist, state = image_state(attempt, raw, cache)
                if front.shape != (256, 256, 3) or wrist.shape != (256, 256, 3) or state.shape != (8,):
                    raise ValueError(f"{attempt}: unexpected observation shape at {row['control_step']}")
                action = np.asarray(row["executed_action"], dtype=np.float64)
                if action.shape != (8,):
                    raise ValueError(f"{attempt}: unexpected action shape at {row['control_step']}")
                g = ep.create_group(f"timestep_{local}")
                obs = g.create_group("obs")
                obs.create_dataset("front_rgb", data=front, compression="gzip", compression_opts=4)
                obs.create_dataset("wrist_rgb", data=wrist, compression="gzip", compression_opts=4)
                obs.create_dataset("joint_state", data=state[:7].astype(np.float32))
                # The official processor consumes only the first finger; the raw recorder stores that finger.
                obs.create_dataset("gripper_state", data=np.repeat(state[7], 2).astype(np.float64))
                g.create_group("action").create_dataset("joint_action", data=action)
                labels = row["labels"]
                subgoal = labels["subgoal"]
                info = g.create_group("info")
                for key in ("simple_subgoal", "grounded_subgoal", "simple_subgoal_online",
                            "grounded_subgoal_online"):
                    info.create_dataset(key, data=subgoal)
                info.create_dataset("is_subgoal_boundary", data=subgoal != last_subgoal or row["phase"] != last_phase)
                info.create_dataset("is_video_demo", data=local < local_exec)
                info.create_dataset("is_completed", data=False)
                info.create_dataset("phase", data=row["phase"])
                info.create_dataset("action_source", data=row["source"])
                info.create_dataset("source_control_step", data=row["control_step"])
                info.create_dataset("is_correction", data=row["phase"] == phase)
                last_subgoal, last_phase = subgoal, row["phase"]
                gt = raw["gt"]
                sidecar.append({"timestep": local, "source_control_step": row["control_step"],
                                "phase": row["phase"], "action_source": row["source"],
                                "target": labels.get("target"), "tcp_pos": gt["tcp_position"],
                                "fruits": [{"pos": pos, "attached": detached,
                                            "in_basket": inside}
                                           for pos, detached, inside in zip(gt["fruit_positions"],
                                                                             gt["detached"], gt["in_basket"])],
                                "fruit_distance": gt["fruit_distance"]})
        tmp_gt.write_text("".join(json.dumps(row) + "\n" for row in sidecar))
        os.replace(tmp_h5, h5)
        os.replace(tmp_gt, gt_path)
        correction_steps = sum(row["phase"] == phase for row in selected)
        record = {"robot": "rm75b_eg2", "task": TASKS[task], "split": "train",
                  "episode": number, "source_episode": episode, "seed": result["environment_seed"],
                  "source_attempt": str(attempt.resolve()), "source_result_sha256": sha256(attempt / "result.json"),
                  "source_trace_sha256": sha256(attempt / "phase_trace.json"),
                  "source_checkpoint": read_json(source / "plan.json")["checkpoint"],
                  "correction_kind": phase, "history_frames": local_exec,
                  "correction_steps": correction_steps, "expert_execution_samples": len(selected) - local_exec,
                  "policy_steps_before_correction": result["phase_steps"].get(mode.split("_")[0] + "_policy_steps", 0),
                  "intervention_distance_m": result.get(mode.split("_")[0] + "_goal_distance_m"),
                  "first_pick_success": True, "steps": len(selected), "n_pick": result["n_pick"],
                  "instruction": original["instruction"],
                  "h5": str(h5.relative_to(task_dir)), "h5_sha256": sha256(h5),
                  "gt": str(gt_path.relative_to(task_dir)), "gt_sha256": sha256(gt_path),
                  "export_script_sha256": sha256(Path(__file__).resolve()), "exported_utc": timestamp()}
        atomic_json(manifest, record)
        return record
    finally:
        tmp_h5.unlink(missing_ok=True)
        tmp_gt.unlink(missing_ok=True)
        if not manifest.is_file():
            h5.unlink(missing_ok=True)
            gt_path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    p.add_argument("--modes", nargs="+", choices=tuple(MODES), default=["reach_collect", "carry_collect"])
    p.add_argument("--history", type=int, default=32)
    p.add_argument("--eligible-only", action="store_true", help="skip failed or uncorrected attempts")
    p.add_argument("--resume", action="store_true", help="reuse complete, hash-verified exports")
    a = p.parse_args()
    source, out = a.source.resolve(), a.out.resolve()
    plan = read_json(source / "plan.json")
    if Path(plan["source_root"]).resolve() != TRAIN.resolve():
        p.error("only clean training episodes may be exported")
    if any(e < 0 or e >= 100 for e in a.episodes) or a.history < 1:
        p.error("episodes must be clean train 0-99 and history must be positive")
    records = []
    skipped = []
    for task in dict.fromkeys(a.tasks):
        for episode in sorted(set(a.episodes)):
            for mode in dict.fromkeys(a.modes):
                attempt = source / mode / task / f"ep{episode}"
                if a.eligible_only:
                    result_path = attempt / "result.json"
                    if not result_path.is_file():
                        skipped.append({"task": task, "episode": episode, "mode": mode,
                                        "reason": "missing_result"})
                        continue
                    result = read_json(result_path)
                    if result.get("outcome") != "first_pick_complete" or not result.get("first_pick_success"):
                        skipped.append({"task": task, "episode": episode, "mode": mode,
                                        "reason": result.get("outcome", "unknown")})
                        continue
                    trace = read_json(attempt / "phase_trace.json")
                    if not any(r["phase"] == MODES[mode][0] for r in trace):
                        skipped.append({"task": task, "episode": episode, "mode": mode,
                                        "reason": "no_expert_correction"})
                        continue
                number = MODES[mode][1] + episode
                task_dir = out / TASKS[task]
                manifest_path = task_dir / f"ep{number}_manifest.json"
                if a.resume and manifest_path.is_file():
                    record = read_json(manifest_path)
                    if record["source_result_sha256"] != sha256(attempt / "result.json") or \
                            record["h5_sha256"] != sha256(task_dir / record["h5"]) or \
                            record["gt_sha256"] != sha256(task_dir / record["gt"]):
                        raise ValueError(f"existing export differs: {manifest_path}")
                else:
                    record = export_one(source, out, task, episode, mode, a.history)
                records.append(record)
                print(json.dumps({"task": task, "episode": record["episode"],
                                  "correction_steps": record["correction_steps"],
                                  "expert_execution_samples": record["expert_execution_samples"]}), flush=True)
    summary = {"source": str(source), "source_checkpoint": plan["checkpoint"],
               "records": len(records), "per_kind": dict(Counter(r["correction_kind"] for r in records)),
               "skipped": skipped, "skipped_count": len(skipped),
               "correction_steps": sum(r["correction_steps"] for r in records),
               "expert_execution_samples": sum(r["expert_execution_samples"] for r in records),
               "history_frames": sum(r["history_frames"] for r in records),
               "episodes": [r["episode"] for r in records], "generated_utc": timestamp()}
    atomic_json(out / "collection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
