#!/usr/bin/env python3
"""Export expert correction clips from successful full-task train rollouts.

Each repair is a separate HDF5 episode: at most 32 preceding observations are
history-only, followed by an expert-only action suffix ending before the next
policy action or repair. This keeps the official contiguous execution contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import h5py
import numpy as np

from export_fruit_dagger_corrections import TRAIN, TASKS, image_state
from v_a_common import atomic_json, read_json, sha256, timestamp

REPAIRS = {"reach_repair", "grasp_repair", "carry_repair"}


def segments(trace: list[dict], history: int):
    starts = [i for i, row in enumerate(trace)
              if row["phase"] in REPAIRS and (i == 0 or trace[i - 1]["phase"] != row["phase"])]
    for seq, start in enumerate(starts):
        following = starts[seq + 1] if seq + 1 < len(starts) else len(trace)
        policy = next((i for i in range(start + 1, following) if trace[i]["source"] == "policy"), following)
        end = min(following, policy)
        begin = max(0, start - history)
        if not all(row["source"] == "expert" for row in trace[start:end]):
            raise ValueError(f"non-expert correction suffix at {start}")
        if end <= start or seq >= 100:
            raise ValueError(f"invalid correction segment at {start}")
        yield seq, begin, start, end


def export_segment(attempt: Path, out: Path, source_plan: dict, result: dict,
                   trace: list[dict], frames: list[dict], seq: int,
                   begin: int, correction: int, end: int):
    task, episode = result["task"], result["episode_id"]
    number = 10000 + episode * 100 + seq
    task_dir = out / TASKS[task]
    for sub in ("hdf5_files", "gt"):
        (task_dir / sub).mkdir(parents=True, exist_ok=True)
    stem = f"{task}_ep{number}_seed{result['environment_seed']}"
    h5 = task_dir / "hdf5_files" / f"{stem}.h5"
    gt_path = task_dir / "gt" / f"{stem}.jsonl"
    manifest = task_dir / f"ep{number}_manifest.json"
    if manifest.is_file():
        old = read_json(manifest)
        if (old["source_result_sha256"] != sha256(attempt / "result.json") or
                old["source_trace_sha256"] != sha256(attempt / "phase_trace.json") or
                old["h5_sha256"] != sha256(task_dir / old["h5"]) or
                old["gt_sha256"] != sha256(task_dir / old["gt"])):
            raise ValueError(f"existing full-task export differs: {manifest}")
        return old
    if h5.exists() or gt_path.exists():
        raise FileExistsError(f"incomplete prior export: {h5}")
    original_dir = TRAIN / TASKS[task]
    original = read_json(original_dir / f"ep{episode}_manifest.json")
    if result["environment_seed"] != original["seed"]:
        raise ValueError(f"train seed differs: {attempt}")
    selected = trace[begin:end]
    local_exec = correction - begin
    kind = trace[correction]["phase"]
    labels = trace[correction]["labels"]
    direct_steps = 0
    for row in trace[correction:end]:
        if row["phase"] != kind:
            break
        direct_steps += 1
    tmp_h5 = h5.with_name(h5.name + ".tmp")
    tmp_gt = gt_path.with_name(gt_path.name + ".tmp")
    cache = {}
    sidecar = []
    try:
        with h5py.File(original_dir / original["h5"], "r") as src, h5py.File(tmp_h5, "w") as dst:
            ep = dst.create_group(f"episode_{number}")
            src[f"episode_{episode}"].copy("setup", ep)
            del ep["setup/n_timesteps"]
            ep["setup"].create_dataset("n_timesteps", data=len(selected))
            ep["setup"].create_dataset("source_episode", data=episode)
            ep["setup"].create_dataset("correction_kind", data=kind)
            ep["setup"].create_dataset("exec_start_idx", data=local_exec)
            previous_subgoal = previous_phase = None
            for local, row in enumerate(selected):
                raw = frames[row["pre_frame_id"]]
                front, wrist, state = image_state(attempt, raw, cache)
                action = np.asarray(row["executed_action"], dtype=np.float64)
                if front.shape != (256, 256, 3) or wrist.shape != (256, 256, 3) or \
                        state.shape != (8,) or action.shape != (8,):
                    raise ValueError(f"observation/action shape differs: {attempt} step {local}")
                g = ep.create_group(f"timestep_{local}")
                obs = g.create_group("obs")
                obs.create_dataset("front_rgb", data=front, compression="gzip", compression_opts=4)
                obs.create_dataset("wrist_rgb", data=wrist, compression="gzip", compression_opts=4)
                obs.create_dataset("joint_state", data=state[:7].astype(np.float32))
                obs.create_dataset("gripper_state", data=np.repeat(state[7], 2).astype(np.float64))
                g.create_group("action").create_dataset("joint_action", data=action)
                subgoal = row["labels"]["subgoal"]
                info = g.create_group("info")
                for key in ("simple_subgoal", "grounded_subgoal", "simple_subgoal_online",
                            "grounded_subgoal_online"):
                    info.create_dataset(key, data=subgoal)
                info.create_dataset("is_subgoal_boundary", data=subgoal != previous_subgoal or
                                    row["phase"] != previous_phase)
                info.create_dataset("is_video_demo", data=local < local_exec)
                info.create_dataset("is_completed", data=end == len(trace) and local == len(selected) - 1)
                info.create_dataset("phase", data=row["phase"])
                info.create_dataset("action_source", data=row["source"])
                info.create_dataset("source_control_step", data=row["control_step"])
                info.create_dataset("is_correction", data=row["phase"] == kind and local >= local_exec)
                previous_subgoal, previous_phase = subgoal, row["phase"]
                gt = raw["gt"]
                sidecar.append({"timestep": local, "source_control_step": row["control_step"],
                                "phase": row["phase"], "action_source": row["source"],
                                "target": row["labels"].get("target"), "pick_index": row["labels"].get("pick_index"),
                                "tcp_pos": gt["tcp_position"],
                                "fruits": [{"pos": pos, "attached": detached, "in_basket": inside}
                                           for pos, detached, inside in zip(gt["fruit_positions"],
                                                                             gt["detached"], gt["in_basket"])],
                                "fruit_distance": gt["fruit_distance"]})
        tmp_gt.write_text("".join(json.dumps(row) + "\n" for row in sidecar))
        os.replace(tmp_h5, h5)
        os.replace(tmp_gt, gt_path)
        record = {"robot": "rm75b_eg2", "task": TASKS[task], "split": "train",
                  "episode": number, "source_episode": episode, "seed": result["environment_seed"],
                  "source_attempt": str(attempt.resolve()),
                  "source_result_sha256": sha256(attempt / "result.json"),
                  "source_trace_sha256": sha256(attempt / "phase_trace.json"),
                  "source_checkpoint": source_plan["checkpoint"],
                  "correction_kind": kind, "source_pick_index": labels.get("pick_index"),
                  "target": labels.get("target"), "history_frames": local_exec,
                  "correction_steps": direct_steps, "expert_execution_samples": len(selected) - local_exec,
                  "policy_steps_before_correction": correction,
                  "first_pick_success": result["first_pick_success"],
                  "full_task_success": True, "steps": len(selected), "n_pick": result["n_pick"],
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
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--stop", type=int, default=64)
    p.add_argument("--history", type=int, default=32)
    a = p.parse_args()
    source, out = a.source.resolve(), a.out.resolve()
    plan = read_json(source / "plan.json")
    if Path(plan["source_root"]).resolve() != TRAIN.resolve() or not 0 <= a.first < a.stop <= 100:
        p.error("only clean train episodes 0-99 may be exported")
    if "full_collect" not in plan["modes"] or a.history < 1:
        p.error("source must contain full_collect and history must be positive")
    records = []
    skipped = []
    for task in TASKS:
        for episode in range(a.first, a.stop):
            attempt = source / "full_collect" / task / f"ep{episode}"
            result_path = attempt / "result.json"
            if not result_path.is_file():
                skipped.append({"task": task, "episode": episode, "reason": "missing_result"})
                continue
            result = read_json(result_path)
            if not result.get("task_success") or result.get("wrong_baskets", 0):
                skipped.append({"task": task, "episode": episode,
                                "reason": result.get("outcome", "not_successful")})
                continue
            trace = read_json(attempt / "phase_trace.json")
            frames = [json.loads(x) for x in (attempt / "raw_frames.jsonl").read_text().splitlines()]
            if len(trace) != result["executed_steps"] or len(frames) != len(trace) + 1 or \
                    any(r["control_step"] != i + 1 or r["pre_frame_id"] != i for i, r in enumerate(trace)) or \
                    any(r["frame_id"] != i for i, r in enumerate(frames)):
                raise ValueError(f"source trace/frame mismatch: {attempt}")
            found = False
            for seq, begin, correction, end in segments(trace, a.history):
                record = export_segment(attempt, out, plan, result, trace, frames,
                                        seq, begin, correction, end)
                records.append(record)
                found = True
                print(json.dumps({"task": task, "episode": episode, "segment": seq,
                                  "pick_index": record["source_pick_index"],
                                  "kind": record["correction_kind"]}), flush=True)
            if not found:
                skipped.append({"task": task, "episode": episode, "reason": "no_expert_correction"})
    summary = {"source": str(source), "source_checkpoint": plan["checkpoint"],
               "records": len(records), "attempts_with_corrections": len(set((r["task"], r["source_episode"])
                                                                   for r in records)),
               "per_kind": dict(Counter(r["correction_kind"] for r in records)),
               "per_pick_index": dict(Counter(str(r["source_pick_index"]) for r in records)),
               "correction_steps": sum(r["correction_steps"] for r in records),
               "expert_execution_samples": sum(r["expert_execution_samples"] for r in records),
               "skipped": skipped, "skipped_count": len(skipped), "generated_utc": timestamp()}
    atomic_json(out / "collection_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
