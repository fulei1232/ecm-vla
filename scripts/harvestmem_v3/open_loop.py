#!/usr/bin/env python3
"""Open-loop action check against recorded HarvestMem expert H5 replays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import sys

import h5py
import numpy as np

POLICY = Path("/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning")
sys.path.insert(0, str(POLICY / "examples/robomme"))

from subgoal import compose_prompt


def _timesteps(h5):
    ep = h5[next(key for key in h5.keys() if key.startswith("episode_"))]
    return sorted((int(name.split("_")[1]), name) for name in ep if name.startswith("timestep_"))


def _load_replay(path: Path):
    result = json.loads((path / "result.json").read_text())
    with h5py.File(path / "demo.h5", "r") as h5:
        ep = h5[next(key for key in h5.keys() if key.startswith("episode_"))]
        setup_instruction = ep["setup/task_goal"][()]
        if isinstance(setup_instruction, np.ndarray):
            setup_instruction = setup_instruction[0]
        if isinstance(setup_instruction, bytes):
            setup_instruction = setup_instruction.decode()
        frames = []
        actions = {}
        prompts = {}
        for timestep, name in _timesteps(h5):
            group = ep[name]
            obs = group["obs"]
            joint = np.asarray(obs["joint_state"][()], dtype=np.float32)
            grip = np.asarray(obs["gripper_state"][()], dtype=np.float32)
            frames.append({
                "front": np.asarray(obs["front_rgb"][()], dtype=np.uint8),
                "wrist": np.asarray(obs["wrist_rgb"][()], dtype=np.uint8),
                "state": np.concatenate([joint, grip[:1]]).astype(np.float32),
            })
            info = group["info"]
            grounded = info["grounded_subgoal"][()]
            if isinstance(grounded, bytes):
                grounded = grounded.decode()
            prompts[timestep] = str(grounded)
            if "joint_action" in group["action"]:
                actions[timestep] = np.asarray(group["action"]["joint_action"][()], dtype=np.float32)
    return {
        "task": result["task"], "episode": result["episode"],
        "instruction": str(setup_instruction), "frames": frames,
        "actions": actions, "grounded": prompts,
        "slot_pixels": result.get("harvestmem_v3_slot_pixels", {}),
        "path": str(path),
    }


def _prompt(replay, arm, timestep):
    instruction = replay["instruction"].lower()
    if arm != "sg":
        return instruction
    return compose_prompt(instruction, replay["grounded"][timestep])


def _replace_coord(prompt: str, replacement: tuple[int, int]) -> str:
    return re.sub(r"<\s*-?\d+\s*,\s*-?\d+\s*>", f"<{replacement[0]}, {replacement[1]}>", prompt, count=1)


def evaluate_replay(client, replay, arm, context: Path, sensitivity=False):
    context.write_text(json.dumps({"memory": "none" if arm == "nm" else "uniform",
                                   "trace": str(context.with_suffix(".trace.jsonl"))}))
    reset = client.reset()
    if not reset.get("reset_finished"):
        raise RuntimeError(f"policy reset failed: {reset}")
    from utils import pack_buffer

    frames = replay["frames"]
    actions = replay["actions"]
    trace_path = context.with_suffix(".trace.jsonl")
    trace_path.write_text("\n".join(json.dumps({"distance": 10.0}) for _ in frames) + "\n")
    ratios = []
    model_abs_sum = 0.0
    idle_abs_sum = 0.0
    abs_count = 0
    details = []
    sensitivity_rows = []
    timestep = 30
    if timestep not in actions:
        return {"ratios": [], "details": [], "sensitivity": []}
    first_sensitivity_done = False
    while timestep in actions:
        expert = np.stack([actions[t] for t in range(timestep, timestep + 16) if t in actions], axis=0)
        if len(expert) == 0:
            break
        history_end = min(timestep, len(frames) - 1)
        if timestep == 30:
            images = [frame["front"] for frame in frames[:history_end + 1]]
            states = [frame["state"] for frame in frames[:history_end + 1]]
            exec_start = history_end
        else:
            images = [frame["front"] for frame in frames[timestep - len(expert) + 1:timestep + 1]]
            states = [frame["state"] for frame in frames[timestep - len(expert) + 1:timestep + 1]]
            exec_start = len(images) - 1
        prompt = _prompt(replay, arm, timestep)
        buffer_payload = pack_buffer(images, states, exec_start)
        client.add_buffer(buffer_payload)
        observation = {"observation/image": frames[timestep]["front"],
                       "observation/wrist_image": frames[timestep]["wrist"],
                       "observation/state": frames[timestep]["state"],
                       "prompt": prompt}
        prediction = np.asarray(client.infer(observation)["actions"], dtype=np.float32)[:len(expert)]
        current = np.repeat(frames[timestep]["state"][:7][None, :], len(expert), axis=0)
        model_error = float(np.mean(np.abs(prediction[:, :7] - expert[:, :7])))
        idle_error = float(np.mean(np.abs(current - expert[:, :7])))
        model_abs_sum += float(np.abs(prediction[:, :7] - expert[:, :7]).sum())
        idle_abs_sum += float(np.abs(current - expert[:, :7]).sum())
        abs_count += int(expert[:, :7].size)
        ratio = model_error / idle_error if idle_error > 1e-9 else float("inf")
        ratios.append(ratio)
        details.append({"timestep": timestep, "model_error": model_error,
                        "idle_error": idle_error, "ratio": ratio,
                        "prompt": prompt, "steps": len(expert)})
        if sensitivity and arm == "sg" and not first_sensitivity_done and "pick" in prompt:
            pixels = [tuple(map(int, value)) for key, value in sorted(replay["slot_pixels"].items())]
            if len(pixels) >= 2:
                other = pixels[1] if pixels[0] == tuple(map(int, re.search(r"<(\d+), (\d+)>", prompt).groups())) else pixels[0]
                same = tuple(map(int, re.search(r"<(\d+), (\d+)>", prompt).groups()))
                variants = {
                    "other_slot": _replace_coord(prompt, other),
                    "same_plus4": _replace_coord(prompt, (same[0] + 4, same[1])),
                    "same_minus4": _replace_coord(prompt, (same[0] - 4, same[1])),
                }
                for label, variant_prompt in variants.items():
                    reset = client.reset()
                    if not reset.get("reset_finished"):
                        raise RuntimeError(f"policy reset failed during sensitivity probe: {reset}")
                    client.add_buffer(buffer_payload)
                    variant = np.asarray(client.infer({**observation, "prompt": variant_prompt})["actions"], dtype=np.float32)[:len(expert)]
                    sensitivity_rows.append({"label": label, "mean_abs_action_change": float(np.mean(np.abs(variant[:, :7] - prediction[:, :7]))),
                                             "prompt": variant_prompt})
                first_sensitivity_done = True
        timestep += len(expert)
    return {"ratios": ratios, "details": details, "sensitivity": sensitivity_rows,
            "aggregate_model_mae": model_abs_sum / abs_count if abs_count else None,
            "aggregate_idle_mae": idle_abs_sum / abs_count if abs_count else None,
            "aggregate_ratio": model_abs_sum / idle_abs_sum if idle_abs_sum else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--arm", choices=("sg", "k0", "nm"), required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    from openpi_client import websocket_client_policy as wcp

    specs = [
        ("count", 0, "train"), ("occluded_ripe", 0, "train"),
        ("count", 200, "dev"), ("count", 201, "dev"),
        ("occluded_ripe", 200, "dev"), ("occluded_ripe", 201, "dev"),
    ]
    replays = []
    for task, episode, split in specs:
        root = Path("results/harvestmem_v3_S1/replays") / ("train_clean" if split == "train" else "dev_expert") / task
        replays.append((_load_replay(root / f"ep{episode}_attempt0"), split))
    client = wcp.MMEVLAWebsocketClientPolicy("127.0.0.1", args.port)
    rows = []
    for replay, split in replays:
        result = evaluate_replay(client, replay, args.arm, args.context, args.sensitivity)
        rows.append({"split": split, "task": replay["task"], "episode": replay["episode"], **result})
    train_rows = [row for row in rows if row["split"] == "train"]
    train_model_abs = sum(detail["model_error"] * detail["steps"] * 7
                          for row in train_rows for detail in row["details"])
    train_idle_abs = sum(detail["idle_error"] * detail["steps"] * 7
                         for row in train_rows for detail in row["details"])
    train_aggregate = train_model_abs / train_idle_abs if train_idle_abs else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"arm": args.arm, "rows": rows,
                                    "train_combined_ratio": train_aggregate},
                                   indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"arm": args.arm, "rows": len(rows),
                      "train_aggregate_ratios": [row["aggregate_ratio"] for row in train_rows],
                      "train_combined_ratio": train_aggregate},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
