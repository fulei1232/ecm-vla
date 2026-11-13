#!/usr/bin/env python3
"""SG prompt sensitivity probe at fixed 16-step replay boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

from open_loop import _load_replay, _replace_coord
from subgoal import compose_prompt

POLICY = Path("/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning")
sys.path.insert(0, str(POLICY / "examples/robomme"))


def prompt_variants(replay, timestep):
    instruction = replay["instruction"].lower()
    grounded = replay["grounded"][timestep]
    original = compose_prompt(instruction, grounded)
    points = [tuple(map(int, value)) for _, value in sorted(replay["slot_pixels"].items())]
    match = re.search(r"<(\d+),\s*(\d+)>", original)
    same = (int(match.group(1)), int(match.group(2))) if match else points[0]
    other = next((point for point in points if point != same), same)
    variants = {
        "original": original,
        "other_slot": _replace_coord(original, other),
        "same_plus4": _replace_coord(original, (same[0] + 4, same[1])),
        "same_minus4": _replace_coord(original, (same[0] - 4, same[1])),
        "no_subgoal": instruction,
    }
    if grounded.startswith("pick"):
        variants["type_swap"] = compose_prompt(instruction, "return to rest and hold still")
    else:
        highest = max(points, key=lambda point: point[1])
        variants["type_swap"] = compose_prompt(instruction, f"pick the fruit at <{highest[0]}, {highest[1]}>")
    return variants


def block_starts(replay):
    starts = list(range(30, max(replay["actions"]) + 1, 16))
    labels = replay["grounded"]
    transitions = [t for t in starts if t > 30 and labels.get(t) != labels.get(t - 1)]
    pick_starts = [t for t in starts if labels.get(t, "").startswith("pick")]
    a = [t for t in pick_starts if any(labels.get(t - 1) == "place the fruit in the basket" for _ in [0])]
    # Prefer the first block whose preceding frame is PLACE; tolerate a label
    # change inside the preceding block by taking the first pick block after it.
    a = []
    for t in pick_starts:
        previous = max((x for x in starts if x < t), default=None)
        if previous is not None and labels.get(previous, "").startswith("place"):
            a.append(t)
    b = []
    for t in pick_starts:
        if any(abs(t - start) >= 16 for start in a):
            b.append(t)
    b = b[:5]
    c = [t for t in starts if labels.get(t, "") == "return to rest and hold still"] if replay["task"] == "count" else []
    return {"a": a[:max(1, len(a))], "b": b, "c": c[:1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from openpi_client import websocket_client_policy as wcp
    from utils import pack_buffer

    client = wcp.MMEVLAWebsocketClientPolicy("127.0.0.1", args.port)
    records = []
    for task in ("count", "occluded_ripe"):
        for episode in range(200, 204):
            root = Path("results/harvestmem_v3_S1/replays/dev_expert") / task / f"ep{episode}_attempt0"
            replay = _load_replay(root)
            points = block_starts(replay)
            for point_type, starts in points.items():
                for timestep in starts:
                    if timestep not in replay["actions"]:
                        continue
                    expert = np.stack([replay["actions"][t] for t in range(timestep, timestep + 16)
                                       if t in replay["actions"]], axis=0)
                    if len(expert) == 0:
                        continue
                    if timestep == 30:
                        images = [frame["front"] for frame in replay["frames"][:31]]
                        states = [frame["state"] for frame in replay["frames"][:31]]
                    else:
                        images = [frame["front"] for frame in replay["frames"][timestep - len(expert) + 1:timestep + 1]]
                        states = [frame["state"] for frame in replay["frames"][timestep - len(expert) + 1:timestep + 1]]
                    buffer = pack_buffer(images, states, len(images) - 1)
                    variants = prompt_variants(replay, timestep)
                    client.reset()
                    client.add_buffer(buffer)
                    base_obs = {"observation/image": replay["frames"][timestep]["front"],
                                "observation/wrist_image": replay["frames"][timestep]["wrist"],
                                "observation/state": replay["frames"][timestep]["state"]}
                    outputs = {}
                    for label, prompt in variants.items():
                        if label != "original":
                            client.reset(); client.add_buffer(buffer)
                        outputs[label] = np.asarray(client.infer({**base_obs, "prompt": prompt})["actions"], dtype=np.float32)[:len(expert)]
                    idle = np.repeat(replay["frames"][timestep]["state"][:7][None, :], len(expert), axis=0)
                    idle_error = float(np.mean(np.abs(idle - expert[:, :7])))
                    original = outputs["original"][:, :7]
                    changes = {label: float(np.mean(np.abs(value[:, :7] - original))) / idle_error
                               for label, value in outputs.items() if label != "original"}
                    type_row = {"type_swap_rest_distance_original": None,
                                "type_swap_rest_distance_variant": None,
                                "type_swap_direction": None}
                    if "type_swap" in outputs:
                        rest = replay["frames"][0]["state"][:7]
                        orig_dist = float(np.linalg.norm(original[-1] - rest))
                        swap_dist = float(np.linalg.norm(outputs["type_swap"][-1, :7] - rest))
                        type_row.update(type_swap_rest_distance_original=orig_dist,
                                        type_swap_rest_distance_variant=swap_dist,
                                        type_swap_direction="closer" if swap_dist < orig_dist else "farther" if swap_dist > orig_dist else "same")
                    records.append({"task": task, "episode": episode, "point_type": point_type,
                                    "timestep": timestep, "grounded": replay["grounded"][timestep],
                                    "changes_over_idle": changes, **type_row})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
