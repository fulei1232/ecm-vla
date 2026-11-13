"""Distance signal of the fruit-picking tasks (fruit_picking/env.py) for the event-keyframe memory.

d_t = distance from the tcp to the centre of the nearest pickable fruit: ripe (every fruit of `count`, the red
ones of `occluded_ripe`) and still on the tree (not yet detached by a grasp). It does not depend on which fruit
a demonstrator or policy is going for, so training (recorded ground truth) and closed-loop evaluation (live
simulator state) use the same definition; a distance module on the robot reports the same quantity. When no
pickable fruit is left the signal is NO_TARGET_M (far from every target), not unknown: unknown frames carry the
previous near/far state in event_keyframes.detect_events, which would keep the last pick's event open.
"""
from __future__ import annotations

import json

import numpy as np

NO_TARGET_M = 10.0


def distance(tcp, positions, ripe, detached) -> float:
    tcp = np.asarray(tcp, dtype=np.float64).reshape(3)
    candidates = [np.asarray(p, dtype=np.float64).reshape(3) for p, r, d in zip(positions, ripe, detached) if r and not d]
    if not candidates:
        return NO_TARGET_M
    return float(min(np.linalg.norm(tcp - p) for p in candidates))


def ripe_flags(layout) -> list[bool]:
    """`layout`: fruit_picking.layout.Layout, its JSON (h5 setup/layout_json) or the parsed dict."""
    if isinstance(layout, (str, bytes)):
        layout = json.loads(layout)
    fruits = layout["fruits"] if isinstance(layout, dict) else layout.fruits
    return [bool(f["ripe"]) for f in fruits]


def from_gt_row(row, ripe) -> float:
    """One row of the generator's ground-truth sidecar (state before the step's action)."""
    return distance(row["tcp_pos"], [f["pos"] for f in row["fruits"]], ripe, [f["attached"] for f in row["fruits"]])


def from_env(env) -> float:
    """Live FruitPick-v1 state (the unwrapped env)."""
    return distance(env.agent.tcp_pose.p[0].cpu().numpy(), env.fruit_positions(), ripe_flags(env.layout), env.attached)


class RecordedDistance:
    """Simulator stand-in for the robot's distance module, loaded by the policy server as an external event
    signal (event_memory.ExternalDistance, factory "memory_management.fruit_distance:RecordedDistance"): returns
    the `fruit_distance` the evaluation client recorded with each frame (fruit_eval.FruitRawRecorder),
    after checking that the row belongs to the very image the server received."""

    def reset(self, context):
        from pathlib import Path
        from .validation.v_a_logger import RawFrameTail
        self.tail = RawFrameTail(Path(context["directory"]) / "raw_frames.jsonl")

    def distance(self, frame):
        import hashlib
        row = self.tail.upto(frame.frame_id)[frame.frame_id]
        if hashlib.sha256(np.asarray(frame.image).tobytes(order="C")).hexdigest() != row["front_sha256"]:
            raise ValueError(f"frame {frame.frame_id}: image differs from the recorded row")
        return row["gt"]["fruit_distance"]
