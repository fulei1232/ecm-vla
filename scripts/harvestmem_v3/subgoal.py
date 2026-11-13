"""Shared HarvestMem v3 prompt and grounded-subgoal formatting."""
from __future__ import annotations

from typing import Iterable

import numpy as np


OBSERVE = "observe the fruit tree"
PLACE = "place the fruit in the basket"
RETURN = "return to rest and hold still"


def compose_prompt(instruction: str, grounded_subgoal: str) -> str:
    """Compose the exact prompt used by both SG training and serving."""
    instruction = " ".join(str(instruction).strip().lower().split())
    grounded_subgoal = " ".join(str(grounded_subgoal).strip().lower().split()).rstrip(".")
    return f"{instruction} current subgoal: {grounded_subgoal}."


def project_world_to_pixel(intrinsic: np.ndarray, extrinsic_cv: np.ndarray,
                          point_world: Iterable[float]) -> tuple[float, float]:
    """Project a world-space point using the simulator's 3x4 CV extrinsic."""
    point = np.asarray([*point_world, 1.0], dtype=np.float64)
    camera = np.asarray(extrinsic_cv, dtype=np.float64) @ point
    if camera[2] <= 0:
        raise ValueError(f"point is behind camera: {camera.tolist()}")
    projected = np.asarray(intrinsic, dtype=np.float64) @ camera
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def quantize_unique(points: dict[int, tuple[float, float]], size: int = 256) -> dict[int, tuple[int, int]]:
    """Round projected points and resolve rare integer-pixel collisions deterministically."""
    result: dict[int, tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()
    for slot, (xf, yf) in sorted(points.items()):
        x, y = int(np.rint(xf)), int(np.rint(yf))
        x, y = max(0, min(size - 1, x)), max(0, min(size - 1, y))
        if (x, y) in used:
            candidates = []
            for radius in range(1, size):
                for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius),
                               (radius, radius), (-radius, radius), (radius, -radius),
                               (-radius, -radius)):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < size and 0 <= yy < size and (xx, yy) not in used:
                        candidates.append((xx, yy))
                if candidates:
                    break
            if not candidates:
                raise ValueError("could not resolve projected pixel collision")
            x, y = min(candidates, key=lambda p: (abs(p[0] - xf) ** 2 + abs(p[1] - yf) ** 2, p))
        result[slot] = (x, y)
        used.add((x, y))
    return result


def pick_subgoal(task: str, xy: tuple[int, int], ripe: bool | None = None) -> str:
    x, y = xy
    if task == "occluded_ripe":
        if ripe is not True:
            raise ValueError("occluded pick subgoal requires an initial ripe flag")
        return f"pick the red fruit at <{x}, {y}>"
    return f"pick the fruit at <{x}, {y}>"


def simple_subgoal(task: str, grounded: str) -> str:
    """Return the non-grounded companion field required by the dataset builder."""
    if grounded.startswith("pick the red fruit"):
        return "pick the red fruit"
    if grounded.startswith("pick the fruit"):
        return "pick the fruit"
    return grounded
