"""Seeded scene layouts for the fruit-picking tasks (pure numpy; the simulator only renders what this returns).

World frame: arm base (RM75-B or RM65-B, same link lengths) at the origin on a 0.40 m pedestal (ground at z = -0.40), +x towards the tree, z up.
Fruits sit at 0.44-0.64 m, where every grasp and pre-grasp pose with a horizontal approach has an IK solution
(lower targets need elbow configurations that hit the base). Fruits hang under the tips of twigs that grow
from a trunk behind them, so the gripper approaches every fruit horizontally along its azimuth without passing
under a branch.

Tasks
- `occluded_ripe`: red (ripe) and green (unripe) fruits. During the opening observation phase a gust swings leaf
  clusters in front of two fruits (always at least one ripe, so "hidden" is not a cue for "ripe"). Goal: put every
  red fruit in the basket. The policy must remember where the hidden red fruits were.
- `count`: all fruits are red. Goal: put exactly N of the M fruits in the basket, where N < M and M varies per
  episode. The basket sits outside the front camera's view, so the policy must count its own picks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json

import numpy as np

TASKS = ("occluded_ripe", "count")

GROUND_Z = -0.40
FRONT_CAMERA_EYE = np.array([-0.15, 0.0, 1.05])
FRONT_CAMERA_TARGET = np.array([0.42, 0.0, 0.52])
FRONT_CAMERA_FOV = np.deg2rad(55)

TRUNK_RHO = 0.68
TRUNK_RADIUS = 0.035
TRUNK_TOP = 1.0
TWIG_RADIUS = 0.010
FRUIT_RHO = (0.38, 0.50)
FRUIT_AZIMUTH = np.deg2rad(28)
FRUIT_Z = (0.44, 0.64)
FRUIT_RADIUS = (0.025, 0.029)
STEM = (0.02, 0.04)
FRUIT_SPACING = 0.10
CORRIDOR = 0.26
CORRIDOR_CLEARANCE = 0.055
SIGHTLINE_CLEARANCE = 0.03
# below the bottom edge of the front camera image; the home / drop tcp pose is straight above it
BASKET_CENTER = np.array([-0.06, -0.34, 0.15])
BASKET_INNER_HALF = 0.09
BASKET_HEIGHT = 0.12
DROP_Z = 0.47
OBSERVE_STEPS = 30
GUST = (8, 20)

RIPE_RGB = ((0.80, 0.08, 0.06), (0.88, 0.16, 0.06))
UNRIPE_RGB = ((0.38, 0.62, 0.14), (0.52, 0.70, 0.18))
LEAF_RGB = ((0.10, 0.36, 0.08), (0.20, 0.50, 0.14))


@dataclass
class Layout:
    task: str
    seed: int
    trunk: dict
    twigs: list = field(default_factory=list)
    fruits: list = field(default_factory=list)
    leaves: list = field(default_factory=list)
    occluders: list = field(default_factory=list)
    basket: dict = field(default_factory=dict)
    targets: list = field(default_factory=list)
    n_pick: int = 0
    instruction: str = ""
    observe_steps: int = OBSERVE_STEPS
    gust: tuple = GUST

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)


def approach_dir(pos):
    a = np.array([pos[0], pos[1], 0.0])
    return a / np.linalg.norm(a)


def segment_distance(p, a, b):
    ab = b - a
    t = np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def segment_segment_distance(a0, a1, b0, b1, n=24):
    ts = np.linspace(0, 1, n)
    return min(segment_distance(a0 + t * (a1 - a0), b0, b1) for t in ts)


def corridor(fruit):
    pos = np.asarray(fruit["pos"])
    return pos, pos - CORRIDOR * approach_dir(pos)


def quat_from_matrix(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def random_rotation(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def facing_rotation(normal, rng):
    """Leaf plane (local xy) facing `normal`, with a random in-plane spin."""
    n = normal / np.linalg.norm(normal)
    helper = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(helper, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    spin = rng.uniform(-0.5, 0.5)
    u, v = np.cos(spin) * u + np.sin(spin) * v, -np.sin(spin) * u + np.cos(spin) * v
    return quat_from_matrix(np.stack([u, v, n], axis=1))


def sample_fruits(rng, n):
    fruits = []
    for _ in range(4000):
        if len(fruits) == n:
            return fruits
        rho, az, z = rng.uniform(*FRUIT_RHO), rng.uniform(-FRUIT_AZIMUTH, FRUIT_AZIMUTH), rng.uniform(*FRUIT_Z)
        pos = np.array([rho * np.cos(az), rho * np.sin(az), z])
        cand = {"pos": pos, "radius": float(rng.uniform(*FRUIT_RADIUS)), "stem": float(rng.uniform(*STEM))}
        ok = True
        for other in fruits:
            o = np.asarray(other["pos"])
            if np.linalg.norm(o - pos) < FRUIT_SPACING:
                ok = False
                break
            if segment_distance(o, *corridor(cand)) < other["radius"] + CORRIDOR_CLEARANCE:
                ok = False
                break
            if segment_distance(pos, *corridor(other)) < cand["radius"] + CORRIDOR_CLEARANCE:
                ok = False
                break
            for f, g in ((cand, other), (other, cand)):
                if segment_distance(np.asarray(g["pos"]), FRONT_CAMERA_EYE, np.asarray(f["pos"])) < f["radius"] + g["radius"] + SIGHTLINE_CLEARANCE \
                        and np.linalg.norm(np.asarray(g["pos"]) - FRONT_CAMERA_EYE) < np.linalg.norm(np.asarray(f["pos"]) - FRONT_CAMERA_EYE):
                    ok = False
        if ok:
            fruits.append(cand)
    raise RuntimeError("could not place fruits")


def twig_for(fruit, trunk_xy, rng):
    pos = np.asarray(fruit["pos"])
    top = pos + np.array([0, 0, fruit["radius"] + fruit["stem"]])
    tip = top + 0.015 * approach_dir(pos)
    base = np.array([trunk_xy[0], trunk_xy[1], top[2] - rng.uniform(0.04, 0.10)])
    return {"start": base, "end": tip, "radius": TWIG_RADIUS, "stem_top": top}


def twig_clear(twig, fruits, owner=None):
    for j, f in enumerate(fruits):
        if j == owner:
            continue
        a, b = corridor(f)
        if segment_segment_distance(twig["start"], twig["end"], a, b) < twig["radius"] + CORRIDOR_CLEARANCE:
            return False
        if segment_distance(np.asarray(f["pos"]), twig["start"], twig["end"]) < f["radius"] + twig["radius"] + 0.02:
            return False
        if segment_segment_distance(twig["start"], twig["end"], FRONT_CAMERA_EYE, np.asarray(f["pos"])) < f["radius"] + SIGHTLINE_CLEARANCE:
            return False
    return True


def leaf_clear(pos, fruits):
    for f in fruits:
        fp = np.asarray(f["pos"])
        if segment_distance(pos, FRONT_CAMERA_EYE, fp) < f["radius"] + 0.05:
            return False
        if segment_distance(pos, *corridor(f)) < CORRIDOR_CLEARANCE + 0.01:
            return False
    return True


def sample_leaves(rng, twigs, fruits, per_twig=(3, 6)):
    leaves = []
    for k, twig in enumerate(twigs):
        start, end = np.asarray(twig["start"]), np.asarray(twig["end"])
        want = int(rng.integers(per_twig[0], per_twig[1] + 1))
        for _ in range(want * 4):
            if sum(l["twig"] == k for l in leaves) >= want:
                break
            pos = start + rng.uniform(0.1, 0.9) * (end - start) + rng.normal(scale=0.025, size=3)
            if leaf_clear(pos, fruits):
                leaves.append({"pos": pos, "quat": random_rotation(rng), "scale": float(rng.uniform(0.8, 1.2)),
                               "rgb": rng.uniform(*LEAF_RGB), "twig": k})
    return leaves


def occluder_for(fruit, rng):
    """A leaf cluster that swings down in front of `fruit` (towards the front camera) during the gust."""
    pos = np.asarray(fruit["pos"])
    to_cam = FRONT_CAMERA_EYE - pos
    to_cam /= np.linalg.norm(to_cam)
    closed = pos + (fruit["radius"] + 0.035) * to_cam
    opened = closed + np.array([0, 0, 0.16]) + 0.04 * approach_dir(pos)
    leaves = []
    for du, dv in ((-0.022, -0.016), (0.022, -0.016), (-0.022, 0.018), (0.022, 0.018), (0.0, 0.0)):
        leaves.append({"offset": np.array([du, dv, rng.normal(scale=0.004)]),
                       "quat": facing_rotation(to_cam + rng.normal(scale=0.15, size=3), rng),
                       "scale": float(rng.uniform(1.0, 1.2)), "rgb": rng.uniform(*LEAF_RGB)})
    return {"normal": to_cam, "closed": closed, "open": opened, "leaves": leaves}


def sample_layout(task: str, seed: int) -> Layout:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {TASKS}")
    rng = np.random.default_rng([seed, TASKS.index(task)])
    for _ in range(200):
        n = int(rng.integers(4, 7))
        fruits = sample_fruits(rng, n)
        az = rng.uniform(-np.deg2rad(5), np.deg2rad(5))
        trunk_xy = TRUNK_RHO * np.array([np.cos(az), np.sin(az)])
        twigs = [twig_for(f, trunk_xy, rng) for f in fruits]
        if all(twig_clear(t, fruits, owner=i) for i, t in enumerate(twigs)):
            break
    else:
        raise RuntimeError(f"no valid tree for seed {seed}")
    order = sorted(range(n), key=lambda i: -fruits[i]["pos"][1])
    for f in fruits:
        f["pos"] = f["pos"].tolist()
    layout = Layout(task=task, seed=seed,
                    trunk={"xy": trunk_xy, "radius": TRUNK_RADIUS, "bottom": GROUND_Z, "top": TRUNK_TOP},
                    basket={"center": BASKET_CENTER, "inner_half": BASKET_INNER_HALF, "height": BASKET_HEIGHT,
                            "drop": BASKET_CENTER + np.array([0, 0, DROP_Z - BASKET_CENTER[2]])})
    if task == "occluded_ripe":
        n_ripe = int(rng.integers(2, min(3, n - 1) + 1))
        ripe = set(rng.choice(n, size=n_ripe, replace=False).tolist())
        hidden = [int(rng.choice(sorted(ripe)))]
        hidden.append(int(rng.choice([i for i in range(n) if i != hidden[0]])))
        for i, f in enumerate(fruits):
            f["ripe"] = i in ripe
            f["rgb"] = rng.uniform(*(RIPE_RGB if f["ripe"] else UNRIPE_RGB))
        layout.occluders = [{"fruit": i, **occluder_for(fruits[i], rng)} for i in hidden]
        layout.targets = [i for i in order if i in ripe]
        layout.n_pick = len(layout.targets)
        layout.instruction = "pick all the red fruits from the tree and put them into the basket"
    else:
        for f in fruits:
            f["ripe"] = True
            f["rgb"] = rng.uniform(*RIPE_RGB)
        layout.n_pick = int(rng.integers(2, n))
        layout.targets = order[:layout.n_pick]
        word = {2: "two", 3: "three", 4: "four", 5: "five"}[layout.n_pick]
        layout.instruction = f"pick {word} fruits from the tree and put them into the basket"
    layout.fruits = fruits
    layout.twigs = twigs
    layout.leaves = sample_leaves(rng, twigs, fruits)
    return layout
