"""Event-aware memory for the counterfactual ablations: more frames where the end effector is close to
its target, mixed with uniform coverage.

Signal. An EventSignal gives every frame t a distance d_t (metres) between the end effector and the
target grasp pose, NaN where unknown. GroundTruthDistance is a privileged stand-in that reads the
simulator rows the client writes to raw_frames.jsonl; ExternalDistance wraps a perception module
(`distance(frame: FrameInput) -> float | None`). Signals only choose frame indices; they never enter the
model input.

Discriminator. Proximity p_t = exp(-max(d_t - d0, 0) / tau), and 0 where d_t is unknown; tau and d0 may be
given per task.

Selection. From the partition of the official selection U, the anchor and the recent window are kept
and the H history slots are refilled from frames 1 .. recent[0] - 1 by systematic sampling of
    rho_t = (1 - alpha) / N + alpha * p_t / sum(p),
water-filled so that no frame carries more than 1/H of the mass, hence every frame is picked at most
once. alpha = 0, a partition without dropped frames, or zero proximity everywhere returns U unchanged.

Memories compared at the same budget (all keep the anchor, the recent window and the frame count):
  EV    the selection above at the configured alpha;
  EVU   the same sampler on a uniform density: what EV changes without any event information;
  EVR   U with exactly the history frames EV dropped, replaced by as many random frames outside U;
  EVA   the selection on 1 - p (far from the target) where the distance is known, same alpha;
  EV@a  the selection at each alpha of the grid (counterfactual candidates of ES/TE).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib
import math
from pathlib import Path
import sys

import numpy as np

from .partition import Partition
from .v_a_logger import RawFrameTail

STRATEGIES = ("EV", "EVU", "EVR", "EVA")
# Simulator keys of the objects a frame's distance is measured to (GroundTruthDistance only).
GT_TARGETS = {"PatternLock": "selected_buttons", "BinFill": "all_cubes", "VideoUnmaskSwap": "spawned_bins"}
# The PatternLock stick hangs below the TCP, so its distance to a button is measured in the table plane.
GT_PLANAR = frozenset({"PatternLock"})
OFF_TABLE_M = 2.0


@dataclass(frozen=True)
class EventMemoryConfig:
    """tau and d0 are metres, either one value or {task: value}; for_task() resolves them."""
    alpha: float = 0.5
    tau: object = 0.05
    d0: object = 0.0
    alphas: tuple = (0.25, 0.5, 0.75, 1.0)
    pool: str = "random"
    signal: dict = field(default_factory=lambda: {"kind": "ground_truth"})

    def __post_init__(self):
        if not all(0 <= a <= 1 for a in (self.alpha, *self.alphas)):
            raise ValueError("alpha must lie in [0, 1]")
        taus = self.tau.values() if isinstance(self.tau, dict) else [self.tau]
        d0s = self.d0.values() if isinstance(self.d0, dict) else [self.d0]
        if not all(t > 0 and math.isfinite(t) for t in taus) or not all(d >= 0 for d in d0s):
            raise ValueError("tau must be finite and positive, d0 nonnegative")
        if self.pool not in ("random", "event"):
            raise ValueError(f"unknown pool {self.pool!r}")

    @classmethod
    def from_params(cls, params):
        params = dict(params)
        if "alphas" in params:
            params["alphas"] = tuple(params["alphas"])
        return cls(**params)

    def for_task(self, task):
        pick = lambda v: float(v[task] if isinstance(v, dict) else v)  # noqa: E731
        return replace(self, tau=pick(self.tau), d0=pick(self.d0))

    def candidates(self):
        """Names of the memories ES/TE score against U, in a fixed order."""
        return [*STRATEGIES, *(f"EV@{a:g}" for a in self.alphas)]


# --------------------------------------------------------------------------- signals

@dataclass(frozen=True)
class FrameInput:
    task: str
    frame_id: int
    image: np.ndarray
    state: np.ndarray
    is_demo: bool


class EventSignal:
    """Per-frame distances; frame t's value may only use frames <= t."""
    needs_frames = False

    def reset(self, context):
        self.context = context

    def observe(self, frames):
        pass

    def distances(self, max_id):
        raise NotImplementedError


def gt_distance(task, gt):
    """Distance from the TCP to the nearest target object on the table; NaN if there is none."""
    tcp = gt.get("tcp_position")
    if not tcp:
        return math.nan
    objects = gt.get(GT_TARGETS[task]) or []
    if task == "BinFill":
        colors = {color for color, _ in gt.get("binfill_language_sequence") or []}
        objects = [o for o in objects if any(f"_{c}_" in o["name"] for c in colors)]
    tcp = np.asarray(tcp, dtype=np.float64).reshape(3)
    positions = [np.asarray(o["position"], dtype=np.float64).reshape(3) for o in objects]
    positions = [p for p in positions if np.abs(p[:2]).max() < OFF_TABLE_M]
    dims = 2 if task in GT_PLANAR else 3
    return float(min(np.linalg.norm(p[:dims] - tcp[:dims]) for p in positions)) if positions else math.nan


class GroundTruthDistance(EventSignal):
    """Privileged stand-in for the perception module, from the simulator rows in raw_frames.jsonl."""

    def reset(self, context):
        super().reset(context)
        self.tail, self.values = RawFrameTail(Path(context["directory"]) / "raw_frames.jsonl"), []

    def distances(self, max_id):
        rows = self.tail.upto(max_id)
        self.values += [gt_distance(self.context["task"], r.get("gt") or {}) for r in rows[len(self.values):]]
        return np.asarray(self.values[:max_id + 1], dtype=np.float64)


class ExternalDistance(EventSignal):
    """A perception module built by `factory` ("module:callable"); it needs `distance(FrameInput)` and may
    define `reset(context)`. Frames arrive once each, in order, as the policy ingests them."""
    needs_frames = True

    def __init__(self, factory, kwargs=None, path=None):
        if path and path not in sys.path:
            sys.path.insert(0, path)
        module, attr = factory.split(":")
        self.module = getattr(importlib.import_module(module), attr)(**(kwargs or {}))

    def reset(self, context):
        super().reset(context)
        self.values = []
        if hasattr(self.module, "reset"):
            self.module.reset(context)

    def observe(self, frames):
        for frame in frames:
            if frame.frame_id != len(self.values):
                raise ValueError(f"frame {frame.frame_id} arrived out of order")
            value = self.module.distance(frame)
            self.values.append(math.nan if value is None else float(value))

    def distances(self, max_id):
        if len(self.values) <= max_id:
            raise ValueError(f"no distance for frames {len(self.values)}..{max_id}")
        return np.asarray(self.values[:max_id + 1], dtype=np.float64)


def make_signal(spec):
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind == "ground_truth":
        return GroundTruthDistance()
    if kind == "external":
        return ExternalDistance(**spec)
    raise ValueError(f"unknown event signal {kind!r}")


# --------------------------------------------------------------------------- selection

def proximity(distances, tau, d0=0.0):
    d = np.asarray(distances, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        p = np.exp(-np.maximum(d - d0, 0.0) / tau)
    return np.where(np.isfinite(d), p, 0.0)


def water_fill(weights, cap):
    """Weights scaled to sum 1 with no entry above cap; the excess goes to the others in proportion."""
    w = np.asarray(weights, dtype=np.float64)
    if w.min() < 0 or w.sum() <= 0 or cap * len(w) < 1 - 1e-12:
        raise ValueError("weights must be nonnegative with positive sum, and cap * len >= 1")
    fixed = np.zeros(len(w), dtype=bool)
    out = w / w.sum()
    while True:
        over = (out > cap) & ~fixed
        if not over.any():
            return out
        fixed |= over
        free = ~fixed
        mass = max(1.0 - cap * fixed.sum(), 0.0)
        out = np.where(fixed, cap, 0.0)
        if free.any():
            base = w[free].sum()
            out[free] = w[free] / base * mass if base > 0 else mass / free.sum()


def systematic(weights, k):
    """k distinct indices: the cells holding the points (i + 1/2) / k of the cumulative weights (each <= 1/k)."""
    # In units of 1/k, rounded: a capped cell spans exactly one unit and can hold only one point even when
    # its edges fall on two points.
    edges = np.round(np.cumsum(weights) * k, 9)
    edges[-1] = k
    picks = np.searchsorted(edges, np.arange(k) + 0.5, side="right")
    if len(set(picks.tolist())) != k:
        raise RuntimeError("systematic sampling picked a frame twice")
    return picks


def history_range(part: Partition):
    return np.arange(1, part.recent[0]) if part.history else np.arange(0)


def event_selection(part: Partition, weights, alpha):
    """U with its history slots refilled by the alpha-mixture of uniform and `weights` (one per frame)."""
    uniform = [int(f) for f in part.uniform]
    frames = history_range(part)
    if alpha <= 0 or not part.full or not len(frames):
        return uniform
    w = np.asarray(weights, dtype=np.float64)[frames]
    if w.sum() <= 0:
        return uniform
    h = len(part.history)
    rho = (1 - alpha) / len(frames) + alpha * w / w.sum()
    picks = frames[systematic(water_fill(rho, 1.0 / h), h)]
    return sorted({*map(int, part.anchor), *map(int, part.recent), *map(int, picks)})


def matched_random(part: Partition, selected, seed):
    uniform = {int(f) for f in part.uniform}
    dropped = uniform - set(selected)
    n = len(set(selected) - uniform)
    if not n:
        return sorted(uniform)
    allowed = [int(f) for f in history_range(part) if int(f) not in uniform]
    extra = np.random.default_rng(seed).choice(allowed, size=n, replace=False)
    return sorted((uniform - dropped) | {int(f) for f in extra})


def strategy_memories(part: Partition, distances, cfg: EventMemoryConfig, seed):
    """{name: frames} for every memory of cfg.candidates() (cfg resolved for the task); seed drives EVR only."""
    d = np.asarray(distances, dtype=np.float64)
    p = proximity(d, cfg.tau, cfg.d0)
    ev = event_selection(part, p, cfg.alpha)
    memories = {"EV": ev, "EVU": event_selection(part, np.ones(len(d)), 1.0), "EVR": matched_random(part, ev, seed),
                "EVA": event_selection(part, np.where(np.isfinite(d), 1.0 - p, 0.0), cfg.alpha)}
    memories.update({f"EV@{a:g}": event_selection(part, p, a) for a in cfg.alphas})
    return memories


def event_pool(part: Partition, distances, cfg: EventMemoryConfig, size):
    """Single-swap candidate pool drawn by the event density among frames outside U; None without signal."""
    uniform = {int(f) for f in part.uniform}
    allowed = np.asarray([int(f) for f in history_range(part) if int(f) not in uniform], dtype=np.int64)
    if not len(allowed):
        return None
    p = proximity(distances, cfg.tau, cfg.d0)[allowed]
    if p.sum() <= 0:
        return None
    k = min(size, len(allowed))
    rho = (1 - cfg.alpha) / len(allowed) + cfg.alpha * p / p.sum()
    return sorted(int(f) for f in allowed[systematic(water_fill(rho, 1.0 / k), k)])


def memory_stats(memory, uniform, distances, tau):
    d = np.asarray(distances, dtype=np.float64)

    def near(frames):
        v = d[list(frames)]
        return int((np.isfinite(v) & (v <= tau)).sum())

    return {"changed": len(set(memory) - set(uniform)), "near": near(memory), "near_uniform": near(uniform)}


def decode_distances(rows, max_id):
    """Distances for frames 0..max_id from event_signal.jsonl rows (None = unknown)."""
    values = []
    for r in rows:
        if r["first_frame"] != len(values):
            raise ValueError("event_signal.jsonl rows are not contiguous")
        values += [math.nan if v is None else float(v) for v in r["distance"]]
    if len(values) <= max_id:
        raise ValueError(f"event_signal.jsonl does not cover frame {max_id}")
    return np.asarray(values[:max_id + 1], dtype=np.float64)
