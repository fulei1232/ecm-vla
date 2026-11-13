"""Event-keyframe memory: a fixed number of frames, up to `events` of them keyframes of approach events
and the rest spread evenly over the episode.

One function serves fine-tuning (the memory of every training sample) and closed-loop serving, so the
policy is trained on exactly the memories it is later given. Distances only choose frame indices; they
never enter the model input.

Signal. d_t is the distance in metres from the end effector to the nearest target object at frame t
(event_memory.gt_distance in simulation, the perception module on the robot), NaN where unknown.

Events. A frame is near when d_t <= radius and far when d_t > radius; an unknown frame keeps the
previous state, so a dropped measurement neither starts nor ends an event. An event is a maximal run
of near frames and contributes two keyframes:
  contact  the first frame of the run whose distance is within `tol` of the run's minimum so far;
  release  the first far frame after the run (absent while the run is still going on).
In BinFill the run covers grasp and carry and the release frame is the drop; in PatternLock it is one
button press.

Selection at frame t, from frames 0..t and distances up to t only (causal):
  * t + 1 <= frames: every frame 0..t;
  * otherwise the keyframes of the most recent events, newest event first and contact before release,
    at most `events` of them; then frames - (number of keyframes) frames at evenly spaced positions of
    the remaining frames 0..t (the first and the last remaining frame included).
With events = 0 this is the official even sampling with `frames` frames.
With per_event = 'release', only ended events contribute their release frame;
`events` caps the number of those frames (KA uses frames=16, events=12).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class KeyframeConfig:
    frames: int = 16
    events: int = 0
    radius: float = 0.03
    tol: float = 0.005
    per_event: str = "both"

    def __post_init__(self):
        if not 0 <= self.events < self.frames:
            raise ValueError("need 0 <= events < frames")
        if not (self.radius > 0 and math.isfinite(self.radius) and self.tol >= 0):
            raise ValueError("radius must be finite and positive, tol nonnegative")
        if self.per_event not in ("both", "release"):
            raise ValueError("per_event must be both or release")

    def to_dict(self):
        return asdict(self)


def detect_events(distances, radius, tol):
    """[{start, end, contact, release}] in time order; end is the last near frame, release None while ongoing."""
    d = np.asarray(distances, dtype=np.float64)
    known = np.isfinite(d)
    near = known & (np.where(known, d, np.inf) <= radius)
    # Forward-fill the state over unknown frames: index of the last known frame at or before t.
    last = np.maximum.accumulate(np.where(known, np.arange(len(d)), -1))
    state = np.where(last >= 0, near[np.maximum(last, 0)], False)
    edges = np.diff(np.concatenate([[False], state, [False]]).astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    events = []
    for start, stop in zip(starts.tolist(), stops.tolist()):
        run = d[start:stop]
        values = np.where(np.isfinite(run), run, np.inf)
        contact = start + int(np.flatnonzero(values <= values.min() + tol)[0])
        end = start + int(np.flatnonzero(np.isfinite(run))[-1])
        events.append({"start": start, "end": end, "contact": contact,
                       "release": stop if stop < len(d) else None})
    return events


def keyframes(events, cap, per_event="both"):
    """Keyframes of the most recent events, newest first and contact before release, at most cap."""
    picks = []
    for event in reversed(events):
        candidates = (event["release"],) if per_event == "release" else (event["contact"], event["release"])
        for frame in candidates:
            if frame is not None and frame not in picks and len(picks) < cap:
                picks.append(frame)
    return picks


def spread(candidates, n):
    """n of the sorted candidates at evenly spaced positions, first and last included (as np.linspace)."""
    if n <= 0:
        return []
    positions = np.linspace(0, len(candidates) - 1, n, dtype=np.int64)
    return [int(candidates[i]) for i in positions]


def select(distances, step_idx, cfg: KeyframeConfig):
    """(sorted frame ids, info) of the memory at frame step_idx; distances[t] for t <= step_idx are used."""
    t = int(step_idx)
    if t + 1 <= cfg.frames:
        return list(range(t + 1)), {"events": 0, "keyframes": []}
    d = np.asarray(distances, dtype=np.float64)
    if len(d) < t + 1:
        raise ValueError(f"no distance for frames {len(d)}..{t}")
    d = d[:t + 1]
    events = detect_events(d, cfg.radius, cfg.tol) if cfg.events else []
    keys = keyframes(events, cfg.events, cfg.per_event)
    taken = set(keys)
    rest = [f for f in range(t + 1) if f not in taken]
    frames = sorted(taken | set(spread(rest, cfg.frames - len(keys))))
    if len(frames) != cfg.frames:
        raise RuntimeError("event-keyframe memory does not have the configured frame count")
    return frames, {"events": len(events), "keyframes": sorted(keys)}
