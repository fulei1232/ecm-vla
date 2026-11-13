"""Deterministic, causal data selection helpers for the V-B teacher probe."""
from __future__ import annotations

import hashlib

import numpy as np


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def decision_boundaries(
    n_frames: int,
    exec_start_idx: int,
    *,
    stride: int = 16,
    min_cutoff_frame: int = 64,
    count: int = 3,
) -> list[int]:
    """Choose distinct early/middle/late execution cutoffs, never beyond data."""
    if n_frames <= 0 or not 0 <= exec_start_idx < n_frames:
        raise ValueError("invalid episode length or execution start")
    if stride <= 0 or count <= 0:
        raise ValueError("stride and count must be positive")
    eligible = np.arange(exec_start_idx, n_frames, stride, dtype=int)
    eligible = eligible[eligible >= min_cutoff_frame]
    if eligible.size < count:
        return []
    quantiles = (np.arange(count, dtype=float) + 0.5) / count
    targets = exec_start_idx + quantiles * (n_frames - 1 - exec_start_idx)
    selected = []
    for target in targets:
        order = np.argsort(np.abs(eligible - target), kind="stable")
        selected.append(next(int(eligible[i]) for i in order if int(eligible[i]) not in selected))
    return sorted(selected)


def kemo_slowdown_saliency(joint_state: np.ndarray, window: int = 16) -> np.ndarray:
    """Causal inverse mean joint speed; the first frame has no motion estimate."""
    q = np.asarray(joint_state, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"expected [frames, 7] joint positions, got {q.shape}")
    if window < 1 or not np.isfinite(q).all():
        raise ValueError("window must be positive and joint positions finite")
    speed = np.zeros(len(q), dtype=np.float64)
    if len(q) > 1:
        step_speed = np.linalg.norm(np.diff(q, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(step_speed)))
        for frame in range(1, len(q)):
            first_delta = max(0, frame - window + 1)
            n = frame - first_delta
            speed[frame] = (cumulative[frame] - cumulative[first_delta]) / n
    return 1.0 / (1.0 + speed)


def causal_peak_frames(
    saliency: np.ndarray,
    cutoff_frame: int,
    *,
    left: int = 8,
    right: int = 4,
    min_gap: int = 8,
) -> list[tuple[int, float]]:
    """Return confirmed local maxima whose confirmation time is at most cutoff."""
    values = np.asarray(saliency, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("saliency must be a finite vector")
    if min(left, right, min_gap) < 0:
        raise ValueError("peak windows and minimum gap cannot be negative")
    stop = min(int(cutoff_frame), len(values) - 1)
    peaks = []
    for frame in range(left, stop - right + 1):
        past = values[frame - left:frame]
        neighborhood = values[frame - left:frame + right + 1]
        # Earliest frame wins a flat maximum, avoiding duplicate plateau events.
        if values[frame] < np.max(neighborhood) or np.any(past == values[frame]):
            continue
        baseline = float(np.median(past)) if past.size else float(values[frame])
        prominence = float(values[frame] - baseline)
        if prominence <= 0:
            continue
        if peaks and frame - peaks[-1][0] < min_gap:
            if prominence > peaks[-1][1]:
                peaks[-1] = (frame, prominence)
        else:
            peaks.append((frame, prominence))
    return peaks


def select_v_b_candidates(
    *,
    task: str,
    episode_id: int,
    model_seed: int,
    cutoff_frame: int,
    recent_first_frame: int,
    d0_frames: list[int],
    joint_state: np.ndarray,
    subgoal_boundaries: list[int],
    kemo_count: int = 2,
    random_count: int = 3,
    left: int = 8,
    right: int = 4,
    min_gap: int = 8,
    window: int = 16,
) -> list[dict]:
    """Select causally available history-frame candidates for one boundary."""
    allowed = {f for f in range(1, min(cutoff_frame, recent_first_frame)) if f not in set(d0_frames)}
    saliency = kemo_slowdown_saliency(joint_state, window=window)
    peaks = [(f, score) for f, score in causal_peak_frames(
        saliency, cutoff_frame, left=left, right=right, min_gap=min_gap) if f in allowed]
    peaks.sort(key=lambda pair: (-pair[1], pair[0]))
    chosen: list[dict] = [
        {"frame_id": frame, "kind": "kemo_peak", "saliency": float(saliency[frame]), "prominence": score}
        for frame, score in peaks[:kemo_count]
    ]

    keyframes = sorted({int(f) for f in subgoal_boundaries if int(f) in allowed
                        and int(f) not in {r["frame_id"] for r in chosen}})
    if keyframes:
        frame = min(keyframes, key=lambda f: (abs(f - cutoff_frame), f))
        chosen.append({"frame_id": frame, "kind": "planner_keyframe", "saliency": float(saliency[frame])})
    elif len(peaks) > kemo_count:
        frame, score = peaks[kemo_count]
        chosen.append({"frame_id": frame, "kind": "kemo_peak_fallback", "saliency": float(saliency[frame]),
                       "prominence": score})

    occupied = {r["frame_id"] for r in chosen}
    random_pool = sorted(allowed - occupied)
    rng = np.random.default_rng(stable_seed(task, episode_id, model_seed, cutoff_frame, "random"))
    n_random = min(random_count, len(random_pool))
    if n_random:
        for frame in sorted(int(x) for x in rng.choice(random_pool, size=n_random, replace=False)):
            chosen.append({"frame_id": frame, "kind": "random", "saliency": float(saliency[frame])})
    return sorted(chosen, key=lambda row: (row["frame_id"], row["kind"]))
