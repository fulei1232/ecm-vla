"""Pure experiment design, causal memory interventions, and paired gain estimates."""
from __future__ import annotations

import hashlib
import numpy as np

from .partition import partition, phase_of
from .teacher_probe import stable_seed


def array_hash(value) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def variant(label: str, pairs=(), *, kind="swap", selection_half=None) -> dict:
    pairs = sorted((int(c), int(j)) for c, j in pairs)
    if len({c for c, _ in pairs}) != len(pairs) or len({j for _, j in pairs}) != len(pairs):
        raise ValueError("replacement candidates and slots must each be distinct")
    return {"label": label, "pairs": [list(p) for p in pairs], "k": len(pairs),
            "kind": kind, "selection_half": selection_half}


def fixed_variants(candidates: list[dict], history: list[int], *, seed: int,
                   random_sets: dict[int, int]) -> list[dict]:
    out = [variant(name, kind=name) for name in ("base", "d0", "hist_off", "identity")]
    out += [variant(f"k1_c{c}_j{j}", [(c, j)])
            for c in range(len(candidates)) for j in range(len(history))]
    rng = np.random.default_rng(seed)
    for k, count in random_sets.items():
        if count <= 0 or k > min(len(candidates), len(history)):
            continue
        seen = set()
        for _ in range(10000):
            pairs = tuple(sorted(zip(rng.choice(len(candidates), k, replace=False).tolist(),
                                     rng.choice(len(history), k, replace=False).tolist())))
            if pairs in seen:
                continue
            seen.add(pairs)
            out.append(variant(f"k{k}_random{len(seen)-1:02d}", pairs))
            if len(seen) == count:
                break
        if len(seen) != count:
            raise ValueError("could not sample requested number of unique multi-slot interventions")
    return out


def memory_frames(cutoff: int, future: int, candidates: list[dict], spec: dict) -> list[int]:
    """Freeze A/H at cutoff and refresh only R; always sort by original frame ID."""
    if future < cutoff:
        raise ValueError("future observation precedes decision")
    start, now = partition(cutoff), partition(future)
    if spec["kind"] == "d0":
        return list(now.uniform)
    history = list(start.history)
    if spec["kind"] == "hist_off":
        history = []
    for c, slot in spec["pairs"]:
        frame = int(candidates[c]["frame_id"])
        if not 1 <= frame < start.recent[0] or frame in start.uniform:
            raise ValueError("candidate violates history-only causal intervention")
        if not 0 <= slot < len(history):
            raise ValueError("replacement slot outside history")
        history[slot] = frame
    if any(h > cutoff for h in history) or any(r > future for r in now.recent):
        raise ValueError("future frame leakage")
    frames = sorted([*start.anchor, *history, *now.recent])
    if len(frames) != len(set(frames)) or max(frames) > future:
        raise ValueError("duplicate or noncausal memory frames")
    return frames


def discounted_gain(losses: np.ndarray, base: np.ndarray, offsets, gamma: float) -> np.ndarray:
    """Input [variants, valid_offsets, seeds], reference [valid_offsets, seeds]."""
    losses, base = np.asarray(losses), np.asarray(base)
    weights = np.power(float(gamma), np.asarray(offsets, dtype=float))
    if losses.shape[-2:] != base.shape or base.shape[0] != len(weights):
        raise ValueError("gain shape mismatch")
    return np.sum((base - losses) * weights[:, None], axis=-2) / weights.sum()


def greedy_variants(specs: list[dict], gains: np.ndarray, *, sizes=(2, 4), count=2) -> list[dict]:
    """Select on even seeds only; odd seeds remain held out for these adaptive sets."""
    singles = [(i, s) for i, s in enumerate(specs) if s["k"] == 1 and s["kind"] == "swap"]
    rank = sorted(singles, key=lambda item: (-float(gains[item[0], ::2].mean()), item[1]["label"]))
    result = []
    for k in sizes:
        seen = set()
        for first in range(len(rank)):
            chosen, used_c, used_j = [], set(), set()
            for _, s in rank[first:]:
                c, j = s["pairs"][0]
                if c not in used_c and j not in used_j:
                    chosen.append((c, j))
                    used_c.add(c)
                    used_j.add(j)
                    if len(chosen) == k:
                        break
            key = tuple(sorted(chosen))
            if len(chosen) != k or key in seen:
                continue
            seen.add(key)
            result.append(variant(f"k{k}_greedy{len(seen)-1}", chosen,
                                  kind="greedy", selection_half="even"))
            if len(seen) >= count:
                break
    return result


def variant_features(cutoff, exec_start, candidates, spec, frame_features) -> dict:
    """Features use only the decision's historical images/metadata; never future actions."""
    history = list(partition(cutoff).history)
    descriptors = []
    all_frames = list(partition(cutoff).uniform)
    def cosine(a, b):
        return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))
    for ci, ji in spec["pairs"]:
        c, old = candidates[ci], history[ji]
        f = int(c["frame_id"])
        a, b = frame_features[f], frame_features[old]
        after = sorted(set(all_frames) - {old} | {f})
        descriptors.append({
            "gap_change": max(np.diff(after)) - max(np.diff(all_frames)),
            "frame_distance": abs(f - old), "slot_age": cutoff - old,
            "candidate_age": cutoff - f, "visual_cosine": cosine(a, b),
            "history_max_cosine": max(cosine(a, frame_features[h]) for h in history if h != old),
            "saliency": float(c["saliency"]),
            "planner_event": int(c["kind"] == "planner_keyframe"),
            "cross_phase": int(phase_of(f, exec_start) != phase_of(old, exec_start)),
        })
    return {key: float(np.mean([d[key] for d in descriptors])) for key in descriptors[0]} if descriptors else {}


def design_seed(task, episode, cutoff):
    return stable_seed(task, episode, cutoff, "interventions")
