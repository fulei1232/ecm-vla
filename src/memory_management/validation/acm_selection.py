"""ACM-VLA selection helpers copied verbatim (sources and SHA256 in acm_copy/SOURCES.json).

support_clip / representative / fill come from acm_vla/scripts/diagnostic_selection.py;
time_stratum comes from acm_vla/scripts/context_selection.py. fill() is kept only to reproduce
the confounded D4L condition.
"""
from __future__ import annotations

import numpy as np

from .acm_copy.diagnostic_annotations import consecutive_windows, required_support, visible_pixels


class EvidenceBudgetError(ValueError):
    pass


def _group_index(annotation, group):
    return next(i for i, candidate in enumerate(annotation["required_groups"])
                if candidate is group or candidate == group)


def support_clip(annotation, group, rows):
    """Choose the strongest required consecutive support, never isolated frames."""
    group_index = _group_index(annotation, group)
    windows = consecutive_windows(group, required_support(annotation, group_index))
    if not windows:
        return []
    by_id = {row["frame_id"]: row for row in rows}
    objects = annotation.get("group_objects", [annotation.get("objects", [])])[group_index]
    middle = (group[0] + group[-1]) / 2

    def score(window):
        per_frame = [sum(visible_pixels(by_id[idx], actor) for actor in objects if actor)
                     for idx in window]
        return (min(per_frame, default=0), sum(per_frame),
                -abs(np.mean(window) - middle), -window[0])

    return max(windows, key=score)


def representative(annotation, group, rows):
    clip = support_clip(annotation, group, rows)
    if not clip:
        raise ValueError(f"No sufficient support for {annotation['annotation_id']}")
    return clip[len(clip) // 2]


def fill(pinned, uniform, max_id, capacity=32):
    # Priority order is deterministic; preserve current observation and source positions.
    if any(i < 0 or i > max_id for i in pinned+uniform):
        raise ValueError("Future or negative selection candidate")
    kept = [max_id]
    for i in pinned + uniform + list(range(max_id, -1, -1)):
        if i not in kept:
            kept.append(i)
        if len(kept) >= min(capacity, max_id+1):
            break
    return sorted(kept)


def time_stratum(frame, max_id, exec_start_idx, bins=8):
    is_demo = frame < exec_start_idx
    start, end = (0, exec_start_idx - 1) if is_demo else (exec_start_idx, max_id)
    return min(bins - 1, (frame - start) * bins // (end - start + 1)), is_demo
