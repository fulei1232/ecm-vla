"""Matched controls for V-A (plan §0A.3).

RNDB: delete exactly the D0 frames D4B deletes; each added pin p is replaced by a random frame r(p)
from the same phase, preferably strictly between p's neighbours in the D4B selection (else the same
time bin), outside GT evidence windows, drawn once per (episode, group) and kept across decisions.
JITB: move every history slot of D0 by at most floor(s/2) frames (s = F/31) inside its phase.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .acm_selection import EvidenceBudgetError, time_stratum
from .partition import Partition, phase_of


def stable_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")


class PersistentRandomReplacements:
    """r(p) per evidence group, drawn on first use and reused for the rest of the episode."""

    def __init__(self, task, episode_id, model_seed, bins=8, evidence_margin=4, fallback_margin=2):
        self.task, self.episode_id, self.model_seed = task, episode_id, model_seed
        self.bins, self.evidence_margin, self.fallback_margin = bins, evidence_margin, fallback_margin
        self.assigned: dict[str, dict] = {}

    def _pool(self, pin, max_id, exec_start_idx, excluded, bin_radius=0):
        key = time_stratum(pin, max_id, exec_start_idx, self.bins)[0]
        return [f for f in range(max_id + 1)
                if f not in excluded and phase_of(f, exec_start_idx) == phase_of(pin, exec_start_idx)
                and abs(time_stratum(f, max_id, exec_start_idx, self.bins)[0] - key) <= bin_radius]

    def _draw(self, group_key, pin, max_id, exec_start_idx, evidence_frames, forbidden, salt=0):
        """Returns (frame, relaxation level).

        Within each pool the frame is drawn uniformly among the legal frames whose distance to the pin
        is at most (closest legal distance + max(2, s/2)), s = FrameSamp stride, so the control stays
        temporally close to the pin and keeps the D4B gap structure as far as the exclusions allow.
        0: same phase, outside evidence windows +-evidence_margin;
        1: same phase, only the pin's own +-fallback_margin excluded (evidence overlap).
        """
        near = lambda frames, m: {f + d for f in frames for d in range(-m, m + 1)}
        same_phase = [f for f in range(max_id + 1) if phase_of(f, exec_start_idx) == phase_of(pin, exec_start_idx)]
        pools = [[f for f in same_phase if f not in set(forbidden) | near(evidence_frames, self.evidence_margin)],
                 [f for f in same_phase if f not in set(forbidden) | near([pin], self.fallback_margin)]]
        width = max(2, (max_id // 31) // 2)
        rng = np.random.default_rng(stable_seed(self.task, self.episode_id, group_key, self.model_seed, salt))
        for level, pool in enumerate(pools):
            if pool:
                closest = min(abs(f - pin) for f in pool)
                return int(rng.choice([f for f in pool if abs(f - pin) <= closest + width])), level
        raise EvidenceBudgetError(f"no random frame for group {group_key} pin {pin}")

    def replacements(self, added_pins, max_id, exec_start_idx, evidence_frames, current_d0, retained, treatment):
        """added_pins: {group_key: pin frame} for pins D4B inserts this decision (pins not in D0)."""
        used = set(retained)
        order = sorted(treatment)
        out, log = {}, []
        for group_key, pin in sorted(added_pins.items(), key=lambda kv: (kv[1], kv[0])):
            k = order.index(pin)
            lo, hi = (order[k - 1] if k else -1, order[k + 1] if k + 1 < len(order) else max_id + 1)
            record = self.assigned.get(group_key)
            forbidden = set(current_d0) | set(added_pins.values()) | used
            if record is None:
                frame, level = self._draw(group_key, pin, max_id, exec_start_idx, evidence_frames, forbidden)
                record = self.assigned[group_key] = {"pin": pin, "frame": frame, "relaxation": level,
                                                     "drawn_at_max_id": max_id}
            frame, redraw, level = record["frame"], False, record["relaxation"]
            if frame in forbidden:
                frame, level = self._draw(group_key, pin, max_id, exec_start_idx, evidence_frames, forbidden,
                                          salt=max_id)
                redraw = True
            used.add(frame)
            out[group_key] = frame
            log.append({"group": group_key, "pin": pin, "random_frame": frame, "persistent": not redraw,
                        "rnd_collision_redraw": redraw, "relaxation": level, "distance_to_pin": abs(frame - pin),
                        "between_d4b_neighbours": lo < frame < hi, "evidence_overlap": level > 0,
                        "same_phase": phase_of(frame, exec_start_idx) == phase_of(pin, exec_start_idx),
                        "same_time_bin_now": time_stratum(frame, max_id, exec_start_idx, self.bins)
                        == time_stratum(pin, max_id, exec_start_idx, self.bins)})
        return out, log


def rndb_select(d4b_selected, uniform, pin_groups, max_id, exec_start_idx, evidence_frames, sampler):
    """pin_groups: {group_key: pin}. Returns (selection, log); deleted D0 set equals D4B's."""
    d0, treatment = set(uniform), set(d4b_selected)
    added = {k: p for k, p in pin_groups.items() if p in treatment and p not in d0}
    if set(added.values()) != treatment - d0:
        raise ValueError("D4B added frames must all be pins")
    retained = treatment & d0
    mapping, log = sampler.replacements(added, max_id, exec_start_idx, evidence_frames, d0, retained, treatment)
    selected = sorted(retained | set(mapping.values()))
    if len(selected) != len(d4b_selected):
        raise ValueError("RNDB must keep the D4B slot count")
    return selected, {"deleted_d0_frames": sorted(d0 - treatment), "replacements": log,
                      "rnd_deleted_set_equals_d4b": sorted(d0 - set(selected)) == sorted(d0 - treatment)}


def jitb_select(part: Partition, exec_start_idx, task, episode_id, decision_idx, model_seed):
    """Jitter history slots by at most floor(s/2), s = F/31; anchor and recent window stay fixed."""
    uniform, max_id = list(part.uniform), part.max_id
    half = math.floor((max_id / (len(uniform) - 1)) / 2) if len(uniform) > 1 else 0
    if not part.full or half == 0 or not part.history:
        return list(uniform), {"half_window": half, "moved": 0}
    fixed = set(part.anchor) | set(part.recent)
    history = list(part.history)
    chosen, last = [], part.anchor[0]
    for k, u in enumerate(history):
        rng = np.random.default_rng(stable_seed(task, episode_id, decision_idx, k, model_seed))
        target = u + int(rng.integers(-half, half + 1))
        phase = phase_of(u, exec_start_idx)
        remaining_same_phase = sum(phase_of(v, exec_start_idx) == phase for v in history[k + 1:])
        upper = part.recent[0] - 1 - (len(history) - k - 1)
        if phase == "demo":
            upper = min(upper, exec_start_idx - 1 - remaining_same_phase)
        lower = last + 1
        if phase == "exec":
            lower = max(lower, exec_start_idx)
        legal = [f for f in range(lower, upper + 1) if f not in fixed and phase_of(f, exec_start_idx) == phase]
        if not legal:
            raise EvidenceBudgetError(f"no legal jitter frame for history slot {k}")
        frame = min(legal, key=lambda f: (abs(f - target), f))
        chosen.append(frame)
        last = frame
    selected = sorted(set(part.anchor) | set(chosen) | set(part.recent))
    assert len(selected) == len(uniform)
    return selected, {"half_window": half, "moved": sum(a != b for a, b in zip(chosen, history)),
                      "max_shift": max(abs(a - b) for a, b in zip(chosen, history))}
