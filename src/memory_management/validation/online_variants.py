"""V-A condition router (plan §0A.3), called once per decision with the official D0 indices.

RNDB deletes exactly the D0 frames D4B deletes; RNDC feeds the same persistent random frames through
phase_balanced_fill, matching D4B coverage at the cost of a different deleted set.

Pins: one Oracle-v2 representative frame per required evidence group (event and retention kinds),
fixed the first time the group has sufficient causal support. Ground-truth rows only choose frame
indices; they never enter the model input.
"""
from __future__ import annotations

import numpy as np

from .acm_copy.diagnostic_annotations import active_annotations, annotate, annotation_sufficient
from .acm_selection import EvidenceBudgetError, fill, representative
from .matched_controls import PersistentRandomReplacements, jitb_select, rndb_select
from .partition import partition, phase_counts
from .phase_balanced_fill import phase_balanced_fill

CONDITIONS = ("D0a", "D0b", "D4L", "D4B", "RNDB", "RNDC", "JITB")
PIN_KINDS = frozenset({"event", "retention"})


class EpisodeState:
    def __init__(self, task, episode_id, model_seed, exec_start_idx, rndb_cfg=None):
        self.task, self.episode_id, self.model_seed = task, episode_id, model_seed
        self.exec_start_idx = exec_start_idx
        self.pin_cache: dict[str, int] = {}
        self.sampler = PersistentRandomReplacements(task, episode_id, model_seed, **(rndb_cfg or {}))
        self.decision = 0


def current_pins(state: EpisodeState, rows, max_id) -> dict[str, int]:
    """{group_key: frame} for active, sufficiently supported evidence groups."""
    pins = {}
    for annotation in active_annotations(state.task, rows, max_id, PIN_KINDS):
        if not annotation_sufficient(annotation):
            continue
        for index, group in enumerate(annotation["required_groups"]):
            key = f"{annotation['annotation_id']}:{index}"
            if key not in state.pin_cache:
                state.pin_cache[key] = int(representative(annotation, group, rows))
            pins[key] = state.pin_cache[key]
    return pins


def evidence_frames(task, rows, max_id) -> set[int]:
    return {f for a in annotate(task, rows, max_id) for g in a["required_groups"] for f in g}


def unique_pin_groups(pins: dict[str, int]) -> dict[str, int]:
    """One group key per distinct pin frame (several groups can share a representative)."""
    by_frame = {}
    for key, frame in sorted(pins.items()):
        by_frame.setdefault(frame, key)
    return {key: frame for frame, key in by_frame.items()}


def selection_stats(selected, uniform, max_id, exec_start_idx, part):
    gaps = np.diff(selected)
    return {
        "n_frames": len(selected),
        "deleted_d0_frames": sorted(set(uniform) - set(selected)),
        "added_frames": sorted(set(selected) - set(uniform)),
        "n_demo_slots_d0": phase_counts(uniform, exec_start_idx)["demo"],
        "n_exec_slots_d0": phase_counts(uniform, exec_start_idx)["exec"],
        "n_demo_slots": phase_counts(selected, exec_start_idx)["demo"],
        "n_exec_slots": phase_counts(selected, exec_start_idx)["exec"],
        "max_gap": int(max(gaps, default=0)),
        "current_to_prev_gap": int(selected[-1] - selected[-2]) if len(selected) > 1 else 0,
        "protected_present": set(part.protected) <= set(selected),
    }


def choose(condition, uniform, max_id, rows, state: EpisodeState):
    uniform = [int(u) for u in uniform]
    part = partition(max_id, capacity=len(uniform) if len(uniform) == 32 else 32)
    log = {"condition": condition, "decision_idx": state.decision, "max_id": max_id,
           "exec_start_idx": state.exec_start_idx, "pin_overflow": False, "phase_borrow": 0}
    state.decision += 1
    if condition in ("D0a", "D0b"):
        selected = uniform
    elif condition == "JITB":
        selected, info = jitb_select(part, state.exec_start_idx, state.task, state.episode_id,
                                     log["decision_idx"], state.model_seed)
        log["jitter"] = info
    elif condition in ("D4L", "D4B", "RNDB", "RNDC"):
        pins = current_pins(state, rows, max_id)
        groups = unique_pin_groups(pins)
        pin_frames = sorted(groups.values())
        log["pins"] = pins
        log["n_pin_frames"] = len(pin_frames)
        log["n_pins_not_in_d0"] = len(set(pin_frames) - set(uniform))
        try:
            d4b = phase_balanced_fill(pin_frames, uniform, max_id, state.exec_start_idx)
        except EvidenceBudgetError as error:
            d4b, log["pin_overflow"], log["overflow_reason"] = None, True, str(error)
        if condition == "D4L":
            selected = uniform if d4b is None else fill(pin_frames, uniform, max_id)
        elif d4b is None:
            selected = uniform
        elif condition == "D4B":
            selected = d4b.selected
            log["phase_borrow"] = d4b.phase_borrow
        else:
            log["phase_borrow"] = d4b.phase_borrow
            try:
                selected, info = rndb_select(d4b.selected, uniform, groups, max_id, state.exec_start_idx,
                                             evidence_frames(state.task, rows, max_id), state.sampler)
                log.update(info)
                d4b_deleted = set(uniform) - set(d4b.selected)
                log["d4b_deleted_d0_frames"] = sorted(d4b_deleted)
                if condition == "RNDC":
                    randoms = {r["random_frame"] for r in info["replacements"]}
                    kept_pins = set(pin_frames) & set(uniform)
                    rndc = phase_balanced_fill(sorted(randoms | kept_pins), uniform, max_id, state.exec_start_idx)
                    rndc_deleted = set(uniform) - set(rndc.selected)
                    selected = rndc.selected
                    log["phase_borrow"] = rndc.phase_borrow
                    log["deleted_jaccard_vs_d4b"] = (len(d4b_deleted & rndc_deleted) / len(d4b_deleted | rndc_deleted)
                                                     if d4b_deleted | rndc_deleted else 1.0)
                    log["rnd_deleted_set_equals_d4b"] = d4b_deleted == rndc_deleted
            except EvidenceBudgetError as error:
                selected, log["rndb_infeasible"], log["rndb_error"] = uniform, True, str(error)
    else:
        raise ValueError(f"unknown V-A condition {condition!r}")
    selected = [int(s) for s in selected]
    if selected != sorted(set(selected)) or any(s < 0 or s > max_id for s in selected):
        raise ValueError("selection must be sorted, unique and causal")
    if len(selected) != len(uniform):
        raise ValueError("selection must keep the D0 frame count")
    log.update(selection_stats(selected, uniform, max_id, state.exec_start_idx, part))
    log["selected"] = selected
    return selected, log
