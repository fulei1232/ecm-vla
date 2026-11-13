import math

from src.memory_management.validation.acm_selection import time_stratum
from src.memory_management.validation.matched_controls import (
    PersistentRandomReplacements, jitb_select, rndb_select)
from src.memory_management.validation.partition import partition, phase_counts, phase_of
from src.memory_management.validation.phase_balanced_fill import phase_balanced_fill


def test_ut22_rndb_matches_d4b_deletions_and_is_persistent():
    exec_start = 114
    groups = {"reveal_0:0": 15, "reveal_0:1": 63, "swap_0:0": 91, "swap_0:1": 111}
    evidence = {14, 15, 16, 62, 63, 64, 90, 91, 92, 110, 111, 112}
    sampler = PersistentRandomReplacements("VideoUnmaskSwap", 3, 7)
    first = {}
    for F in range(150, 400, 16):
        part = partition(F)
        uniform = list(part.uniform)
        d4b = phase_balanced_fill(sorted(groups.values()), uniform, F, exec_start).selected
        sel, info = rndb_select(d4b, uniform, groups, F, exec_start, evidence, sampler)
        assert info["rnd_deleted_set_equals_d4b"]
        assert sorted(set(uniform) - set(sel)) == sorted(set(uniform) - set(d4b))
        assert len(sel) == len(uniform) and sel == sorted(set(sel))
        assert phase_counts(sel, exec_start) == phase_counts(d4b, exec_start)
        for rep in info["replacements"]:
            assert rep["same_phase"]
            assert all(abs(rep["random_frame"] - e) > 4 for e in evidence) or rep["evidence_overlap"]
            if not rep["rnd_collision_redraw"]:
                first.setdefault(rep["group"], rep["random_frame"])
                assert first[rep["group"]] == rep["random_frame"]
    assert first


def test_rndb_prefers_d4b_neighbour_interval():
    sampler = PersistentRandomReplacements("BinFill", 0, 7)
    F, exec_start = 300, 0
    uniform = list(partition(F).uniform)
    groups = {"drop_red_1:0": 151, "drop_red_1:1": 155}
    d4b = phase_balanced_fill(sorted(groups.values()), uniform, F, exec_start).selected
    _, info = rndb_select(d4b, uniform, groups, F, exec_start, set(range(145, 162)), sampler)
    for rep in info["replacements"]:
        assert not rep["evidence_overlap"]
        assert not 145 - 4 <= rep["random_frame"] <= 161 + 4
        assert rep["distance_to_pin"] <= 161 + 4 - 151 + 1 + max(2, (F // 31) // 2)


def test_jitb_invariants():
    for exec_start in (0, 114):
        for F in range(max(32, exec_start), 900, 37):
            part = partition(F)
            sel, info = jitb_select(part, exec_start, "T", 1, F, 7)
            uniform = list(part.uniform)
            assert len(sel) == len(uniform) and sel == sorted(set(sel))
            assert set(part.anchor) | set(part.recent) <= set(sel)
            assert phase_counts(sel, exec_start) == phase_counts(uniform, exec_start)
            hist = [s for s in sel if s not in set(part.anchor) | set(part.recent)]
            s = F / 31
            assert all(abs(a - b) <= max(1, math.ceil(s)) for a, b in zip(hist, part.history))
            assert all(phase_of(a, exec_start) == phase_of(b, exec_start) for a, b in zip(hist, part.history))
            assert jitb_select(part, exec_start, "T", 1, F, 7)[0] == sel


def test_jitb_is_identity_when_stride_below_two():
    part = partition(50)
    assert jitb_select(part, 0, "T", 0, 0, 7)[0] == list(part.uniform)
