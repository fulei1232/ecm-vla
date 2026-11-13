"""Phase-balanced evidence insertion for the D4B condition (plan §0A.3).

Each phase (demo / exec) keeps exactly the number of slots D0 gives it, must keep its pins plus the
anchor and recent-window slots that fall in it, and otherwise picks from its own D0 frames with the
two-pass objective of ACM balanced_fill: minimize the largest gap, then squared displacement from
the reference slots. With a single phase the result equals ACM balanced_fill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .acm_selection import EvidenceBudgetError
from .partition import DEMO, EXEC, phase_of


@dataclass
class FillResult:
    selected: list[int]
    phase_borrow: int = 0
    borrowed_into: str | None = None
    per_phase: dict[str, list[int]] = field(default_factory=dict)


def balanced_subset(candidates, required, n, reference, lo=None, hi=None):
    """Choose n sorted frames from candidates including all required ones.

    Objective is lexicographic: (largest gap, sum of squared displacement against reference).
    Gaps include lo -> first and last -> hi when those boundaries are given.
    """
    candidates = sorted(set(candidates))
    required = set(required)
    if not required <= set(candidates):
        raise ValueError("required frames must be candidates")
    if len(reference) != n:
        raise ValueError("reference must have n entries")
    if len(required) > n or n > len(candidates):
        raise EvidenceBudgetError(f"cannot choose {n} of {len(candidates)} with {len(required)} required")
    if n == 0:
        return []

    def solve(gap_bound=None):
        states = {(0, None): (0, ())}
        for frame in candidates:
            following = {}
            for (count, last), (cost, chosen) in states.items():
                if frame not in required:
                    key = (count, last)
                    following[key] = min(following.get(key, (float("inf"), ())), (cost, chosen))
                if count == n:
                    continue
                previous = last if last is not None else lo
                gap = frame - previous if previous is not None else 0
                if gap_bound is not None and gap > gap_bound:
                    continue
                new_cost = max(cost, gap) if gap_bound is None else cost + (frame - reference[count]) ** 2
                key = (count + 1, frame)
                following[key] = min(following.get(key, (float("inf"), ())), (new_cost, chosen + (frame,)))
            states = following
        best = None
        for (count, last), (cost, chosen) in states.items():
            if count != n:
                continue
            tail = hi - last if hi is not None else 0
            if gap_bound is not None and tail > gap_bound:
                continue
            final = max(cost, tail) if gap_bound is None else cost
            if best is None or (final, chosen) < best:
                best = (final, chosen)
        return best

    optimum = solve()
    if optimum is None:
        raise EvidenceBudgetError("no feasible balanced subset")
    return list(solve(gap_bound=optimum[0])[1])


def phase_balanced_fill(pins, uniform, max_id, exec_start_idx, recent_slots=4, allow_borrow=True) -> FillResult:
    uniform = list(uniform)
    if (not uniform or uniform != sorted(set(uniform)) or uniform[-1] != max_id
            or any(i < 0 or i > max_id for i in list(pins) + uniform)):
        raise ValueError("expected causal sorted unique D0 frames ending at max_id")
    required = set(pins) | {uniform[0]} | set(uniform[-recent_slots:])
    if len(required) > len(uniform):
        raise EvidenceBudgetError(f"pins plus protected slots need {len(required)} of {len(uniform)}")

    phases = [p for p in (DEMO, EXEC) if any(phase_of(u, exec_start_idx) == p for u in uniform)
              or any(phase_of(r, exec_start_idx) == p for r in required)]
    d0 = {p: [u for u in uniform if phase_of(u, exec_start_idx) == p] for p in (DEMO, EXEC)}
    req = {p: {r for r in required if phase_of(r, exec_start_idx) == p} for p in (DEMO, EXEC)}
    slots = {p: len(d0[p]) for p in (DEMO, EXEC)}

    borrow, borrowed_into = 0, None
    for p, q in ((DEMO, EXEC), (EXEC, DEMO)):
        deficit = len(req[p]) - slots[p]
        if deficit > 0:
            if not allow_borrow or len(req[q]) > slots[q] - deficit:
                raise EvidenceBudgetError(f"phase {p} needs {len(req[p])} slots, has {slots[p]}")
            slots[p] += deficit
            slots[q] -= deficit
            borrow, borrowed_into = deficit, p

    per_phase = {}
    for p in phases:
        n = slots[p]
        candidates = sorted(set(d0[p]) | req[p])
        if p == DEMO:
            lo, hi = None, exec_start_idx
            span = (0, exec_start_idx - 1)
        else:
            lo = d0[DEMO][-1] if d0[DEMO] else None
            hi = max_id
            span = (exec_start_idx, max_id)
        if n == len(d0[p]):
            reference = d0[p]
        else:
            reference = np.linspace(span[0], span[1], n).round().astype(int).tolist() if n else []
        if n > len(candidates):
            raise EvidenceBudgetError(f"phase {p} has {len(candidates)} candidates for {n} slots")
        per_phase[p] = balanced_subset(candidates, req[p], n, reference, lo=lo, hi=hi)

    selected = sorted(f for p in phases for f in per_phase[p])
    assert len(selected) == len(uniform) == len(set(selected)), (selected, uniform)
    assert required <= set(selected)
    return FillResult(selected=selected, phase_borrow=borrow, borrowed_into=borrowed_into, per_phase=per_phase)
