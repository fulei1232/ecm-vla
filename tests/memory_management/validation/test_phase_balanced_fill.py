import random
import sys

import pytest

from conftest import ACM_SCRIPTS
from src.memory_management.validation.acm_selection import EvidenceBudgetError
from src.memory_management.validation.partition import even_sampling_indices, phase_counts
from src.memory_management.validation.phase_balanced_fill import phase_balanced_fill


def acm_balanced_fill():
    sys.path.insert(0, str(ACM_SCRIPTS))
    from context_selection import balanced_fill
    return balanced_fill


def test_single_phase_equals_acm_balanced_fill():
    balanced_fill = acm_balanced_fill()
    rng = random.Random(0)
    for _ in range(150):
        F = rng.randint(32, 900)
        uniform = even_sampling_indices(F, 32)
        pins = sorted(rng.sample(range(F), rng.randint(0, 8)))
        ours = phase_balanced_fill(pins, uniform, F, exec_start_idx=0).selected
        assert ours == balanced_fill(pins, uniform, F, recent_slots=4)


def test_phase_counts_conserved_and_required_kept():
    rng = random.Random(1)
    for _ in range(150):
        exec_start = rng.randint(40, 200)
        F = exec_start + rng.randint(0, 300)
        uniform = even_sampling_indices(F, 32)
        demo_pins = rng.sample(range(1, exec_start), rng.randint(0, 6))
        exec_pins = rng.sample(range(exec_start, F), rng.randint(0, min(3, F - exec_start)))
        pins = sorted(set(demo_pins + exec_pins))
        try:
            result = phase_balanced_fill(pins, uniform, F, exec_start)
        except EvidenceBudgetError:
            continue
        sel = result.selected
        assert len(sel) == len(uniform) and sel == sorted(set(sel))
        assert set(pins) | {uniform[0]} | set(uniform[-4:]) <= set(sel)
        if result.phase_borrow == 0:
            assert phase_counts(sel, exec_start) == phase_counts(uniform, exec_start)


def test_exec_phase_untouched_when_pins_are_demo_only():
    exec_start, F = 114, 230
    uniform = even_sampling_indices(F, 32)
    sel = phase_balanced_fill([15, 63, 91, 111], uniform, F, exec_start).selected
    assert [u for u in uniform if u >= exec_start] == [s for s in sel if s >= exec_start]


def test_borrow_and_overflow():
    exec_start, F = 20, 400
    uniform = even_sampling_indices(F, 32)
    d0_demo = phase_counts(uniform, exec_start)["demo"]
    pins = list(range(1, 1 + d0_demo + 2))
    result = phase_balanced_fill(pins, uniform, F, exec_start)
    assert result.phase_borrow == len(pins) + 1 - d0_demo and result.borrowed_into == "demo"
    many = phase_balanced_fill(list(range(1, 19)), uniform, F, exec_start)
    assert many.phase_borrow == 19 - d0_demo and set(uniform[-4:]) <= set(many.selected)
    with pytest.raises(EvidenceBudgetError):
        phase_balanced_fill(list(range(1, 29)), uniform, F, exec_start)
    with pytest.raises(EvidenceBudgetError):
        phase_balanced_fill(pins, uniform, F, exec_start, allow_borrow=False)
