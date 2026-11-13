import numpy as np

from src.memory_management.validation import phase_expert as pe
from src.memory_management.validation import planner_branch as pb

GRASP = np.array([0.1, 0.2, 0.02])


def test_corridor():
    assert pe.in_corridor([0.1, 0.2, 0.15], GRASP, 0.15)
    assert pe.in_corridor([0.115, 0.2, 0.0], GRASP, 0.15)
    assert not pe.in_corridor([0.125, 0.2, 0.08], GRASP, 0.15)
    assert not pe.in_corridor([0.1, 0.2, 0.18], GRASP, 0.15)
    assert not pe.in_corridor([0.1, 0.2, -0.01], GRASP, 0.15)


def test_pickup_phase():
    above, high, away = [0.1, 0.2, 0.15], [0.1, 0.2, 0.08], [0.3, 0.2, 0.15]
    assert pe.pickup_phase(False, False, away, GRASP, 0.15, True) == "reach"
    assert pe.pickup_phase(False, False, above, GRASP, 0.15, True) == "descend"
    assert pe.pickup_phase(False, False, high, GRASP, 0.15, False) == "descend"
    assert pe.pickup_phase(False, True, GRASP, GRASP, 0.15, True) == "finish_close"
    assert pe.pickup_phase(False, True, GRASP, GRASP, 0.15, False) == "open"
    assert pe.pickup_phase(False, True, high, GRASP, 0.15, True) == "open"
    assert pe.pickup_phase(True, True, GRASP, GRASP, 0.15, True) == "lift"
    assert pe.pickup_phase(True, False, GRASP, GRASP, 0.15, True) == "close"


def test_release_and_button_phase():
    assert pe.release_phase(True) == "release" and pe.release_phase(False) == "retreat"
    button = [0.0, -0.2, 0.01]
    assert pe.button_phase(False, [0.3, 0.0, 0.15], button, 0.15) == "approach"
    assert pe.button_phase(False, [0.0, -0.2, 0.15], button, 0.15) == "close_press"
    assert pe.button_phase(True, [0.01, -0.2, 0.05], button, 0.15) == "press"


def test_pick_cube_retargets_only_when_the_planned_cube_is_gone():
    a, b, c = object(), object(), object()
    gone, held = set(), set()
    pick = lambda planned: pe.pick_cube(planned, [a, b, c], lambda x: x in gone, lambda x: x in held)  # noqa: E731
    assert pick(b) is b
    gone.add(b)
    assert pick(b) is a
    gone.add(a)
    assert pick(b) is c
    held.add(c)
    gone.clear()
    assert pick(a) is c
    held.clear()
    gone.update({a, b, c})
    assert pick(b) is b


class _Base:
    class spec:
        id = "PatternLock"

    def __init__(self, calls):
        self.task_list = [{"choice_label": "swing", "solve": lambda env, planner: calls.append((env, planner))}]


def test_unlisted_subgoals_run_the_original_solve():
    calls, run = [], pb.ExpertRun(t=0)
    solver = pe.PhaseSolver(_Base(calls), "wrapper", "planner", run)
    solver.solve(0)
    assert calls == [("wrapper", "planner")] and run.stats["phase"]["fallbacks"] == 1


def test_primitive_limit():
    run = pb.ExpertRun(t=0)
    base = _Base([])
    base.spec.id = "BinFill"
    base.task_list = [{"choice_label": "pick up the cube"}]
    solver = pe.PhaseSolver(base, None, None, run)
    steps = []
    solver._pickup = lambda kind, task, first: steps.append(first) or False
    solver.solve(0)
    assert steps == [True] + [False] * (pe.MAX_PRIMITIVES - 1) and run.stats["phase"]["primitive_limit"] == 1
    base.spec.id = "PatternLock"
