import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fruit_picking import layout as fl  # noqa: E402

SEEDS = range(40)


@pytest.mark.parametrize("task", fl.TASKS)
def test_layout_is_deterministic_per_task_and_seed(task):
    assert fl.sample_layout(task, 7).to_json() == fl.sample_layout(task, 7).to_json()
    assert fl.sample_layout(task, 7).to_json() != fl.sample_layout(task, 8).to_json()


def test_tasks_do_not_share_trees():
    a, b = fl.sample_layout("occluded_ripe", 3), fl.sample_layout("count", 3)
    assert [f["pos"] for f in a.fruits] != [f["pos"] for f in b.fruits]


@pytest.mark.parametrize("task", fl.TASKS)
def test_fruits_are_spaced_reachable_and_unblocked(task):
    for seed in SEEDS:
        L = fl.sample_layout(task, seed)
        pos = np.array([f["pos"] for f in L.fruits])
        assert 4 <= len(pos) <= 6
        rho, z = np.linalg.norm(pos[:, :2], axis=1), pos[:, 2]
        assert np.all((rho >= fl.FRUIT_RHO[0]) & (rho <= fl.FRUIT_RHO[1]))
        assert np.all((z >= fl.FRUIT_Z[0]) & (z <= fl.FRUIT_Z[1]))
        for i, f in enumerate(L.fruits):
            for j, g in enumerate(L.fruits):
                if i == j:
                    continue
                assert np.linalg.norm(pos[i] - pos[j]) >= fl.FRUIT_SPACING
                assert fl.segment_distance(pos[j], *fl.corridor(f)) >= g["radius"] + fl.CORRIDOR_CLEARANCE
        for i, twig in enumerate(L.twigs):
            assert fl.twig_clear(twig, L.fruits, owner=i)
        for leaf in L.leaves:
            assert fl.leaf_clear(np.asarray(leaf["pos"]), L.fruits)


def test_occluded_ripe_invariants():
    any_unripe_hidden = False
    for seed in SEEDS:
        L = fl.sample_layout("occluded_ripe", seed)
        ripe = [i for i, f in enumerate(L.fruits) if f["ripe"]]
        hidden = [o["fruit"] for o in L.occluders]
        assert 2 <= len(ripe) < len(L.fruits)
        assert len(hidden) == 2 and len(set(hidden)) == 2
        assert any(L.fruits[i]["ripe"] for i in hidden)
        any_unripe_hidden |= any(not L.fruits[i]["ripe"] for i in hidden)
        assert sorted(L.targets) == ripe and L.n_pick == len(ripe)
        ys = [L.fruits[i]["pos"][1] for i in L.targets]
        assert ys == sorted(ys, reverse=True)
        for o in L.occluders:
            assert np.asarray(o["closed"])[2] < np.asarray(o["open"])[2]
    assert any_unripe_hidden, "an occluder should sometimes cover a green fruit, so hiding is not a ripeness cue"


def test_count_invariants():
    picks = set()
    for seed in SEEDS:
        L = fl.sample_layout("count", seed)
        assert all(f["ripe"] for f in L.fruits) and not L.occluders
        assert 2 <= L.n_pick < len(L.fruits) and len(L.targets) == L.n_pick
        assert str(L.n_pick) not in L.instruction and L.instruction.startswith("pick ")
        picks.add(L.n_pick)
    assert picks >= {2, 3, 4}
