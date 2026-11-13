import json

import numpy as np
import pytest

from src.memory_management import event_keyframes as ek
from src.memory_management import fruit_distance as fd


def test_nearest_ripe_fruit_still_on_the_tree():
    positions = [[0.3, 0.0, 0.5], [0.4, 0.0, 0.5], [0.35, 0.0, 0.5]]
    tcp = [0.3, 0.0, 0.5]
    assert fd.distance(tcp, positions, [True, True, True], [False, False, False]) == pytest.approx(0.0)
    assert fd.distance(tcp, positions, [True, True, True], [True, False, False]) == pytest.approx(0.05)
    assert fd.distance(tcp, positions, [True, True, False], [True, False, False]) == pytest.approx(0.1)


def test_no_pickable_fruit_is_far_not_unknown():
    d = fd.distance([0, 0, 0], [[0.1, 0, 0]], [True], [True])
    assert d == fd.NO_TARGET_M and np.isfinite(d)
    # the last pick closes with a release frame instead of staying open
    trace = [0.5] * 20 + [0.01] * 5 + [d] * 20
    events = ek.detect_events(trace, 0.03, 0.005)
    assert len(events) == 1 and events[0]["release"] == 25


def test_gt_row_and_ripe_flags():
    layout = json.dumps({"fruits": [{"ripe": True}, {"ripe": False}]})
    row = {"tcp_pos": [0, 0, 0], "fruits": [{"pos": [0.2, 0, 0], "attached": False}, {"pos": [0.1, 0, 0], "attached": False}]}
    assert fd.ripe_flags(layout) == [True, False]
    assert fd.from_gt_row(row, fd.ripe_flags(layout)) == pytest.approx(0.2)
