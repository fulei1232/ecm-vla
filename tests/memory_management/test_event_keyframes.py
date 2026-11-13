import math

import numpy as np
import pytest

from src.memory_management import event_keyframes as ek
from src.memory_management.validation.partition import even_sampling_indices


def trace(n, visits, far=0.3, near=0.005):
    """Distances that are `near` on each [start, stop) of `visits` and `far` elsewhere."""
    d = np.full(n, far)
    for start, stop in visits:
        d[start:stop] = near
    return d


def test_without_events_it_is_the_official_even_sampling():
    rng = np.random.default_rng(0)
    for t in (0, 5, 15, 16, 17, 40, 300, 1299):
        d = rng.uniform(0, 0.5, t + 1)
        for cfg in (ek.KeyframeConfig(events=0), ek.KeyframeConfig(events=8, radius=1e-6)):
            assert ek.select(d, t, cfg)[0] == even_sampling_indices(t, 16)


def test_events_contact_and_release():
    d = trace(200, [(20, 40), (100, 130)])
    d[25] = 0.001
    d[110:115] = np.nan
    events = ek.detect_events(d, 0.03, 0.005)
    assert [(e["start"], e["end"], e["contact"], e["release"]) for e in events] == [(20, 39, 20, 40), (100, 129, 100, 130)]
    d[22] = 0.0
    assert ek.detect_events(d, 0.03, 0.0)[0]["contact"] == 22
    # An event that is still going on has no release frame yet; unknown frames do not end it.
    ongoing = ek.detect_events(d[:112], 0.03, 0.005)
    assert ongoing[-1]["release"] is None and ongoing[-1]["end"] == 109
    # Unknown frames between two near frames do not split the event or start one.
    gap = trace(60, [(10, 20), (23, 30)])
    gap[20:23] = np.nan
    gap[40:45] = np.nan
    assert [(e["start"], e["release"]) for e in ek.detect_events(gap, 0.03, 0.005)] == [(10, 30)]


def test_keyframes_take_the_most_recent_events_within_the_cap():
    d = trace(400, [(20, 40), (100, 130), (200, 220), (300, 330)])
    for cap, want in ((0, []), (1, [300]), (2, [300, 330]), (3, [200, 300, 330]), (4, [200, 220, 300, 330]),
                      (12, [20, 40, 100, 130, 200, 220, 300, 330])):
        frames, info = ek.select(d, 399, ek.KeyframeConfig(events=cap))
        assert info["keyframes"] == want and set(want) <= set(frames)
    # At frame 310 the last event is ongoing: its contact is the newest keyframe, no release yet.
    _, info = ek.select(d, 310, ek.KeyframeConfig(events=4))
    assert info["keyframes"] == [100, 200, 220, 300] and info["events"] == 4


def test_selection_is_causal_sorted_unique_and_has_the_frame_count():
    rng = np.random.default_rng(1)
    for _ in range(300):
        t = int(rng.integers(16, 1300))
        d = rng.choice([0.002, 0.02, 0.2, math.nan], size=t + 1 + int(rng.integers(0, 50)), p=[0.2, 0.2, 0.5, 0.1])
        for k in (0, 4, 8, 12):
            frames, info = ek.select(d, t, ek.KeyframeConfig(events=k))
            assert frames == sorted(set(frames)) and len(frames) == 16 and 0 <= frames[0] and frames[-1] <= t
            assert len(info["keyframes"]) <= k and set(info["keyframes"]) <= set(frames)
            tail = d.copy()
            tail[t + 1:] = rng.uniform(0, 0.5, len(d) - t - 1)
            assert ek.select(tail, t, ek.KeyframeConfig(events=k))[0] == frames
    assert ek.select(np.zeros(10), 9, ek.KeyframeConfig(events=12))[0] == list(range(10))


def test_uniform_part_covers_the_episode_end_to_end():
    d = trace(500, [(100, 200)])
    frames, info = ek.select(d, 499, ek.KeyframeConfig(events=12))
    assert info["keyframes"] == [100, 200] and frames[0] == 0 and frames[-1] == 499
    assert len(frames) == 16 and max(np.diff(frames)) <= 41


def test_config_validation():
    for bad in ({"events": 16}, {"events": -1}, {"radius": 0.0}, {"radius": math.inf}, {"tol": -1.0}):
        with pytest.raises(ValueError):
            ek.KeyframeConfig(**bad)
    with pytest.raises(ValueError):
        ek.select(np.zeros(10), 20, ek.KeyframeConfig(events=4))
