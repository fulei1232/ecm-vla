import json
import math

import numpy as np
import pytest

from src.memory_management.validation import event_memory as em
from src.memory_management.validation.partition import partition


def near_target(n, center, width=10, near=0.01, far=0.4):
    """Distances that dip to `near` within `width` frames of `center`."""
    d = np.full(n, far)
    d[max(center - width, 0):center + width + 1] = near
    return d


def write_raw_frames(directory, gts):
    with (directory / "raw_frames.jsonl").open("w") as f:
        for i, gt in enumerate(gts):
            f.write(json.dumps({"frame_id": i, "exec_start_idx": 0, "gt": gt}) + "\n")


def test_proximity():
    p = em.proximity([0.0, 0.05, 0.10, math.nan, 0.02], tau=0.05, d0=0.02)
    np.testing.assert_allclose(p, [1.0, math.exp(-0.6), math.exp(-1.6), 0.0, 1.0])


def test_water_fill_and_systematic_pick_distinct_frames():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = int(rng.integers(5, 400))
        k = int(rng.integers(1, n + 1))
        w = rng.exponential(size=n) * (rng.random(n) < rng.choice([0.05, 0.3, 1.0]))
        w[rng.integers(n)] += 1.0
        filled = em.water_fill(w, 1.0 / k)
        assert abs(filled.sum() - 1) < 1e-9 and filled.max() <= 1.0 / k + 1e-12
        picks = em.systematic(filled, k)
        assert len(set(picks.tolist())) == k and picks.min() >= 0 and picks.max() < n
    with pytest.raises(ValueError):
        em.water_fill(np.ones(3), 0.2)


def test_event_selection_keeps_anchor_recent_and_the_frame_count():
    rng = np.random.default_rng(1)
    for max_id in (31, 32, 40, 100, 300, 1200):
        part = partition(max_id)
        d = rng.uniform(0, 0.5, max_id + 1)
        d[rng.random(max_id + 1) < 0.2] = np.nan
        for alpha in (0.25, 0.5, 1.0):
            sel = em.event_selection(part, em.proximity(d, 0.05), alpha)
            assert sel == sorted(set(sel)) and len(sel) == len(part.uniform)
            assert set(part.anchor) | set(part.recent) <= set(sel) and max(sel) == max_id
    part = partition(300)
    uniform = list(part.uniform)
    assert em.event_selection(part, em.proximity(near_target(301, 150), 0.05), 0.0) == uniform
    assert em.event_selection(part, np.zeros(301), 1.0) == uniform
    assert em.event_selection(partition(20), np.ones(21), 1.0) == list(partition(20).uniform)


def test_event_memory_is_denser_near_the_target_and_still_covers_the_episode():
    part, d = partition(300), near_target(301, 150)
    p = em.proximity(d, 0.05)
    window = lambda frames: sum(140 <= f <= 160 for f in frames)  # noqa: E731
    u, half, full = list(part.uniform), em.event_selection(part, p, 0.5), em.event_selection(part, p, 1.0)
    assert window(u) < window(half) < window(full) and window(full) == 21
    assert max(np.diff(half)) < max(np.diff(full))
    anti = em.event_selection(part, np.where(np.isfinite(d), 1 - p, 0), 1.0)
    assert window(anti) < window(u)


def test_matched_random_replaces_the_same_frames_with_random_ones():
    part, d = partition(300), near_target(301, 150)
    ev = em.event_selection(part, em.proximity(d, 0.05), 0.5)
    evr = em.matched_random(part, ev, seed=11)
    uniform = set(part.uniform)
    assert uniform - set(evr) == uniform - set(ev) and len(evr) == len(ev)
    assert all(f not in uniform and 1 <= f < part.recent[0] for f in set(evr) - uniform)
    assert evr == em.matched_random(part, ev, seed=11) and evr != em.matched_random(part, ev, seed=12)
    assert em.matched_random(part, list(part.uniform), seed=1) == list(part.uniform)


def test_strategy_memories_and_event_pool():
    cfg = em.EventMemoryConfig(alpha=0.5)
    part, d = partition(300), near_target(301, 150)
    memories = em.strategy_memories(part, d, cfg, seed=3)
    assert list(memories) == cfg.candidates() == ["EV", "EVU", "EVR", "EVA", "EV@0.25", "EV@0.5", "EV@0.75", "EV@1"]
    assert memories["EV"] == memories["EV@0.5"] and all(len(m) == 32 for m in memories.values())
    window = lambda frames: sum(140 <= f <= 160 for f in frames)  # noqa: E731
    assert memories["EVU"] != list(part.uniform) and window(memories["EVU"]) <= window(part.uniform) + 1
    assert memories["EVU"] == em.strategy_memories(part, np.full(301, np.nan), cfg, seed=3)["EVU"]
    pool = em.event_pool(part, d, cfg, 8)
    assert len(pool) == 8 and not set(pool) & set(part.uniform) and sum(140 <= f <= 160 for f in pool) >= 4
    assert em.event_pool(part, np.full(301, np.nan), cfg, 8) is None


def test_config_validation_and_per_task_scales():
    assert em.EventMemoryConfig.from_params({"alphas": [0.5], "tau": 0.1}).alphas == (0.5,)
    cfg = em.EventMemoryConfig.from_params({"tau": {"BinFill": 0.05, "VideoUnmaskSwap": 0.15}, "d0": 0.01})
    assert (cfg.for_task("VideoUnmaskSwap").tau, cfg.for_task("VideoUnmaskSwap").d0) == (0.15, 0.01)
    with pytest.raises(KeyError):
        cfg.for_task("PatternLock")
    for bad in ({"alpha": 1.5}, {"tau": 0}, {"pool": "best"}, {"d0": -1}, {"tau": {"BinFill": -1}}):
        with pytest.raises(ValueError):
            em.EventMemoryConfig(**bad)


def test_gt_distance():
    tcp = {"tcp_position": [[0.0, 0.0, 0.1]]}
    buttons = {**tcp, "selected_buttons": [{"name": "target_1", "position": [[0.3, 0.0, 0.01]]},
                                           {"name": "target_2", "position": [[0.0, 0.2, 0.01]]}]}
    assert em.gt_distance("PatternLock", buttons) == pytest.approx(0.2)
    cubes = {**tcp, "binfill_language_sequence": [["red", 2]],
             "all_cubes": [{"name": "cube_blue_0", "position": [[0.05, 0.0, 0.1]]},
                           {"name": "cube_red_0", "position": [[10.0, 10.0, -0.9]]},
                           {"name": "cube_red_1", "position": [[0.0, 0.0, 0.4]]}]}
    assert em.gt_distance("BinFill", cubes) == pytest.approx(0.3)
    assert math.isnan(em.gt_distance("VideoUnmaskSwap", {"spawned_bins": []}))
    assert math.isnan(em.gt_distance("VideoUnmaskSwap", {**tcp, "spawned_bins": []}))


def test_ground_truth_signal_reads_the_episode_incrementally(tmp_path):
    gts = [{"tcp_position": [[0.0, 0.0, 0.0]], "spawned_bins": [{"name": "bin_0", "position": [[0.0, 0.0, z]]}]}
           for z in (0.3, 0.2, 0.1)]
    write_raw_frames(tmp_path, gts[:2])
    signal = em.make_signal({"kind": "ground_truth"})
    signal.reset({"task": "VideoUnmaskSwap", "directory": str(tmp_path)})
    np.testing.assert_allclose(signal.distances(1), [0.3, 0.2])
    write_raw_frames(tmp_path, gts)
    signal.tail.offset = 0
    signal.tail.rows = []
    np.testing.assert_allclose(signal.distances(2), [0.3, 0.2, 0.1])
    with pytest.raises(ValueError):
        signal.distances(3)


PLUGIN = '''
class Plugin:
    def __init__(self, scale=1.0):
        self.scale, self.contexts = scale, []
    def reset(self, context):
        self.contexts.append(context)
    def distance(self, frame):
        return None if frame.is_demo else self.scale * frame.frame_id
'''


def test_external_signal_loads_the_perception_module(tmp_path):
    (tmp_path / "my_perception.py").write_text(PLUGIN)
    signal = em.make_signal({"kind": "external", "factory": "my_perception:Plugin", "kwargs": {"scale": 0.5},
                             "path": str(tmp_path)})
    assert signal.needs_frames
    signal.reset({"task": "BinFill"})
    frame = lambda i, demo=False: em.FrameInput("BinFill", i, np.zeros((2, 2, 3), np.uint8), np.zeros(8), demo)  # noqa: E731
    signal.observe([frame(0, True), frame(1), frame(2)])
    np.testing.assert_array_equal(signal.distances(2), [np.nan, 0.5, 1.0])
    with pytest.raises(ValueError):
        signal.observe([frame(4)])
    with pytest.raises(ValueError):
        signal.distances(3)
    assert signal.module.contexts == [{"task": "BinFill"}]


def test_template_is_a_valid_but_unimplemented_signal():
    signal = em.make_signal({"kind": "external", "factory": "src.memory_management.event_signal_template:"
                                                           "EndEffectorTargetDistance"})
    signal.reset({"task": "BinFill"})
    with pytest.raises(NotImplementedError):
        signal.observe([em.FrameInput("BinFill", 0, np.zeros((2, 2, 3), np.uint8), np.zeros(8), False)])


def test_decode_distances():
    rows = [{"decision": 0, "first_frame": 0, "distance": [0.1, None]}, {"decision": 1, "first_frame": 2, "distance": [0.3]}]
    np.testing.assert_array_equal(em.decode_distances(rows, 2), [0.1, np.nan, 0.3])
    with pytest.raises(ValueError):
        em.decode_distances(rows[1:], 2)
    with pytest.raises(ValueError):
        em.decode_distances(rows, 3)
