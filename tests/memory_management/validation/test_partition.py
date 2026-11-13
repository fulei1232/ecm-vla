from src.memory_management.validation.partition import (
    even_sampling_indices, partition, phase_counts, recent_window)


def test_even_sampling_matches_recorded_official_d0(m3_episodes):
    checked = 0
    for ep in m3_episodes:
        for d in ep["decisions"]:
            assert even_sampling_indices(d["available_max_frame_id"], 32) == d["selected_frame_ids"]
            checked += 1
    assert checked > 100


def test_partition_sizes():
    p = partition(31)
    assert len(p.uniform) == 32 and not p.full
    assert p.anchor == (0,) and len(p.history) == 27 and p.recent == (28, 29, 30, 31)
    p = partition(467)
    assert p.full and len(p.history) == 27 and p.recent[-1] == 467
    assert set(p.anchor) | set(p.history) | set(p.recent) == set(p.uniform)


def test_ut19_recent_window_matches_framesamp_and_fits_ring_buffer():
    for F in (31, 32, 100, 467, 1339):
        r = recent_window(F)
        assert r == even_sampling_indices(F, 32)[-4:]
        assert F - r[0] + 1 <= 160
    assert max(F - recent_window(F)[0] + 1 for F in range(32, 1340)) <= 132


def test_ut21_valid_frames_equal_framesamp():
    for F in range(0, 600):
        assert len(partition(F).uniform) == min(F + 1, 32)


def test_phase_counts():
    assert phase_counts([0, 5, 10, 20], exec_start_idx=10) == {"demo": 2, "exec": 2}
