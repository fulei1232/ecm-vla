import numpy as np

from src.memory_management.validation.teacher_probe import (
    causal_peak_frames,
    decision_boundaries,
    kemo_slowdown_saliency,
    select_v_b_candidates,
    stable_seed,
)
from src.memory_management.validation.demo_source import action_chunk
from src.memory_management.validation.cached_flow_loss import common_random_conditions


def test_decision_boundaries_are_early_middle_late_and_eligible():
    cuts = decision_boundaries(600, 120)
    assert len(cuts) == 3
    assert cuts == sorted(set(cuts))
    assert all(c >= 120 and c >= 64 and c < 600 and (c - 120) % 16 == 0 for c in cuts)
    assert decision_boundaries(100, 80) == []


def test_slowdown_saliency_and_peak_confirmation_are_causal():
    q = np.zeros((60, 7), dtype=float)
    q[:, 0] = np.arange(60)
    q[24:, 0] = 24 + np.arange(36) * 0.01
    score = kemo_slowdown_saliency(q, window=4)
    peaks = causal_peak_frames(score, cutoff_frame=27, left=3, right=2, min_gap=1)
    assert score[30] > score[10]
    assert all(frame + 2 <= 27 for frame, _ in peaks)
    assert all(value > 0 for _, value in peaks)


def test_candidate_selection_is_causal_unique_and_reproducible():
    q = np.zeros((200, 7), dtype=float)
    q[:, 0] = np.arange(200)
    q[40:, 0] = 40 + np.arange(160) * 0.01
    kwargs = dict(task="BinFill", episode_id=21, model_seed=7, cutoff_frame=120,
                  recent_first_frame=105, d0_frames=list(range(0, 121, 4)),
                  joint_state=q, subgoal_boundaries=[90, 110])
    a = select_v_b_candidates(**kwargs)
    b = select_v_b_candidates(**kwargs)
    assert a == b
    assert len({x["frame_id"] for x in a}) == len(a)
    assert all(x["frame_id"] <= 120 and x["frame_id"] < 105 for x in a)
    assert stable_seed("x", 1) == stable_seed("x", 1)


def test_expert_action_chunk_pads_with_last_valid_action():
    actions = np.arange(24, dtype=np.float32).reshape(3, 8)
    chunk, valid = action_chunk(actions, 1, horizon=4)
    assert valid == 2
    np.testing.assert_array_equal(chunk[:2], actions[1:])
    np.testing.assert_array_equal(chunk[2:], np.repeat(actions[-1:], 2, axis=0))


def test_common_random_conditions_are_shared_and_stratified():
    noise_a, tau_a = common_random_conditions("BinFill", 22, 128, 16, 20, 32)
    noise_b, tau_b = common_random_conditions("BinFill", 22, 128, 16, 20, 32)
    np.testing.assert_array_equal(noise_a, noise_b)
    np.testing.assert_array_equal(tau_a, tau_b)
    assert noise_a.shape == (16, 20, 32)
    assert tau_a.shape == (16,) and np.all(np.diff(tau_a) > 0)
    assert np.all((tau_a > 0.001) & (tau_a < 1.0))
