import numpy as np
import pytest

from src.memory_management.config import MemoryConfig
from src.memory_management.types import FrameRecord
from src.memory_management.event_detection import CausalEventDetector, EventDetectorConfig
from src.memory_management.candidate_generation import CandidateGeneratorConfig, CausalCandidateGenerator
from src.memory_management.online_controller import OnlineMemoryController


CFG = MemoryConfig(image_dim=8, position_dim=12, candidate_stride=8)


def make_frame(i, state0=0.0, feature=1.0):
    state = np.zeros(8, np.float32)
    state[0] = state0
    state[7] = state0
    return FrameRecord("e", i, 0, i + 1.0,
                       np.full((1, 16, 8), feature, np.float32),
                       np.zeros((1, 16, 12), np.float32), state,
                       vision_global_feat=np.array([feature, 0, 0], np.float32))


def detector():
    return CausalEventDetector(EventDetectorConfig(peak_window=2, min_peak_separation=3,
                                                    saliency_threshold=1.0,
                                                    velocity_scale=0.01,
                                                    displacement_scale=0.01,
                                                    gripper_scale=0.05))


def test_causal_peak_has_no_future_confirmation():
    d = detector()
    assert not d.observe(make_frame(0)).is_peak
    assert d.observe(make_frame(1, 0.2)).is_peak
    assert d.observations[-1].event_ids[0].startswith("e:event:1:")


def test_future_perturbation_does_not_change_prefix():
    base, changed = detector(), detector()
    for i in range(8):
        a = base.observe(make_frame(i, 0.2 if i == 1 else 0.0)).event_ids
        b = changed.observe(make_frame(i, 0.2 if i == 1 else 0.0, feature=99.0 if i >= 5 else 1.0)).event_ids
        assert a == b


def test_periodic_candidate_and_visual_dedup_and_event_merge():
    g = CausalCandidateGenerator(CandidateGeneratorConfig(periodic_stride=2,
                                                            visual_similarity_threshold=0.999,
                                                            visual_dedup_window=8),
                                 EventDetectorConfig(peak_window=1, min_peak_separation=3,
                                                     saliency_threshold=1.0,
                                                     velocity_scale=0.01,
                                                     displacement_scale=0.01,
                                                     gripper_scale=0.05))
    outputs = []
    for i in range(8):
        outputs.extend(g.observe(make_frame(i, 0.2 if i == 2 else 0.0)))
    assert any("periodic" in c.sources for c in outputs)
    assert any("event" in c.sources for c in outputs)
    assert all(c.confirmed_at_frame == c.frame.frame_id for c in outputs)
    assert len([c for c in outputs if c.frame.frame_id == 2]) == 1


def test_candidate_priority_prefers_event_saliency_then_stable_time():
    from src.memory_management.types import Candidate
    low = Candidate(make_frame(2), frozenset({"event"}), ("motion_peak",), 1.0, 2,
                    ("e:event:2:motion_peak",))
    high = Candidate(make_frame(4), frozenset({"event"}), ("motion_peak",), 2.0, 4,
                     ("e:event:4:motion_peak",))
    periodic = Candidate(make_frame(1), frozenset({"periodic"}), confirmed_at_frame=1)
    assert high.priority > low.priority > periodic.priority


def test_controller_online_candidate_integration_preserves_bound():
    g = CausalCandidateGenerator(CandidateGeneratorConfig(periodic_stride=8,
                                                            visual_similarity_threshold=0.0), detector().config)
    c = OnlineMemoryController(CFG, "fifo", candidate_generator=g)
    c.reset("e", 0)
    for i in range(140):
        c.ingest(make_frame(i, 0.2 if i in (20, 60, 100) else 0.0))
    stats = c.stats()
    assert stats["bank_frames"] == 32
    assert stats["candidate_records"] == 140
    assert stats["candidate_queue"] >= 0
    assert any(stats["selected_sources"].values())


def test_detector_rejects_noncontiguous_stream():
    d = detector()
    d.observe(make_frame(0))
    with pytest.raises(ValueError, match="contiguous"):
        d.observe(make_frame(2))


def test_adaptive_scale_suppresses_startup_and_flat_motion_peaks():
    cfg = EventDetectorConfig(peak_window=2, min_peak_separation=3,
                              saliency_threshold=1.0, velocity_scale=0.02,
                              displacement_scale=0.02, gripper_scale=0.05,
                              adaptive_scale=True, scale_window=8,
                              scale_floor=1e-4)
    d = CausalEventDetector(cfg)
    peaks = []
    for i in range(30):
        # A steady motion background plus one causal impulse.
        state = 0.05 * i + (0.2 if i == 15 else 0.0)
        obs = d.observe(make_frame(i, state))
        if obs.is_peak:
            peaks.append(i)
    assert peaks == [15]
