"""M1 acceptance: tensors, atomic updates, bounded storage, phases and source time."""
from dataclasses import replace
import numpy as np
import pytest

from src.memory_management.config import MemoryConfig
from src.memory_management.types import FrameRecord, Candidate, MemorySlot
from src.memory_management.rolling_buffer import RollingObservationBuffer
from src.memory_management.online_controller import OnlineMemoryController
from src.memory_management.packer import ManagedMemoryPacker
from src.memory_management.policy_adapter import install_memory
from src.memory_management.validation.partition import partition, phase_counts

SMALL = MemoryConfig(image_dim=8, position_dim=12)


def frame(i, cfg=SMALL, episode="test", decision=0):
    return FrameRecord(episode, i, decision, None,
                       np.full((1, 16, cfg.image_dim), i + .125, np.float32),
                       np.full((1, 16, cfg.position_dim), i + .25, np.float32),
                       np.arange(8, dtype=np.float32) + i)


def stream(end, cfg=SMALL, mode="fifo", boundary=0, seed=7):
    controller = OnlineMemoryController(cfg, mode, seed)
    controller.reset("test", boundary)
    for i in range(end + 1):
        controller.ingest(frame(i, cfg))
    controller.update()
    return controller


@pytest.mark.parametrize("count", [1, 2, 31, 32])
def test_ut01_02_tensor_shapes_mask_and_official_padding_normalization(count):
    cfg = MemoryConfig()
    frames = [frame(i, cfg) for i in range(count)]
    slots = [MemorySlot(f, "history", "exec") for f in frames]
    normalize = lambda x: (x - 3) / 2
    result = ManagedMemoryPacker(cfg, normalize).pack(slots, max_frame_id=count - 1, decision_id=0)
    assert [v.shape for v in result.values()] == [(512, 2048), (512, 768), (512, 8), (512,)]
    assert result["static_mask"].dtype == bool
    assert result["static_mask"].sum() == 16 * count
    assert result["static_mask"][:16 * count].all() and not result["static_mask"][16 * count:].any()
    assert not result["static_image_emb"][16 * count:].any()
    assert not result["static_pos_emb"][16 * count:].any()
    assert (result["static_state_emb"][16 * count:] == -1.5).all()
    np.testing.assert_array_equal(result["static_pos_emb"][:16], frames[0].pos_feat.reshape(16, 768))


def test_ut04_05_atomic_single_replacement_and_protection():
    c = stream(100)
    before = c.bank.snapshot()
    ids = {s.frame_id for s in before}
    eligible = [i for i in range(min(c.bank.recent_frame_ids)) if i not in ids]
    candidate = Candidate(frame(eligible[-1]), confirmed_at_frame=100)
    victim = before[2].frame_id
    assert before[2].is_history
    result = c.bank.commit_replace(victim, candidate)
    assert result["accepted"]
    after = c.bank.snapshot()
    assert {s.frame_id for s in after} == ids - {victim} | {candidate.frame.frame_id}
    for a in after:
        if a.frame_id != candidate.frame.frame_id:
            assert any(a is s for s in before)
    saved = c.bank.snapshot()
    for victim in [0, c.bank.recent_frame_ids[-1]]:
        with pytest.raises(ValueError, match="protected"):
            c.bank.commit_replace(victim, Candidate(frame(eligible[0]), confirmed_at_frame=100))
        assert c.bank.snapshot() is saved
    with pytest.raises(ValueError, match="future"):
        c.bank.commit_replace(after[1].frame_id, Candidate(frame(101), confirmed_at_frame=101))
    assert c.bank.snapshot() is saved


def test_ut06_duplicate_merges_sources_without_replacement():
    c = stream(100)
    original = c.bank.snapshot()
    existing = next(s for s in original if s.is_history)
    c.bank.commit_replace(0, Candidate(existing.frame, frozenset({"event"}), ("contact",), confirmed_at_frame=100))
    c.bank.commit_replace(0, Candidate(existing.frame, frozenset({"periodic"}), confirmed_at_frame=100))
    merged = next(s for s in c.bank.snapshot() if s.frame_id == existing.frame_id)
    assert merged.sources >= {"event", "periodic"} and merged.event_types == ("contact",)
    assert [s.frame_id for s in original] == [s.frame_id for s in c.bank.snapshot()]


def test_ut07_monotonic_episode_reset_and_owned_arrays():
    c = stream(200, mode="random")
    with pytest.raises(ValueError, match="contiguous"):
        c.ingest(frame(199))
    with pytest.raises(ValueError, match="episode"):
        c.ingest(frame(201, episode="other"))
    c.reset("other", 0)
    c.ingest(frame(0, episode="other"))
    assert c.stats()["bank_frames"] == 1 and c.stats()["rolling_frames"] == 1
    assert c.bank.snapshot()[0].frame.episode_id == "other"
    f = frame(3)
    assert f.image_feat.base is None and not f.image_feat.flags.writeable
    with pytest.raises(ValueError):
        f.raw_robot_state[0] = 42


@pytest.mark.parametrize("mode", ["fifo", "random"])
def test_ut15_candidates_requery_current_bank_and_rng_reset(mode):
    c = stream(100, mode=mode)
    ids = {s.frame_id for s in c.bank.snapshot()}
    missing = [i for i in range(1, min(c.bank.recent_frame_ids)) if i not in ids]
    candidates = [Candidate(frame(i), confirmed_at_frame=100) for i in missing[-3:]]
    first = c.bank.apply_candidates(candidates)
    assert len(first) == 3 and all(r["accepted"] for r in first)
    replay = stream(100, mode=mode)
    assert replay.bank.apply_candidates(candidates) == first
    assert replay.stats()["selected_frame_ids"] == c.stats()["selected_frame_ids"]


def test_ut17_baseline_flag_is_exact_noop():
    class Baseline:
        def __init__(self):
            self.counter = 0
        def infer(self):
            self.counter += 1
            return self.counter
    policy = Baseline()
    before = vars(policy).copy()
    assert install_memory(policy, "baseline_framesamp") is None
    assert vars(policy) == before and policy.infer() == 1
    with pytest.raises(ValueError):
        install_memory(policy, "learned")


@pytest.mark.parametrize("F", [31, 32, 100, 467, 1339])
def test_ut19_exact_recent_window(F):
    c = stream(F)
    assert c.stats()["recent_frame_ids"] == list(partition(F).recent)
    assert c.stats()["recent_window_approx"] is False


@pytest.mark.parametrize("mode", ["fifo", "random"])
def test_ut18_20_21_long_stream_phase_quota_fullness_and_storage(mode):
    cfg = SMALL
    c = OnlineMemoryController(cfg, mode)
    c.reset("test", 280)
    peak, bytes_at = 0, {}
    for i in range(6201):
        c.ingest(frame(i))
        if i < 34 or i % 16 == 0 or i in (100, 467, 1339, 4097, 6200):
            c.update()
            st = c.stats()
            assert st["bank_frames"] == min(i + 1, 32)
            assert st["rolling_frames"] <= 160 and st["retained_unique_frames"] <= 192
            assert st["phase_slot_counts"] == phase_counts(partition(i).uniform, 280)
            assert st["selected_frame_ids"] == sorted(set(st["selected_frame_ids"]))
            peak = max(peak, st["retained_unique_bytes"])
            if i in (1339, 4097, 6200):
                bytes_at[i] = st["retained_unique_bytes"]
    assert c.stats()["recent_window_approx"]  # fixed ring cannot preserve arbitrarily old exact D0 recent frames
    assert max(bytes_at.values()) <= 192 * frame(0).nbytes
    assert peak <= 192 * frame(0).nbytes
    packed = c.prepare()
    for i, slot in enumerate(c.bank.snapshot()):
        np.testing.assert_array_equal(packed["static_pos_emb"][i * 16:(i + 1) * 16], slot.frame.pos_feat[0])


def test_future_availability_and_nonfinite_inputs_rejected():
    c = stream(100)
    before = c.bank.snapshot()
    future = Candidate(frame(13, decision=1), confirmed_at_frame=100)
    with pytest.raises(ValueError, match="available"):
        c.bank.commit_replace(next(s.frame_id for s in before if s.is_history), future)
    assert c.bank.snapshot() is before
    with pytest.raises(ValueError, match="finite"):
        replace(frame(0), raw_robot_state=np.full(8, np.nan))
    with pytest.raises(ValueError, match="future"):
        ManagedMemoryPacker(SMALL).pack([MemorySlot(frame(1), "history", "exec")], max_frame_id=0, decision_id=0)


def test_approximation_can_be_disabled():
    with pytest.raises(ValueError, match="exact recent"):
        stream(3000, cfg=replace(SMALL, recent_overflow="error"))


@pytest.mark.parametrize("mode", ["fifo", "random"])
@pytest.mark.parametrize("boundary", [1000, 2000])
def test_long_demonstration_does_not_strand_phase_slots(mode, boundary):
    c = stream(4200, mode=mode, boundary=boundary)
    assert c.stats()["bank_frames"] == 32
    assert c.stats()["phase_slot_counts"] == phase_counts(partition(4200).uniform, boundary)
    assert c.stats()["recent_window_approx"]
    assert c.stats()["retained_unique_frames"] <= 192


def test_failed_recent_refresh_restores_rng_and_availability():
    c = stream(100, mode="random")
    bank = c.bank
    before = bank.snapshot()
    rng = bank.rng.bit_generator.state
    with pytest.raises(ValueError, match="noncausal"):
        bank.refresh_recent(100, c.rolling, decision_id=-1)
    assert bank.snapshot() is before and bank.decision_id == 0
    assert bank.rng.bit_generator.state == rng


def test_ingest_rejects_invalid_feature_shape_before_mutation():
    c = stream(100)
    before = c.rolling.snapshot()
    with pytest.raises(ValueError, match="tensor contract"):
        c.ingest(frame(101, cfg=replace(SMALL, image_dim=9)))
    assert c.rolling.snapshot() == before


def test_policy_adapter_treats_later_zero_boundary_as_unchanged():
    class FakePolicy:
        _seed = 7
        config = type("Config", (), {"representation_type": "perceptual",
            "perceptual_memory": type("P", (), {"type": "frame_sampling"})(),
            "num_views": 1, "budget": 512, "token_per_image": 16})()
        step_idx = -1
        exec_start_idx = 0
        def _normalize_state(self, x): return x
        def _vision_encode(self, x): raise AssertionError("not reached")
        def reset(self): self.step_idx, self.exec_start_idx = -1, 0
        def add_buffer(self, obs): self.step_idx += len(obs["images"])
        def _prepare_history(self, x): return x
        def _prepare_mem_buffer(self): pass
    # Avoid vision initialization: exercise the boundary routing with a fake
    # original add_buffer and a prebuilt controller.
    p = FakePolicy()
    adapter = object.__new__(__import__("src.memory_management.policy_adapter", fromlist=["ManagedPolicyAdapter"]).ManagedPolicyAdapter)
    adapter.policy, adapter.mode = p, "fifo"
    adapter.config = __import__("src.memory_management.config", fromlist=["MemoryConfig"]).MemoryConfig(image_dim=1, position_dim=6)
    adapter.context, adapter.emit, adapter.episode_number = lambda: {}, lambda _: None, 0
    adapter.original = {"reset": p.reset, "add_buffer": p.add_buffer}
    from src.memory_management.online_controller import OnlineMemoryController
    p.mem_buffer = type("B", (), {"controller": OnlineMemoryController(adapter.config, "fifo", 7)})()
    adapter.initialized = False
    adapter.add_buffer({"images": [], "exec_start_idx": 168})
    assert p.exec_start_idx == 168
    adapter.add_buffer({"images": [], "exec_start_idx": 0})
    assert p.exec_start_idx == 168
