"""Run with the policy venv on CPU: real upstream packer, position formula and ingestion."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("mme_vla_suite.shared.mem_buffer")
from mme_vla_suite.shared.mem_buffer import MemoryBuffer
from mme_vla_suite.shared.posemb_3d import PosEmb3D
from mme_vla_suite.policies.policy import MME_VLA_Policy
from omegaconf import OmegaConf

from src.memory_management.config import MemoryConfig
from src.memory_management.types import FrameRecord, MemorySlot
from src.memory_management.packer import ManagedMemoryPacker
from src.memory_management.policy_adapter import SourcePositionEncoder, install_memory

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT.parent / "acm_vla/vendor/robomme_policy_learning"


@pytest.mark.parametrize("n", [1, 2, 31, 32])
def test_ut03_actual_upstream_packer_equivalence(n):
    rng = np.random.default_rng(19)
    ids = list(range(n))
    records, slots = {}, []
    for i in ids:
        image = np.asarray(jnp.asarray(rng.normal(size=(1, 16, 2048)), dtype=jnp.bfloat16))
        position = rng.normal(size=(1, 16, 768)).astype(np.float32)
        state = rng.normal(size=8).astype(np.float32)
        records[i] = {"image_emb_4x4": image, "pos_emb_4x4": position, "state_emb": state}
        f = FrameRecord("fixture", i, 0, None, image, position, state)
        slots.append(MemorySlot(f, "history", "exec"))
    official = MemoryBuffer(prepare_buffer=False)._prepare_frame_sampling(records, ids, 512, 16)
    # Upstream may promote right padding to float64; model inputs are explicitly float32.
    normalize = lambda x: (x - np.float32(.25)) / np.float32(1.75)
    actual = ManagedMemoryPacker(MemoryConfig(), normalize).pack(slots, max_frame_id=n - 1, decision_id=0)
    for k, expected in zip(actual, official, strict=True):
        if k == "static_state_emb":
            expected = normalize(expected)
        if k == "static_mask":
            np.testing.assert_array_equal(actual[k], expected)
        else:
            np.testing.assert_allclose(actual[k], np.asarray(expected, dtype=np.float32), atol=2e-7, rtol=1e-6)


def test_ut18_source_positions_beyond_4096_match_upstream():
    ids = [0, 1, 31, 1339, 4095, 4096, 10000]
    official = np.asarray(PosEmb3D(768)(jnp.asarray(ids), 4))
    actual = SourcePositionEncoder(768)(ids)
    assert actual.shape == (len(ids), 16, 768)
    np.testing.assert_allclose(actual, official, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["fifo", "random"])
def test_real_policy_methods_use_bounded_buffer_and_reset(mode):
    p = MME_VLA_Policy.__new__(MME_VLA_Policy)
    p.config = OmegaConf.load(POLICY / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml")
    p._seed = 7
    p.use_quantiles = False
    p.state_norm_stats = SimpleNamespace(mean=np.zeros(8, np.float32), std=np.ones(8, np.float32))
    p._prepare_mem_buffer = lambda: setattr(p, "mem_buffer", None)
    p.mem_buffer, p.step_idx, p.exec_start_idx = None, -1, 0
    # Real preprocessing and upstream pooling, deterministic inexpensive replacement for SigLIP only.
    def vision(images):
        return jnp.broadcast_to(images.mean(axis=(-3, -2, -1))[..., None, None], (*images.shape[:2], 64, 2048))
    p._vision_encode = vision
    logs = []
    adapter = install_memory(p, mode, emit=logs.append)
    p.reset()
    for start in range(0, 192, 16):
        p.add_buffer({"images": np.full((16, 1, 16, 16, 3), start, np.uint8),
                      "state": np.zeros((16, 8), np.float32), "exec_start_idx": 40})
        result = p._prepare_history({})
        assert result["static_mask"].sum() == min(start + 16, 32) * 16
        assert len(p.mem_buffer._history_feats) <= 192
    assert logs[-1]["recent_window_approx"] is False
    assert logs[-1]["accepted_updates"] > 0
    assert not hasattr(p.mem_buffer, "pos_emb_dict")
    assert p.step_idx == 191 and p.exec_start_idx == 40
    p.reset()
    assert p.step_idx == -1 and not p.mem_buffer._history_feats
    p.add_buffer({"images": np.zeros((1, 1, 16, 16, 3), np.uint8), "state": np.zeros((1, 8), np.float32)})
    assert p._prepare_history({})["static_mask"].sum() == 16
