#!/usr/bin/env python3
"""Checkpoint-backed baseline flag/action regression and real-feature packer equivalence."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np

from v_a_common import CHECKPOINT, ECM, POLICY, atomic_json, read_json, read_jsonl, sha256, timestamp
sys.path.insert(0, str(ECM / "src"))
from memory_management.config import MemoryConfig
from memory_management.types import FrameRecord, MemorySlot
from memory_management.packer import ManagedMemoryPacker
from memory_management.policy_adapter import install_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=ECM / "results/V_A_formal_20261023/episodes/D0a/VideoUnmaskSwap/ep00_seed7")
    args = parser.parse_args()
    import jax
    import flax
    from mme_vla_suite.policies import policy_config
    from mme_vla_suite.training import config
    cfg = MemoryConfig()
    args.out.mkdir(parents=True, exist_ok=True)
    first = read_jsonl(args.fixture / "client_decisions.jsonl")[0]
    F = first["available_max_frame_id"]
    raw = read_jsonl(args.fixture / "raw_frames.jsonl")[:F + 1]
    fronts, states = [], []
    for shard in sorted(set(r["shard"] for r in raw)):
        with np.load(args.fixture / shard, allow_pickle=False) as data:
            for row in (r for r in raw if r["shard"] == shard):
                fronts.append(data["front"][row["shard_offset"]])
                states.append(data["state"][row["shard_offset"]])
    with np.load(args.fixture / "observations/decision_0000.npz", allow_pickle=False) as data:
        obs = {k: data[k].copy() for k in data.files}
    obs["prompt"] = str(obs["prompt"].item())
    print(f"Loading checkpoint for baseline regression; fixture cutoff={F}", flush=True)
    policy = policy_config.create_trained_policy(config.get_config("mme_vla_suite"), CHECKPOINT, seed=7)
    print(f"Encoding {len(fronts)} original fixture frames", flush=True)
    policy.add_buffer({"images": np.stack(fronts)[:, None], "state": np.stack(states),
                       "exec_start_idx": raw[0]["exec_start_idx"]})
    reference_packed = policy._prepare_history(dict(obs))
    policy._rng = jax.random.key(7)
    print("Running baseline reference actions", flush=True)
    reference = policy.infer(dict(obs))["actions"]
    before_methods = {k: getattr(policy, k) for k in ("reset", "add_buffer", "_prepare_history", "infer")}
    assert install_memory(policy, "baseline_framesamp") is None
    assert all(getattr(policy, k) == v for k, v in before_methods.items())
    policy._rng = jax.random.key(7)
    print("Running baseline flag action regression", flush=True)
    repeated = policy.infer(dict(obs))["actions"]
    baseline_equal = np.array_equal(reference, repeated)
    if not baseline_equal:
        raise RuntimeError(f"baseline flag/action regression failed: max abs={np.max(abs(reference - repeated))}")
    ids = policy.mem_buffer.get_frame_sampling_indices(F, 512, 16)
    slots = []
    for i in ids:
        f = policy.mem_buffer._history_feats[i]
        record = FrameRecord("real-baseline-fixture", i, 0, None,
                             f["image_emb_4x4"], f["pos_emb_4x4"], f["state_emb"])
        slots.append(MemorySlot(record, "history", "demo" if i < policy.exec_start_idx else "exec"))
    packed = ManagedMemoryPacker(cfg, policy._normalize_state).pack(slots, max_frame_id=F, decision_id=0)
    errors = {}
    for key in packed:
        expected = np.asarray(reference_packed[key])
        if key == "static_mask":
            assert np.array_equal(expected, packed[key])
        else:
            np.testing.assert_allclose(packed[key], expected.astype(np.float32), rtol=1e-6, atol=2e-7)
        errors[key] = float(np.max(np.abs(packed[key].astype(float) - expected.astype(float))))
    np.savez_compressed(args.out / "baseline_fixture.npz", **packed, actions=reference,
                        selected_frame_ids=np.asarray(ids), frame_cutoff=F)
    manifest = {"schema": "m1-baseline-regression-v1", "created_utc": timestamp(), "passed": True,
                "checkpoint": str(CHECKPOINT), "checkpoint_identity_source": str(ECM.parent / "acm_vla/artifacts/checkpoint_identity.json"),
                "checkpoint_identity": read_json(ECM.parent / "acm_vla/artifacts/checkpoint_identity.json"),
                "python": sys.version, "jax": jax.__version__, "flax": flax.__version__, "devices": list(map(str, jax.devices())),
                "action_horizon": policy._model.action_horizon, "model_action_dim": policy._model.action_dim,
                "returned_action_shape": list(reference.shape), "state_dim": 8,
                "joint_indices": list(range(7)), "gripper_index": 7,
                "state_semantics_source": str(POLICY / "examples/robomme/env_runner.py"),
                "state_semantics_source_sha256": sha256(POLICY / "examples/robomme/env_runner.py"),
                "execution_horizon": 16, "input_fps": None, "input_fps_source": "see each live episode timebase.json; no guessed capture times",
                "baseline_flag_noop": True, "actions_bitwise_equal": baseline_equal,
                "action_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                "packer_max_abs_error": errors, "fixture_source": str(args.fixture), "fixture_cutoff": F,
                "fixture_sha256": sha256(args.out / "baseline_fixture.npz"),
                "dtype_policy": "managed tensors float32; compare upstream values after float32 conversion, rtol1e-6/atol2e-7",
                "scope": "baseline switch leaves actions unchanged; does not assert managed FIFO/random actions equal baseline"}
    atomic_json(args.out / "baseline_manifest.json", manifest)
    print(f"Checkpoint regression passed: {args.out / 'baseline_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
