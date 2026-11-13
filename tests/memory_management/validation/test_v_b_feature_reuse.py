"""Encoded frame reuse must preserve the old protocol and reject changed inputs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ECM = Path(__file__).resolve().parents[3]


def test_frame_reuse_checks_provenance_without_importing_loss_labels(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ECM / "scripts"))
    spec = importlib.util.spec_from_file_location("v_b_runner_reuse_test", ECM / "scripts/run_v_b_teacher_probe.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    old = {
        "checkpoint_path": "/test/checkpoint", "g0b_sha256": "gate", "primary_dtype": "float32",
        "checkpoint_restore_dtype": "bfloat16", "matmul_precision": "highest",
        "code_sha256": {runner.__file__: "old-runner", "encoder.py": "unchanged"},
        "config": {"runtime": {"model_seed": 7, "encode_batch_size": 8, "feature_cache": str(tmp_path / "features")}},
        "manifest_provenance": {"VideoUnmaskSwap/20": {"h5_sha256": "h5", "manifest_sha256": "manifest"}},
    }
    source = tmp_path / "source"
    source.mkdir()
    source_path = source / "protocol.json"
    source_path.write_text(json.dumps(old))
    namespace = runner.feature_namespace(old)
    cache = tmp_path / "features" / namespace / "VideoUnmaskSwap" / "ep20.npz"
    cache.parent.mkdir(parents=True)
    metadata = {"h5_sha256": "h5", "namespace": namespace}
    arrays = {k: np.ones((2, 1, 2), dtype=np.float32) for k in ("image", "position", "state")}
    np.savez(cache, metadata=json.dumps(metadata), **arrays)
    new = json.loads(json.dumps(old))
    new["code_sha256"][runner.__file__] = "fixed-runner"
    imports = runner.seal_feature_imports(source, new)
    assert imports["VideoUnmaskSwap/20"]["source_protocol_sha256"] == runner.sha256(source_path)
    new["feature_cache_imports"] = imports
    features = runner.episode_features(None, {"manifest": {"h5_sha256": "h5"}, "joint_action": np.zeros((2, 8))},
                                       "VideoUnmaskSwap", 20, new)
    np.testing.assert_array_equal(features["image"], arrays["image"])
    changed = json.loads(json.dumps(new))
    changed["code_sha256"]["encoder.py"] = "modified"
    with pytest.raises(ValueError, match="unchanged encoder/model"):
        runner.seal_feature_imports(source, changed)
    changed = json.loads(json.dumps(new))
    changed["manifest_provenance"]["VideoUnmaskSwap/20"]["h5_sha256"] = "different-data"
    with pytest.raises(ValueError, match="incompatible demo"):
        runner.seal_feature_imports(source, changed)
    cache.write_bytes(b"changed cached features")
    with pytest.raises(ValueError, match="imported feature cache"):
        runner.episode_features(None, {}, "VideoUnmaskSwap", 20, new)
