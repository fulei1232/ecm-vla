"""Exercise the actual acceptance auditor, including corrupted and incomplete evidence."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ECM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ECM / "scripts"))
spec = importlib.util.spec_from_file_location("m1_acceptance", ECM / "scripts/run_milestone1_acceptance.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def recorded(tmp_path):
    memory = runner.MemoryConfig()
    small = runner.MemoryConfig(image_dim=1, position_dim=6)
    c = runner.OnlineMemoryController(small, "random", 7)
    c.reset("live", 80)
    logs, ingests, client = [], [], []
    next_frame = 0
    for index, F in enumerate((100, 116, 132, 148, 300)):
        for f in range(next_frame, F + 1):
            c.ingest(runner.FrameRecord("live", f, c.decision_id, None, np.zeros((1, 16, 1), np.float32),
                                       np.zeros((1, 16, 6), np.float32), np.zeros(8, np.float32)))
        c.prepare()
        st = c.stats()
        shapes = {"static_image_emb": [512, 2048], "static_pos_emb": [512, 768],
                  "static_state_emb": [512, 8], "static_mask": [512]}
        logs.append({**st, "mode": "random", "exec_start_idx": 80, "decision_idx": index,
                     "available_max_frame_id": F, "source_position_frame_ids": st["selected_frame_ids"],
                     "valid_tokens": 512, "tensor_shapes": shapes,
                     "tensor_dtypes": {k: "bool" if k == "static_mask" else "float32" for k in shapes}})
        ingests.append({"first_frame_id": next_frame, "last_frame_id": F, "samples": F - next_frame + 1,
                        "decision_idx": index})
        client.append({"decision_idx": index, "available_max_frame_id": F})
        next_frame = F + 1
    write_lines(tmp_path / "policy_decisions.jsonl", logs)
    write_lines(tmp_path / "client_ingest.jsonl", ingests)
    write_lines(tmp_path / "client_decisions.jsonl", client)
    runner.atomic_json(tmp_path / "timebase.json", {"control_freq_hz": 20, "sim_freq_hz": 100})
    row = {"attempt_directory": str(tmp_path), "outcome": "timeout", "model_seed": 7,
           "condition": "random", "exec_start_idx": 80}
    return row, runner.asdict(memory), logs, ingests


def test_actual_auditor_replays_selections_and_accepts_valid_task_timeout(recorded):
    row, memory, _, _ = recorded
    audit = runner.audit_episode(row, memory)
    assert audit["passed"] and audit["decisions"] == 5 and audit["accepted_updates"] > 0


@pytest.mark.parametrize("field,value", [("selected_frame_ids", [0]), ("valid_tokens", 511),
    ("tensor_dtypes", {}), ("mode", "fifo"), ("source_position_frame_ids", [999]),
    ("retained_unique_frames", 193), ("phase_slot_counts", {"demo": 0, "exec": 32})])
def test_auditor_rejects_corrupted_live_evidence(recorded, field, value):
    row, memory, logs, _ = recorded
    logs[-1][field] = value
    write_lines(Path(row["attempt_directory"]) / "policy_decisions.jsonl", logs)
    with pytest.raises(AssertionError):
        runner.audit_episode(row, memory)


def test_auditor_rejects_missing_client_decision(recorded):
    row, memory, _, _ = recorded
    (Path(row["attempt_directory"]) / "client_decisions.jsonl").unlink()
    assert not runner.audit_episode(row, memory)["passed"]


def test_auditor_rejects_engineering_error(recorded):
    row, memory, _, _ = recorded
    assert not runner.audit_episode({**row, "outcome": "process_error"}, memory)["passed"]


def test_cpu_evidence_is_bound_to_protocol_and_artifacts(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    artifact = tmp_path / "test.log"
    artifact.write_text("passed")
    runner.atomic_json(tmp_path / "cpu_tests.json", {"passed": True,
        "protocol_sha256": runner.sha256(protocol),
        "artifacts": [{"path": artifact.name, "sha256": runner.sha256(artifact)}]})
    assert runner.cpu_evidence(tmp_path)
    artifact.write_text("changed")
    assert not runner.cpu_evidence(tmp_path)


def test_analyzer_does_not_accept_cpu_only_run(tmp_path, monkeypatch):
    protocol = {"config": {"memory": runner.asdict(runner.MemoryConfig()),
                           "acceptance": {"modes": ["fifo", "random"]}}, "planned_episodes": 6}
    monkeypatch.setattr(runner, "verify", lambda _: protocol)
    runner.atomic_json(tmp_path / "plan.json", [])
    runner.atomic_json(tmp_path / "cpu_memory_stress.json", {"passed": True})
    result = runner.analyze(tmp_path)
    assert not result["passed"] and not result["baseline_regression_passed"]
    assert result["completed"] == 0 and result["planned"] == 6
