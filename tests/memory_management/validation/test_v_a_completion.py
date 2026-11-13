"""V-A integrity gates, paired scope, and one-shot extension lifecycle."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ECM = Path(__file__).resolve().parents[3]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ECM / "scripts"))
    return tuple(importlib.import_module(n) for n in ("run_v_a_closed_loop", "analyze_v_a", "complete_v_a"))


def test_missing_log_cannot_pass_selection_gate(tmp_path, modules):
    _, analysis, _ = modules
    _, failures = analysis.audit_episode({"attempt_directory": str(tmp_path)})
    assert failures[0]["check"] == "missing_policy_decisions"


def test_degradation_requires_negative_95ci_and_holm(modules):
    _, analysis, _ = modules
    base = {"delta": -.03, "ci90": [-.06, -.001], "ci95": [-.08, .001], "p_holm": .04}
    assert not analysis.judge(base, .12)["degraded"]
    assert analysis.judge({**base, "ci95": [-.08, -.001]}, .12)["degraded"]
    assert not analysis.judge({**base, "ci95": [-.08, -.001], "p_holm": .06}, .12)["degraded"]


def test_empty_scope_and_pairwise_errors_do_not_pool_other_tasks(modules):
    _, analysis, _ = modules
    outcomes = {("PatternLock", 0, 7, "D4B"): "success", ("PatternLock", 0, 7, "RNDC"): "fail",
                ("PatternLock", 1, 7, "D4B"): "fail"}  # unpaired ID must not enter estimate
    boot = analysis.PairedBootstrap(outcomes, ["PatternLock"], 100)
    assert boot.contrast("D4B", "RNDC", []) is None
    result = boot.contrast("D4B", "RNDC")
    assert result["delta"] == 1 and result["per_task"]["PatternLock"]["n_pairs"] == 1


def test_formal_seal_is_complete_and_changed_source_rejected(tmp_path, modules, monkeypatch):
    runner, _, _ = modules
    cfg = yaml.safe_load((ECM / "configs/memory_management/validation/v_a_closed_loop.yaml").read_text())
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    source = tmp_path / "sentinel.py"
    source.write_text("original")
    monkeypatch.setattr(runner, "CODE", [source])
    monkeypatch.setattr(runner, "ECM", tmp_path)
    monkeypatch.setattr(runner, "dependencies", lambda _: [])
    run = runner.seal(SimpleNamespace(config=config, tasks=None, episode_ids=None, seeds=None, conditions=None,
                       servers_per_seed=None, tag="", out=tmp_path / "primary", formal=True,
                       stage="primary", parent_run=None))
    p = runner.verify(run)
    assert p["formal"] and p["planned_episodes"] == 1260
    plan = json.loads((run / "plan.json").read_text())
    assert len({(r["task"], r["episode_id"], r["model_seed"], r["condition"]) for r in plan}) == 1260
    source.write_text("changed")
    with pytest.raises(AssertionError, match="code changed"):
        runner.verify(run)


def test_extension_once_and_resume_does_not_reselect_conditions(tmp_path, modules, monkeypatch):
    _, _, pipeline = modules
    run, vb_run = tmp_path / "primary", tmp_path / "vb"
    run.mkdir(); vb_run.mkdir()
    config = tmp_path / "config.yaml"
    cfg = yaml.safe_load(pipeline.DEFAULT_CONFIG.read_text())
    config.write_text(yaml.safe_dump(cfg))
    pipeline.atomic_json(run / "protocol.json", {"config": cfg["v_a"]})
    vb = {"formal": True, "completed_boundaries": 111, "planned_boundaries": 111,
          "combined": {"1": {"P2_passed": True, "P3_passed": True}}}
    pipeline.atomic_json(vb_run / "analysis.json", vb)
    (vb_run / "protocol.json").write_text("{}")
    (vb_run / "REPORT.md").write_text("bound report")
    pipeline.atomic_json(run / "v_b_binding.json", {"run": str(vb_run), "sha256": {
        n: pipeline.sha256(vb_run / n) for n in ("analysis.json", "protocol.json", "REPORT.md")}})
    primary = {"validity": {"valid": True}, "preregistered_extension_conditions": ["D0a", "D4B"],
               "v_a_result": "未能判定"}
    final = {**primary, "v_a_result": "无明显退化", "preregistered_extension_conditions": []}
    calls = {"prepare": [], "execute": [], "analyze": []}

    def analyze(runs):
        calls["analyze"].append(list(runs))
        result = primary if len(runs) == 1 else final
        pipeline.atomic_json(run / "analysis.json", result)
        (run / "REPORT.md").write_text("analysis report")
        return result

    def prepare(extension, config, parent, conditions):
        calls["prepare"].append((parent, conditions))
        extension.mkdir()

    monkeypatch.setattr(pipeline, "analyze", analyze)
    monkeypatch.setattr(pipeline, "execute", lambda r: calls["execute"].append(r))
    monkeypatch.setattr(pipeline, "prepare", prepare)
    assert pipeline.finish(run, config, vb_run) == 0
    assert len(calls["prepare"]) == 1 and calls["prepare"][0][1] == ["D0a", "D4B"]
    assert pipeline.read_json(run / "joint_decision.json")["status"] == "go_value_predictor"
    assert pipeline.finish(run, config, vb_run) == 0
    assert len(calls["prepare"]) == 1  # only the sealed extension is resumed
    assert sum(len(r) == 1 for r in calls["analyze"]) == 1


def test_invalid_primary_cannot_extend_and_undetermined_uses_conservative_route(tmp_path, modules, monkeypatch):
    _, _, pipeline = modules
    vb = {"formal": True, "completed_boundaries": 111, "planned_boundaries": 111,
          "combined": {"1": {"P2_passed": True, "P3_passed": True}}}
    assert pipeline.joint_decision({"validity": {"valid": False}}, vb)["status"] == "invalid_v_a"
    assert pipeline.joint_decision({"validity": {"valid": True}, "v_a_result": "未能判定"}, vb)["status"] == "analyze_distribution"
