#!/usr/bin/env python3
"""Independently validate the completed D1 artifacts and stop at G0 for review."""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np

from run_fruit_exec_guard import CONDITIONS, TASKS
from v_a_common import ECM, atomic_json, read_json, read_jsonl, sha256, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ECM / "results/fruit_exec_guard_dev_20261102")
    out = parser.parse_args().out.resolve()
    status, plan, summary = (read_json(out / name) for name in ("status.json", "plan.json", "summary.json"))
    if status["state"] != "awaiting_G0_confirmation":
        raise SystemExit("Stage 1 is not complete; refusing finalization")
    assert sha256(ECM / "reports/split_registry_20261102.json") == plan["split_registry_sha256"]
    assert sha256(ECM / "reports/fruit_joint_guard_v1_20261102.json") == plan["guard_stats_sha256"]
    for path, digest in plan["source_sha256"].items():
        assert sha256(Path(path)) == digest, f"source changed during experiments: {path}"
    sessions, decisions, engineering_aborts = [], 0, 0
    output_shapes = set()
    for task in TASKS:
        for e in range(200, 220):
            group = []
            for c, (guard, horizon) in CONDITIONS.items():
                d = out / c / task / f"ep{e}"
                result, audit = read_json(d / "result.json"), read_json(d / "memory_audit.json")
                known_abort = bool(plan.get("continue_guard_infeasible") and result.get("joint_guard_infeasible"))
                assert known_abort or (not result["error"] and result["outcome"] in ("success", "fail", "timeout"))
                engineering_aborts += int(known_abort)
                assert result["exec_horizon"] == horizon and result["obs_horizon"] == 16 and result["joint_guard"] == guard
                assert audit["passed"] and audit["mismatches"] == audit["formal_selection_audit"]["mismatches"] == 0
                decisions += audit["decisions"]
                group.append(result["server_session"])
                output_shapes.update(tuple(r["action_shape"]) for r in read_jsonl(d / "policy_actions.jsonl"))
            assert len(set(group)) == 1, f"unpaired sessions: {task}/{e}"
            sessions.append({"task": task, "episode": e, "server_session": group[0]})
    for c in CONDITIONS:
        data = summary["conditions"][c]
        assert data["all"]["episodes"] == 40
        assert all(data["by_task"][t]["episodes"] == 20 for t in TASKS)
        assert not data["missing_expected_episodes"]
        for failure in data["failed_runs"]:
            assert plan.get("continue_guard_infeasible") and read_json(Path(failure["directory"])/"result.json").get("joint_guard_infeasible")
    assert len(sessions) == 40 and all(a["passed"] for a in summary["memory_smokes"])
    from memory_management.fruit_execution import load_joint_guard
    limits = load_joint_guard(ECM / "reports/fruit_joint_guard_v1_20261102.json")
    controls = 0
    for label in ("smoke_A2", "smoke_A3"):
        for task in TASKS:
            for row in read_jsonl(out / label / task / "ep200/control_actions.jsonl"):
                a, p, q = (np.asarray(row[k]) for k in ("action", "proposed_action", "measured_arm_qpos"))
                assert np.all(a[:7] >= limits["lower"]-1e-12) and np.all(a[:7] <= limits["upper"]+1e-12)
                assert np.all(np.abs(a[:7]-q) <= limits["v_max"]+1e-12) and a[7] == p[7]
                assert row["guard_clipped_joints"] == (a[:7] != p[:7]).tolist()
                controls += 1
    best = summary["conditions"][summary["best_condition"]]["all"]
    assert summary["G0_thresholds_met"] == (best["first_target_in_basket"]["numerator"] >= 20 and
                                             best["success"]["numerator"] >= 10)
    for name in ("A0.png", "best.png"):
        assert (out / "frames" / name).stat().st_size > 0
    validation = {"validated_utc": timestamp(), "episodes": 200, "paired_layouts": 40,
                  "main_decisions_audited": decisions, "main_audit_mismatches": 0,
                  "smoke_decisions_audited": sum(a["decisions"] for a in summary["memory_smokes"]),
                  "smoke_audit_mismatches": 0, "smoke_controls_validated": controls,
                  "frozen_split_unchanged": True, "runtime_sources_unchanged": True,
                  "model_action_shapes": sorted(output_shapes), "actual_execution_horizons": [16, 8, 4]}
    validation["guard_infeasible_engineering_aborts"] = engineering_aborts
    summary["independent_validation"] = validation
    # Correct only report wording after the live runner has completed; do not
    # mutate any runtime source or rewrite the sealed per-episode evidence.
    report = out / "report.md"
    text = report.read_text()
    text = text.replace("旧 policy_actions.execution_horizon 字段仍为完整预测块长度 16，真实执行长度见 result.json 的 exec_horizon。",
                        "实测模型返回 20×8 动作块，基线执行前 16 步、干预执行前 8/4 步。旧 policy_actions.execution_horizon 字段被硬编码为 16，不能当作模型输出长度或本次实际执行长度；真实执行长度见 result.json 的 exec_horizon 和决策帧间隔。")
    text += ("\n\n## 独立复核\n\n"
             f"200/200 个主实验回合完整，40/40 个布局五条件同一服务会话；"
             f"主实验 {decisions} 个决策、冒烟 {validation['smoke_decisions_audited']} 个决策，审计不一致均为 0。"
             f"独立验证 {controls} 个冒烟控制步，训练包络、实测状态限速、夹爪保留及裁剪标记均一致。"
             "冻结划分及运行源文件哈希保持不变，详见 independent_validation.json。"
             f"包络不可达而拒绝执行的工程失败为 {engineering_aborts}/200，保留在全部指标中，未计作任务 fail。"
             "本复核仅修正文档中旧执行长度日志字段的说明，没有改动作、轨迹、模型、指标或运行源文件。\n")
    report.write_text(text)
    atomic_json(out / "independent_validation.json", validation)
    atomic_json(out / "summary.json", summary)
    tracked = ECM / "reports/lowlevel_rescue_D1_20261102"
    tracked.mkdir(exist_ok=True)
    atomic_json(tracked / "summary.json", summary)
    (ECM / "reports/lowlevel_rescue_D1_20261102.md").write_text(
        text.replace("](frames/", f"](../results/{out.name}/frames/") +
        f"\n运行原报告及所有相对数据路径的基目录：`results/{out.name}`。\n")
    root_status = read_json(ECM / "reports/lowlevel_rescue_status.json")
    root_status.update(phase=1, state="awaiting_G0_confirmation", updated_utc=timestamp())
    root_status["links"].update(D1_versioned_report="reports/lowlevel_rescue_D1_20261102.md",
                                D1_versioned_summary="reports/lowlevel_rescue_D1_20261102/summary.json",
                                D1_validation=str(out / "independent_validation.json"))
    atomic_json(ECM / "reports/lowlevel_rescue_status.json", root_status)
    print(validation, flush=True)
    print({"best": summary["best_condition"], "G0_thresholds_met": summary["G0_thresholds_met"],
           "state": "awaiting_G0_confirmation"}, flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ECM / "src"))
    main()
