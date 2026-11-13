#!/usr/bin/env python3
"""Finish formal V-A: sealed primary -> audit/analysis -> one seed extension -> joint routing.

python3 scripts/complete_v_a.py --run results/V_A_formal_20261023 --prepare-only
python3 -u scripts/complete_v_a.py --run results/V_A_formal_20261023

Rerun the same command after interruption. Durable episode attempts are never retried.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace

import yaml

from v_a_common import (CHECKPOINT, ECM, POLICY, POLICY_PYTHON, SIM, SIM_PYTHON,
                        atomic_json, read_json, sha256, timestamp)
from run_v_a_closed_loop import controller, done_keys, seal, verify

DEFAULT_CONFIG = ECM / "configs/memory_management/validation/v_a_closed_loop.yaml"
DEFAULT_VB = ECM / "results/V_B_fp32_full_20261022"


def preflight(config):
    cfg = yaml.safe_load(config.read_text())["v_a"]
    assert cfg["split"] == "val" and cfg["execution_horizon"] == 16 and cfg["max_steps"] == 1300
    assert cfg["memory"] == {"budget_tokens": 512, "slots": 32, "tokens_per_frame": 16, "views": ["front"]}
    assert cfg["protection"] == {"anchor_slots": 1, "recent_d0_slots": 4}
    assert cfg["model_seeds"] == [7, 8, 9] and cfg["extension_seeds"] == [10, 11]
    for path in (POLICY_PYTHON, SIM_PYTHON, CHECKPOINT / "params", CHECKPOINT / "assets",
                 Path("/etc/vulkan/icd.d/nvidia_icd.json")):
        if not path.exists():
            raise RuntimeError(f"missing runtime asset: {path}")
    ids = list(range(cfg["episode_ids"]["start"], cfg["episode_ids"]["stop"]))
    metadata = {}
    for task in cfg["tasks"]:
        path = SIM / f"src/robomme/env_metadata/val/record_dataset_{task}_metadata.json"
        records = read_json(path)["records"]
        own = {r["episode"]: r for r in records if r["task"] == task}
        assert all(i in own for i in ids), f"missing val IDs for {task}"
        metadata[task] = {str(i): {k: own[i][k] for k in ("seed", "difficulty")} for i in ids}
    return {"checked_utc": timestamp(), "status": "static_ready", "val_metadata": metadata,
            "disk_free_bytes": shutil.disk_usage(ECM).free,
            "planned_primary": len(cfg["tasks"]) * len(ids) * len(cfg["model_seeds"]) * len(cfg["conditions"]),
            "cuda_checked": False}


def runtime_preflight(run):
    protocol = verify(run)
    configs = [read_json(p) for p in run.glob("server*/config.json")]
    for cfg in configs:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", cfg["port"]))
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader"],
                         text=True, capture_output=True, timeout=30)
    if smi.returncode:
        raise RuntimeError(f"NVIDIA driver unavailable; run this entry point in the GPU terminal:\n{smi.stderr}")
    gpus = protocol["config"]["runtime"]["gpus"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "JAX_PLATFORMS": "cuda",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    probe = subprocess.run([str(POLICY_PYTHON), "-c",
                            "import jax; d=jax.devices(); print(d); "
                            f"assert len(d)=={len(gpus)} and all(x.platform=='gpu' for x in d)"],
                           cwd=POLICY, env=env, text=True, capture_output=True, timeout=60)
    if probe.returncode:
        raise RuntimeError(f"CUDA/JAX preflight failed:\n{probe.stdout}\n{probe.stderr}")
    atomic_json(run / "runtime_preflight.json", {"checked_utc": timestamp(), "cuda_checked": True,
                                                "nvidia_smi": smi.stdout, "jax": probe.stdout})
    print(smi.stdout + probe.stdout, flush=True)


def prepare(run, config, parent=None, conditions=None):
    cfg = yaml.safe_load(config.read_text())["v_a"]
    return seal(SimpleNamespace(config=config, tasks=None, episode_ids=None,
                seeds=cfg["extension_seeds"] if parent else None, conditions=conditions,
                servers_per_seed=None, tag="formal", out=run, formal=True,
                stage="extension" if parent else "primary", parent_run=parent))


def analyze(runs):
    subprocess.run([sys.executable, "-u", str(ECM / "scripts/analyze_v_a.py"), *map(str, runs)],
                   cwd=ECM, check=True)
    return read_json(runs[0] / "analysis.json")


def execute(run):
    protocol = verify(run)
    if len(done_keys(run)[1]) < protocol["planned_episodes"]:
        runtime_preflight(run)
        print(f"Starting {run} ({protocol['planned_episodes']} planned episodes)", flush=True)
        if controller(run):
            raise RuntimeError(f"V-A stopped before completion; inspect {run}/progress.json and controller_error.json")


def joint_decision(va, vb):
    if not va["validity"]["valid"]:
        return {"status": "invalid_v_a", "next_step": "先排查 V-A 有效性门槛，不能进入价值预测器开发。"}
    if not vb.get("formal") or vb["completed_boundaries"] != vb["planned_boundaries"]:
        return {"status": "invalid_v_b", "next_step": "先完成 V-B 有效性核验。"}
    combined = vb["combined"]
    passing = [int(k) for k, v in combined.items() if v["P2_passed"] and v["P3_passed"]]
    if not passing:
        reliable = any(v["P2_passed"] for v in combined.values())
        return {"status": "retention_research" if reliable else "teacher_review_required",
                "next_step": "研究何时保留原记忆与拒绝替换。" if reliable else "按预注册检查标签可靠性复测要求。"}
    k = min(passing)
    if va["v_a_result"] == "无明显退化":
        return {"status": "go_value_predictor" if k == 1 else "go_set_value_predictor", "minimum_k": k,
                "next_step": "进入 Milestone 1–5 价值预测器开发。" if k == 1 else "修订为集合级价值学习后进入 Milestone 1。"}
    return {"status": "analyze_distribution", "minimum_k": k,
            "next_step": "先按 §0A.6 分析 C3/C4、删除槽与事件证据；未能判定时保守处理，不直接训练策略。"}


def finish(run, config, vb_run):
    binding = read_json(run / "v_b_binding.json")
    assert str(vb_run) == binding["run"], "V-B run differs from prepared binding"
    for name, digest in binding["sha256"].items():
        assert sha256(vb_run / name) == digest, f"bound V-B artifact changed: {name}"
    state = {"status": "running_primary", "updated_utc": timestamp(), "primary_run": str(run)}
    atomic_json(run / "pipeline_state.json", state)
    execute(run)
    decision_path = run / "extension_decision.json"
    if decision_path.exists():
        decision = read_json(decision_path)
        assert sha256(run / "primary_analysis.json") == decision["primary_analysis_sha256"]
        primary = read_json(run / "primary_analysis.json")
    else:
        primary = analyze([run])
        if not primary["validity"]["valid"]:
            state.update(status="invalid_primary", updated_utc=timestamp(), validity=primary["validity"])
            atomic_json(run / "pipeline_state.json", state)
            return 2
        shutil.copyfile(run / "analysis.json", run / "primary_analysis.json")
        shutil.copyfile(run / "REPORT.md", run / "PRIMARY_REPORT.md")
        decision = {"created_utc": timestamp(), "primary_analysis_sha256": sha256(run / "primary_analysis.json"),
                    "conditions": primary["preregistered_extension_conditions"],
                    "extension_seeds": read_json(run / "protocol.json")["config"]["extension_seeds"],
                    "extension_run": str(run.with_name(run.name + "_extension")), "max_extensions": 1}
        atomic_json(decision_path, decision)
    runs = [run]
    if decision["conditions"]:
        extension = Path(decision["extension_run"])
        state.update(status="running_extension", extension_run=str(extension), updated_utc=timestamp())
        atomic_json(run / "pipeline_state.json", state)
        if not extension.exists():
            prepare(extension, config, parent=run, conditions=decision["conditions"])
        execute(extension)
        runs.append(extension)
    final = analyze(runs) if len(runs) > 1 else primary
    joint = {"generated_utc": timestamp(), "v_a_result": final["v_a_result"],
             "v_a_analysis_sha256": sha256(run / "analysis.json"), "v_b_binding": binding,
             **joint_decision(final, read_json(vb_run / "analysis.json"))}
    atomic_json(run / "joint_decision.json", joint)
    (run / "JOINT_REPORT.md").write_text(
        f"# V-A / V-B 联合决策\n\nV-A：{final['v_a_result']}；有效性：{final['validity']['valid']}。\n\n"
        f"路由：`{joint['status']}`。{joint['next_step']}\n\n"
        "V-A 详细证据见 REPORT.md、analysis.json、selection_audit.json；初轮判定见 PRIMARY_REPORT.md。\n"
        "V-B FP32 loss 探针与 V-A 官方 BF16 策略推理是不同验证环节；不能把 loss 收益写成成功率提升。\n")
    state.update(status="completed" if final["validity"]["valid"] else "invalid_final",
                 updated_utc=timestamp(), joint_decision=joint["status"], completed_runs=list(map(str, runs)))
    atomic_json(run / "pipeline_state.json", state)
    print(json.dumps(joint, ensure_ascii=False, indent=2), flush=True)
    return 0 if final["validity"]["valid"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--vb-run", type=Path, default=DEFAULT_VB)
    parser.add_argument("--prepare-only", action="store_true")
    opts = parser.parse_args()
    run, config, vb_run = opts.run.resolve(), opts.config.resolve(), opts.vb_run.resolve()
    static = preflight(config)
    if not run.exists():
        vb = read_json(vb_run / "analysis.json")
        assert vb["formal"] and vb["completed_boundaries"] == vb["planned_boundaries"]
        prepare(run, config)
        atomic_json(run / "preflight.json", static)
        atomic_json(run / "v_b_binding.json", {"run": str(vb_run), "sha256": {
                    name: sha256(vb_run / name) for name in ("protocol.json", "analysis.json", "REPORT.md")}})
    protocol = verify(run)
    assert protocol.get("formal") and protocol["experiment_stage"] == "primary"
    assert sha256(config) == protocol["config_sha256"], "configuration changed since preparation"
    print(f"Formal primary prepared: {run} ({protocol['planned_episodes']} episodes)", flush=True)
    if opts.prepare_only:
        return 0
    with (run / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another V-A pipeline already owns {run}") from None
        try:
            return finish(run, config, vb_run)
        except Exception as exc:
            state = read_json(run / "pipeline_state.json") if (run / "pipeline_state.json").exists() else {}
            state.update(status="error", error=str(exc), updated_utc=timestamp())
            atomic_json(run / "pipeline_state.json", state)
            raise


if __name__ == "__main__":
    sys.exit(main())
