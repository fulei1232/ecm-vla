#!/usr/bin/env python3
"""Sealed M1 GPU acceptance: checkpoint regression and six official closed-loop episodes.

python3 scripts/run_milestone1_acceptance.py --run results/M1_acceptance_20261023 --prepare-only
python3 -u scripts/run_milestone1_acceptance.py --run results/M1_acceptance_20261023

The same command resumes unfinished work. Recorded attempts are never silently retried.
"""
import argparse
from dataclasses import asdict
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from v_a_common import (CHECKPOINT, ECM, OPENPI_DATA_HOME, POLICY, POLICY_PYTHON, SIM, SIM_PYTHON,
                        VALID, append_jsonl, atomic_json, read_json, read_jsonl, sha256, timestamp, git_head)
from run_v_a_closed_loop import stop_child, dependencies
sys.path.insert(0, str(ECM / "src"))
from memory_management.config import MemoryConfig
from memory_management.types import FrameRecord
from memory_management.online_controller import OnlineMemoryController
from memory_management.validation.partition import partition, phase_counts

CONFIG = ECM / "configs/memory_management/milestone1.yaml"
CODE = [Path(__file__).resolve(), ECM / "run_milestone1_20261023.sh", *(ECM / "scripts" / name for name in (
    "serve_milestone1_policy.py", "eval_milestone1_episode.py", "verify_milestone1_policy.py",
    "verify_milestone1_memory.py", "eval_v_a_episode.py", "v_a_common.py", "run_v_a_closed_loop.py")),
    *sorted((ECM / "src/memory_management").rglob("*.py")),
    *sorted((ECM / "tests/memory_management").rglob("*.py")), CONFIG]
CHECKPOINT_EVIDENCE = ECM.parent / "acm_vla/artifacts/checkpoint_identity.json"


def prepare(run):
    run.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text())
    memory = MemoryConfig(**cfg["memory"])
    rt = cfg["acceptance"]
    assert rt["split"] == "val" and rt["modes"] == ["fifo", "random"]
    plan = []
    for k, mode in enumerate(rt["modes"]):
        server = run / f"server{k}"
        server.mkdir()
        atomic_json(server / "config.json", {
            "model_seed": rt["seed"], "replica": 0, "mode": mode, "memory": asdict(memory),
            "port": rt["port"], "policy_gpu": rt["policy_gpu"], "sim_gpu": rt["sim_gpu"],
            "checkpoint_path": str(CHECKPOINT), "policy_config": "mme_vla_suite",
            "openpi_data_home": str(OPENPI_DATA_HOME)})
        for task in rt["tasks"]:
            for episode in rt["episode_ids"]:
                plan.append({"task": task, "episode_id": episode, "model_seed": rt["seed"],
                             "condition": mode, "server": k, "block": len(plan)})
    atomic_json(run / "plan.json", plan)
    snapshot = run / "code_snapshot"
    snapshot.mkdir()
    for path in CODE:
        (snapshot / str(path.relative_to(ECM)).replace("/", "__")).write_bytes(path.read_bytes())
    atomic_json(run / "protocol.json", {
        "stage": "Milestone 1 engineering acceptance", "split": "val", "created_utc": timestamp(),
        "git_commit": git_head(ECM),
        "config": cfg, "planned_episodes": len(plan), "plan_sha256": sha256(run / "plan.json"),
        "code_identity": [{"path": str(p), "sha256": sha256(p)} for p in CODE],
        "dependency_identity": [{"path": str(p), "sha256": sha256(p)} for p in [
            *dependencies(rt["tasks"]), CHECKPOINT_EVIDENCE,
            ECM.parent / "acm_vla/logs/20261012T045914_checkpoint_2/run_manifest.json"]],
        "server_config_sha256": [sha256(run / f"server{k}/config.json") for k in range(2)],
        "acceptance": "real checkpoint baseline flag bitwise action regression, packer equivalence, bounded feature stress, and both modes finish official episodes with all decision audits passing",
        "recent_window": "exact when retained; ring160 supports tested cutoffs through1339; explicit nearest approximation beyond available window",
        "attempt_policy": "record all outcomes; no automatic episode retries; this is engineering acceptance, not success-rate evaluation"})
    atomic_json(run / "progress.json", {"status": "prepared", "completed": 0, "planned": len(plan)})


def verify(run):
    p = read_json(run / "protocol.json")
    assert sha256(run / "plan.json") == p["plan_sha256"]
    for row in p["code_identity"] + p["dependency_identity"]:
        assert sha256(row["path"]) == row["sha256"], f"source changed: {row['path']}"
    for k, digest in enumerate(p["server_config_sha256"]):
        assert sha256(run / f"server{k}/config.json") == digest
    return p


def gpu_env(cfg):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["policy_gpu"]), "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONDONTWRITEBYTECODE": "1",
            "OPENPI_DATA_HOME": str(OPENPI_DATA_HOME), "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
            "PYTHONPATH": f"{POLICY}/src:{POLICY}/packages/openpi-client/src:{ECM}/src"}


def cpu_evidence(run):
    path = run / "cpu_tests.json"
    if not path.exists():
        return False
    evidence = read_json(path)
    return (evidence.get("passed", False)
            and evidence.get("protocol_sha256") == sha256(run / "protocol.json")
            and all(sha256(run / r["path"]) == r["sha256"] for r in evidence["artifacts"]))


def run_cpu_checks(run):
    if cpu_evidence(run):
        return
    env = {**gpu_env(read_json(run / "server0/config.json")), "JAX_PLATFORMS": "cpu"}
    groups = [
        ("core_and_regression", sys.executable, ["tests/memory_management/test_milestone1.py",
          "tests/memory_management/test_milestone1_acceptance.py", "tests/memory_management/validation"], True),
        ("official", str(POLICY_PYTHON), ["tests/memory_management/test_milestone1_official.py",
          "tests/memory_management/validation/test_cached_flow_loss_reference.py"], False)]
    artifacts, results = [], []
    for name, python, tests, allow_skip in groups:
        junit = run / f"{name}_tests.xml"
        log_path = run / f"{name}_tests.log"
        print(f"CPU tests: {name}", flush=True)
        with log_path.open("w") as log:
            subprocess.run([python, "-m", "pytest", *tests, "-q", f"--junitxml={junit}"],
                           env=env, cwd=ECM, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
        assert counts["tests"] > 0 and not counts["failures"] and not counts["errors"]
        assert allow_skip or not counts["skipped"], "official upstream checks must execute"
        results.append({"group": name, **counts})
        artifacts.extend({"path": p.name, "sha256": sha256(p)} for p in (junit, log_path))
    subprocess.run([sys.executable, str(ECM / "scripts/verify_milestone1_memory.py"),
                    "--out", str(run / "cpu_memory_stress.json")], check=True, timeout=180)
    artifacts.append({"path": "cpu_memory_stress.json", "sha256": sha256(run / "cpu_memory_stress.json")})
    atomic_json(run / "cpu_tests.json", {"passed": True, "protocol_sha256": sha256(run / "protocol.json"),
                                        "results": results, "artifacts": artifacts, "updated_utc": timestamp()})


def start_server(run, k, rt):
    directory = run / f"server{k}"
    cfg = read_json(directory / "config.json")
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", cfg["port"]))
    log = (directory / f"server_{time.time_ns()}.log").open("x", buffering=1)
    atomic_json(run / "progress.json", {"status": "loading_policy", "mode": cfg["mode"],
                "completed": len(list((run / "episodes").glob("*/*/*/result.json"))),
                "planned": read_json(run / "protocol.json")["planned_episodes"],
                "log": log.name, "updated_utc": timestamp()})
    print(f"Loading {cfg['mode']} policy; log={log.name}", flush=True)
    child = subprocess.Popen([str(POLICY_PYTHON), "-u", str(ECM / "scripts/serve_milestone1_policy.py"),
                              "--config", str(directory / "config.json"), "--context", str(directory / "context.json")],
                             cwd=POLICY, env=gpu_env(cfg), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    manifest = {"server_pid": child.pid, "status": "loading", "server_ready": False,
                "started_utc": timestamp(), "log": log.name,
                **{key: cfg[key] for key in ("model_seed", "port", "checkpoint_path")}}
    atomic_json(directory / "server_manifest.json", manifest)
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + rt["server_load_timeout_s"]
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"policy exited; see {log.name}")
            try:
                with opener.open(f"http://127.0.0.1:{cfg['port']}/healthz", timeout=1) as response:
                    assert response.read() == b"OK\n"
                manifest.update(status="serving", server_ready=True, ready_utc=timestamp())
                atomic_json(directory / "server_manifest.json", manifest)
                return child, log, manifest
            except OSError:
                time.sleep(2)
        raise RuntimeError("policy load timeout")
    except BaseException:
        stop_child(child)
        log.close()
        raise


def episode(run, item, timeout):
    k = item["server"]
    cfg = read_json(run / f"server{k}/config.json")
    d = run / "episodes" / item["condition"] / item["task"] / f"ep{item['episode_id']:02d}_seed{item['model_seed']}"
    result = d / "result.json"
    if result.exists():
        return read_json(result)
    if d.exists():
        row = {**item, "outcome": "process_error", "error": "interrupted before durable result; not retried", "attempt_directory": str(d)}
        atomic_json(result, row)
        return row
    d.mkdir(parents=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["sim_gpu"]), "PYTHONDONTWRITEBYTECODE": "1",
           "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json", "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
           "PYTHONPATH": f"{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src"}
    cmd = [str(SIM_PYTHON), "-u", str(ECM / "scripts/eval_milestone1_episode.py"), "--run-dir", str(run),
           "--server", str(k), "--task", item["task"], "--episode-id", str(item["episode_id"]),
           "--condition", item["condition"], "--attempt-dir", str(d)]
    atomic_json(d / "started.json", {"command": cmd, "utc": timestamp()})
    with (d / "console.log").open("x", buffering=1) as log:
        child = subprocess.Popen(cmd, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            atomic_json(result, {**item, "outcome": "wall_timeout", "error": "watchdog", "attempt_directory": str(d)})
        finally:
            stop_child(child)
    if not result.exists():
        atomic_json(result, {**item, "outcome": "process_error", "error": f"exit={child.returncode}", "attempt_directory": str(d)})
    return read_json(result)


def audit_episode(row, memory):
    """Replay IDs from actual client ingest boundaries; never use GT to manage memory."""
    directory = Path(row["attempt_directory"])
    decisions = read_jsonl(directory / "policy_decisions.jsonl")
    ingest = read_jsonl(directory / "client_ingest.jsonl")
    client = read_jsonl(directory / "client_decisions.jsonl")
    if row["outcome"] not in VALID or not decisions or not len(decisions) == len(ingest) == len(client):
        return {"passed": False, "error": "engineering outcome or missing/unpaired decisions"}
    timebase = read_json(directory / "timebase.json")
    assert 0 < timebase["control_freq_hz"] <= timebase["sim_freq_hz"]
    # Selection uses IDs, roles, phases and RNG only. Compact values avoid duplicating live SigLIP features.
    small = MemoryConfig(**{**memory, "image_dim": 1, "position_dim": 6})
    c = OnlineMemoryController(small, row["condition"], row["model_seed"])
    c.reset("audit", row["exec_start_idx"])
    approximations = 0
    expected_shapes = {"static_image_emb": [512, 2048], "static_pos_emb": [512, 768],
                       "static_state_emb": [512, 8], "static_mask": [512]}
    expected_dtypes = {k: "bool" if k == "static_mask" else "float32" for k in expected_shapes}
    next_frame = 0
    for index, (d, chunk) in enumerate(zip(decisions, ingest, strict=True)):
        assert chunk["decision_idx"] == client[index]["decision_idx"] == index
        assert chunk["first_frame_id"] == next_frame
        assert chunk["samples"] == chunk["last_frame_id"] - next_frame + 1 > 0
        next_frame = chunk["last_frame_id"] + 1
        for f in range(chunk["first_frame_id"], chunk["last_frame_id"] + 1):
            c.ingest(FrameRecord("audit", f, c.decision_id, None, np.zeros((1, 16, 1), np.float32),
                                 np.zeros((1, 16, 6), np.float32), np.zeros(8, np.float32)))
        c.prepare()
        F = chunk["last_frame_id"]
        assert d["decision_idx"] == index and d["available_max_frame_id"] == F
        assert client[index]["available_max_frame_id"] == F
        assert d["mode"] == row["condition"] and d["exec_start_idx"] == row["exec_start_idx"]
        assert d["selected_frame_ids"] == c.stats()["selected_frame_ids"], "live/replay selections differ"
        assert d["source_position_frame_ids"] == d["selected_frame_ids"]
        assert d["tensor_shapes"] == expected_shapes
        assert d["tensor_dtypes"] == expected_dtypes
        assert d["valid_tokens"] == min(F + 1, 32) * 16
        assert d["bank_frames"] == min(F + 1, 32) and d["rolling_frames"] <= 160
        assert d["retained_unique_frames"] <= 192 and d["retained_unique_bytes"] <= d["max_frame_bytes"] * 192
        assert d["phase_slot_counts"] == phase_counts(partition(F).uniform, row["exec_start_idx"])
        assert d["recent_window_approx"] == c.stats()["recent_window_approx"]
        approximations += int(d["recent_window_approx"])
        if not d["recent_window_approx"]:
            assert d["recent_frame_ids"] == list(partition(F).recent)
    return {"passed": True, "decisions": len(decisions), "recent_approx_decisions": approximations,
            "accepted_updates": decisions[-1]["accepted_updates"], "peak_retained_feature_bytes": max(d["retained_unique_bytes"] for d in decisions)}


def analyze(run):
    p = verify(run)
    rows, audits = [], []
    for item in read_json(run / "plan.json"):
        d = run / "episodes" / item["condition"] / item["task"] / f"ep{item['episode_id']:02d}_seed{item['model_seed']}"
        if not (d / "result.json").exists():
            continue
        row = read_json(d / "result.json")
        assert all(row[k] == item[k] for k in ("task", "episode_id", "model_seed", "condition", "server"))
        rows.append(row)
        try:
            audit = audit_episode(row, p["config"]["memory"])
        except Exception:
            audit = {"passed": False, "error": traceback.format_exc()}
        audits.append({"task": row["task"], "mode": row["condition"], "episode_id": row["episode_id"], **audit})
    baseline = read_json(run / "baseline_manifest.json") if (run / "baseline_manifest.json").exists() else {}
    cpu = read_json(run / "cpu_memory_stress.json") if (run / "cpu_memory_stress.json").exists() else {}
    errors = sum(r["outcome"] not in VALID for r in rows)
    modes = p["config"]["acceptance"]["modes"]
    updates_ok = all(sum(a.get("accepted_updates", 0) for a in audits if a["mode"] == mode) > 0 for mode in modes)
    cpu_ok = cpu_evidence(run) and cpu.get("passed", False)
    baseline_ok = (baseline.get("passed", False) and baseline.get("actions_bitwise_equal", False)
                   and sha256(run / "baseline_fixture.npz") == baseline.get("fixture_sha256"))
    passed = (len(rows) == p["planned_episodes"] and errors == 0 and all(a["passed"] for a in audits)
              and baseline_ok and cpu_ok and updates_ok)
    result = {"stage": "M1", "updated_utc": timestamp(), "passed": passed,
              "completed": len(rows), "planned": p["planned_episodes"], "engineering_errors": errors,
              "baseline_regression_passed": bool(baseline_ok), "cpu_storage_passed": cpu.get("passed", False),
              "cpu_tests_and_provenance_passed": bool(cpu_ok),
              "both_modes_exercised_replacements": updates_ok, "audits": audits,
              "episode_outcomes": [{k: r[k] for k in ("task", "condition", "episode_id", "outcome")} for r in rows],
              "scope": "engineering smoke on val ep0; success/fail/timeout are valid task outcomes, no performance improvement claim"}
    atomic_json(run / "acceptance.json", result)
    (run / "REPORT.md").write_text(
        f"# Milestone 1 验收\n\n完成 {len(rows)}/{p['planned_episodes']} 场；验收通过：{passed}；工程错误：{errors}。\n\n"
        f"真实 checkpoint 基线回归：{result['baseline_regression_passed']}；CPU 存储压力：{result['cpu_storage_passed']}；"
        f"CPU 测试及来源绑定：{cpu_ok}；"
        f"两种模式均有实际替换：{updates_ok}。\n\n"
        "逐场选帧因果重放、预算、阶段、近期窗口及内存上限检查见 acceptance.json。\n\n"
        "这是工程验收，不是成功率评估。160 帧 ring 不保证任意长轨迹的精确近期窗口，近似会逐决策记录；"
        "位置编码按原帧索引生成，可超过4096。\n")
    return result


def run_acceptance(run):
    protocol = verify(run)
    rt = protocol["config"]["acceptance"]
    with (run / "acceptance.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = read_json(run / "server0/config.json")
        run_cpu_checks(run)
        atomic_json(run / "progress.json", {"status": "checking_gpu", "updated_utc": timestamp(),
                                            "planned": protocol["planned_episodes"], "completed": 0})
        # Test driver access before loading a checkpoint. A failed preflight consumes no episode.
        probe = subprocess.run([str(POLICY_PYTHON), "-c", "import jax; d=jax.devices(); print(d); assert d[0].platform=='gpu'"],
                               env=gpu_env(cfg), cwd=POLICY, capture_output=True, text=True, timeout=60)
        atomic_json(run / "gpu_preflight.json", {"returncode": probe.returncode, "stdout": probe.stdout, "stderr": probe.stderr})
        if probe.returncode:
            raise RuntimeError("CUDA unavailable; run acceptance in the user's GPU terminal; see gpu_preflight.json")
        if not (run / "baseline_manifest.json").exists():
            atomic_json(run / "progress.json", {"status": "baseline_regression", "updated_utc": timestamp(),
                        "completed": 0, "planned": protocol["planned_episodes"],
                        "log": str(run / "baseline_regression.log")})
            print(f"Checkpoint action regression; log={run / 'baseline_regression.log'}", flush=True)
            with (run / "baseline_regression.log").open("a", buffering=1) as log:
                subprocess.run([str(POLICY_PYTHON), "-u", str(ECM / "scripts/verify_milestone1_policy.py"), "--out", str(run)],
                               env=gpu_env(cfg), cwd=POLICY, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        plan = read_json(run / "plan.json")
        for k in range(2):
            own = [item for item in plan if item["server"] == k]
            pending = [item for item in own if not (run / "episodes" / item["condition"] / item["task"] /
                        f"ep{item['episode_id']:02d}_seed{item['model_seed']}" / "result.json").exists()]
            if not pending:
                continue
            server, log, manifest = start_server(run, k, rt)
            try:
                for item in pending:
                    atomic_json(run / "active.json", {**item, "started_utc": timestamp()})
                    atomic_json(run / "progress.json", {"status": "running_episode", "active": item,
                        "completed": len(list((run / "episodes").glob("*/*/*/result.json"))),
                        "planned": len(plan), "updated_utc": timestamp()})
                    print(f"Starting {item['condition']} {item['task']} ep{item['episode_id']}", flush=True)
                    row = episode(run, item, rt["episode_wall_timeout_s"])
                    append_jsonl(run / f"server{k}/attempts.jsonl", row)
                    completed = len(list((run / "episodes").glob("*/*/*/result.json")))
                    atomic_json(run / "progress.json", {"status": "running", "updated_utc": timestamp(),
                                                       "completed": completed, "planned": len(plan), "last": item})
                    print(f"{completed}/{len(plan)} {item['condition']} {item['task']} ep{item['episode_id']}: {row['outcome']}", flush=True)
                    if row["outcome"] not in VALID:
                        raise RuntimeError(f"engineering error: {row['attempt_directory']}")
            finally:
                stop_child(server)
                log.close()
                manifest.update(status="stopped", server_ready=False, stopped_utc=timestamp())
                atomic_json(run / f"server{k}/server_manifest.json", manifest)
        result = analyze(run)
        atomic_json(run / "progress.json", {"status": "completed" if result["passed"] else "acceptance_failed",
                                            "updated_utc": timestamp(), "completed": result["completed"], "planned": len(plan)})
        print(f"Milestone 1 acceptance passed={result['passed']}; report={run / 'REPORT.md'}", flush=True)
        return 0 if result["passed"] else 2


def main():
    def terminate(signum, _frame):
        raise InterruptedError(f"acceptance interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    opts = parser.parse_args()
    run = opts.run.resolve()
    if not (run / "protocol.json").exists():
        prepare(run)
    verify(run)
    print(f"M1 acceptance directory: {run}", flush=True)
    if opts.prepare_only:
        return 0
    if opts.analyze_only:
        return 0 if analyze(run)["passed"] else 2
    try:
        return run_acceptance(run)
    except Exception:
        atomic_json(run / "acceptance_error.json", {"error": traceback.format_exc(), "updated_utc": timestamp()})
        atomic_json(run / "progress.json", {"status": "blocked_or_failed", "updated_utc": timestamp(),
                    "planned": read_json(run / "protocol.json")["planned_episodes"],
                    "completed": len(list((run / "episodes").glob("*/*/*/result.json"))),
                    "details": "acceptance_error.json"})
        raise


if __name__ == "__main__":
    sys.exit(main())
