#!/usr/bin/env python3
"""Stage 1 only: four cadence smokes, then paired A0-A4 on dev 200-219; stop at G0.

Run outside the sandbox with nohup setsid. One K0 server per GPU, at most four.
All five conditions for a layout run sequentially in the same server session.
Never opens dev 220-259 or val layouts. A failed/partial layout is archived on
explicit resume; complete layouts remain untouched. Failures are never deleted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from audit_fruit_execution_memory import audit
from run_fruit_diagnosis import start_server, wait_ready
from v_a_common import ECM, POLICY, SIM, SIM_PYTHON, atomic_json, read_json, read_jsonl, sha256, timestamp

CONDITIONS = {"A0": ("off", 16), "A1": ("v1", 16), "A2": ("v1", 8),
              "A3": ("v1", 4), "A4": ("off", 8)}
DEV = ECM / "data/fruit_picking_dev/rm75b_eg2/train"
TASKS = ("FruitCount", "FruitOccludedRipe")
GUARD = ECM / "reports/fruit_joint_guard_v1_20261102.json"


def run_one(out, label, task, episode, server, session, continue_guard_infeasible=False):
    child, cfg, context = server
    directory = out / label / task / f"ep{episode}"
    directory.mkdir(parents=True, exist_ok=True)
    guard, horizon = CONDITIONS[label.replace("smoke_", "")]
    cmd = [str(SIM_PYTHON), "-u", str(ECM / "scripts/diagnose_fruit_episode.py"),
           "--port", str(cfg["port"]), "--context", str(context), "--root", str(DEV),
           "--task", task, "--episode-id", str(episode), "--condition", "K0", "--mode", "closed",
           "--attempt-dir", str(directory), "--joint-guard", guard, "--joint-guard-stats", str(GUARD),
           "--exec-horizon", str(horizon), "--log-execution"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["policy_gpu"]),
           "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json", "PYTHONDONTWRITEBYTECODE": "1",
           "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
           "PYTHONPATH": f"{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    atomic_json(directory / "started.json", {"command": cmd, "session": session, "started_utc": timestamp()})
    with (directory / "console.log").open("a") as log:
        try:
            code = subprocess.run(cmd, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired:
            code = 124
    row = read_json(directory / "result.json") if (directory / "result.json").exists() else {
        "task": task, "episode_id": episode, "outcome": "error", "error": f"client exit {code}; no result"}
    row.update(server_session=session, experiment_condition=label, client_exit_code=code)
    guard_abort = bool(code != 0 and guard == "v1" and
                       "joint guard has no feasible envelope/step intersection" in row.get("error", ""))
    if guard_abort:
        row["joint_guard_infeasible"] = True
    atomic_json(directory / "result.json", row)
    if (code != 0 or row.get("error") or row["outcome"] not in ("success", "fail", "timeout")) and not (
            continue_guard_infeasible and guard_abort):
        raise RuntimeError(f"episode engineering error: {directory}")
    evidence = audit(directory, horizon, cfg["oracle"], allow_control_abort=continue_guard_infeasible and guard_abort)
    atomic_json(directory / "memory_audit.json", evidence)
    if not evidence["passed"]:
        raise RuntimeError(f"STOP: memory audit mismatch: {directory}")
    print({"condition": label, "task": task, "episode": episode,
           "outcome": row["outcome"], "audit_decisions": evidence["decisions"]}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--continue-guard-infeasible", action="store_true",
                   help="keep guard-refused episodes as engineering errors and run remaining comparisons; no controller relaxation")
    a = p.parse_args()
    if not 1 <= len(a.gpus) <= 4 or len(set(a.gpus)) != len(a.gpus) or any(g < 0 or g > 7 for g in a.gpus):
        p.error("use 1-4 distinct GPUs from 0-7")
    out = a.out.resolve()
    if out not in (ECM / "results/fruit_exec_guard_dev_20261102", ECM / "results/fruit_exec_guard_dev_20261102_v2"):
        p.error("use an explicitly registered stage 1 output directory")
    if a.continue_guard_infeasible and out.name != "fruit_exec_guard_dev_20261102_v2":
        p.error("--continue-guard-infeasible requires the separate _v2 output; original run stays intact")
    out.mkdir(parents=True, exist_ok=True)
    lock_file = (out / "run.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("stage 1 already running")
    registry = read_json(ECM / "reports/split_registry_20261102.json")
    if not registry.get("frozen") or registry.get("present_manifests") != 180:
        p.error("stage 0 frozen splits are required")
    formal = read_json(ECM / "results/fruit_dart_kf_eval_20261027/protocol.json")["params"]
    paths = [Path(__file__), ECM / "scripts/diagnose_fruit_episode.py", ECM / "scripts/audit_fruit_execution_memory.py",
             ECM / "scripts/report_fruit_exec_guard.py", ECM / "scripts/analyze_fruit_failure_modes.py",
             ECM / "scripts/run_fruit_diagnosis.py", ECM / "scripts/serve_teacher_policy.py",
             ECM / "src/memory_management/fruit_execution.py", ECM / "src/memory_management/fruit_eval.py",
             ECM / "src/memory_management/event_keyframes.py",
             ECM / "src/memory_management/validation/teacher_oracle.py",
             ECM / "src/memory_management/validation/acm_copy/diagnostic_logging.py",
             POLICY / "examples/robomme/eval.py", POLICY / "examples/robomme/utils.py",
             POLICY / "examples/robomme/subgoal_predictor.py"]
    plan = {"phase": 1, "tasks": TASKS, "episodes": list(range(200, 220)), "model_seed": 7,
            "conditions": CONDITIONS, "gpus": a.gpus, "checkpoint": formal["checkpoints"]["K0"],
            "checkpoint_manifest_sha256": sha256(Path(formal["checkpoints"]["K0"]) / "params/manifest.ocdbt"),
            "guard_stats_sha256": sha256(GUARD), "split_registry_sha256": sha256(ECM / "reports/split_registry_20261102.json"),
            "source_sha256": {str(f): sha256(f) for f in paths},
            "best_rule": "prefer conditions meeting both G0 thresholds; then first_target_in_basket, success, lower executed wrist deviation, then A0-A4 order",
            "smoke": "A2 and A3, each task ep200; all four must pass formal frame audit before A0-A4",
            "stop": "G0; never run G1 or phase 2 automatically"}
    if a.continue_guard_infeasible:
        plan["continue_guard_infeasible"] = True
        plan["guard_abort_rule"] = "no action on infeasible intersection; outcome=error retained; continue other episodes; memory audits still mandatory"
    # Normalize tuples to their JSON representation before comparing a resume plan.
    import json
    plan = json.loads(json.dumps(plan))
    if (out / "plan.json").exists() and read_json(out / "plan.json") != plan:
        p.error("stage 1 plan changed; refusing mixed results")
    atomic_json(out / "plan.json", plan)
    prior = read_json(out / "status.json") if (out / "status.json").exists() else {}
    start_utc = prior.get("started_utc", timestamp())
    session = timestamp().replace(":", "").replace("+", "_")
    servers, mutex, stop = [], threading.Lock(), threading.Event()
    progress = {"phase": 1, "state": "starting_servers", "started_utc": start_utc, "session": session,
                "updated_utc": timestamp(), "completed_layouts": 0}
    def status(**changes):
        with mutex:
            progress.update(changes, updated_utc=timestamp())
            atomic_json(out / "status.json", progress)
            root_status = read_json(ECM / "reports/lowlevel_rescue_status.json")
            root_status.update(phase=1, state=progress["state"], updated_utc=timestamp())
            atomic_json(ECM / "reports/lowlevel_rescue_status.json", root_status)
    status()
    # Starting a resume does not overwrite or erase a previous failed status/log.
    atomic_json(out / f"session_{session}.json", {"previous_status": prior, "started_utc": timestamp()})
    try:
        for i, gpu in enumerate(a.gpus):
            child, cfg = start_server(out / "servers", i, gpu, formal, conditions=("K0",), base_port=29021)
            servers.append((child, cfg, out / f"servers/server{i}/context.json"))
        wait_ready([(child, cfg) for child, cfg, _ in servers])
        status(state="memory_smokes")
        for label in ("smoke_A2", "smoke_A3"):
            for task in TASKS:
                directory = out / label / task / "ep200"
                # Archive smokes from a previous session; keep the evidence intact.
                if directory.exists():
                    dest = out / "superseded" / session / label / task / "ep200"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(directory), dest)
                run_one(out, label, task, 200, servers[0], session)
        status(state="paired_experiments")
        layouts = [(task, e) for e in range(200, 220) for task in TASKS]
        def lane(i):
            for task, e in layouts[i::len(servers)]:
                if stop.is_set():
                    return
                completed = all((out / c / task / f"ep{e}/memory_audit.json").exists()
                                and read_json(out / c / task / f"ep{e}/memory_audit.json")["passed"] for c in CONDITIONS)
                if not completed:
                    for c in CONDITIONS:
                        old = out / c / task / f"ep{e}"
                        if old.exists():
                            dest = out / "superseded" / session / c / task / f"ep{e}"
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(old), dest)
                    for c in CONDITIONS:
                        if stop.is_set():
                            return
                        try:
                            run_one(out, c, task, e, servers[i], session+f"/server{i}", a.continue_guard_infeasible)
                        except BaseException:
                            stop.set()
                            raise
                with mutex:
                    progress["completed_layouts"] += 1
                    progress.update(updated_utc=timestamp())
                    atomic_json(out / "status.json", progress)
        with ThreadPoolExecutor(max_workers=len(servers)) as pool:
            for future in as_completed([pool.submit(lane, i) for i in range(len(servers))]):
                future.result()
        status(state="summarizing")
        subprocess.run([sys.executable, str(ECM / "scripts/report_fruit_exec_guard.py"), "--out", str(out)],
                       cwd=ECM, check=True)
        status(state="awaiting_G0_confirmation", completed_layouts=40)
    except BaseException as exc:
        status(state="blocked", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        for child, _, _ in servers:
            child.terminate()
        for child, _, _ in servers:
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
