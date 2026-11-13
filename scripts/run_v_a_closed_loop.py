#!/usr/bin/env python3
"""V-A closed-loop supervisor (plan §0A.3): one policy server per (model seed, replica), paired conditions.

    python3 scripts/run_v_a_closed_loop.py --config configs/memory_management/validation/v_a_closed_loop.yaml
    python3 scripts/run_v_a_closed_loop.py --config ... --tag smoke --tasks VideoUnmaskSwap --episode-ids 0 --seeds 7
    python3 scripts/run_v_a_closed_loop.py --resume results/V_A_<run_id>
"""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

import numpy as np
import yaml

from v_a_common import (CHECKPOINT, ECM, OPENPI_DATA_HOME, POLICY, POLICY_PYTHON, SIM, SIM_PYTHON, VALID,
                        append_jsonl, atomic_json, git_head, read_json, read_jsonl, sha256, timestamp)

SCRIPT = Path(__file__).resolve()
CODE = [SCRIPT, ECM / "scripts/v_a_common.py", ECM / "scripts/serve_v_a_policy.py", ECM / "scripts/eval_v_a_episode.py",
        ECM / "scripts/analyze_v_a.py", ECM / "scripts/complete_v_a.py",
        *sorted((ECM / "src/memory_management/validation").rglob("*.py"))]


def dependencies(tasks):
    """Bind external policy/evaluation source and val metadata without copying checkpoint weights."""
    paths = [POLICY / "scripts/serve_policy.py",
             POLICY / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml"]
    for folder in (POLICY / "src/openpi", POLICY / "src/mme_vla_suite", POLICY / "packages/openpi-client/src",
                   POLICY / "examples/robomme", SIM / "src/robomme"):
        paths.extend(sorted(folder.rglob("*.py")))
    paths.extend(SIM / f"src/robomme/env_metadata/val/record_dataset_{t}_metadata.json" for t in tasks)
    paths.extend(sorted((CHECKPOINT / "assets").rglob("*.json")))
    return sorted(set(paths))


def build_plan(cfg, tasks, ids, seeds, conditions):
    rng = np.random.default_rng(cfg["schedule"]["shuffle_seed"])
    blocks = [(t, i, s) for t in tasks for i in ids for s in seeds]
    plan = []
    for b in rng.permutation(len(blocks)):
        task, episode_id, seed = blocks[b]
        for c in rng.permutation(len(conditions)):
            plan.append({"task": task, "episode_id": int(episode_id), "model_seed": int(seed),
                         "condition": conditions[c], "block": len(plan) // len(conditions)})
    return plan


def seal(opts):
    cfg = yaml.safe_load(opts.config.read_text())["v_a"]
    tasks = opts.tasks or cfg["tasks"]
    ids = opts.episode_ids or list(range(cfg["episode_ids"]["start"], cfg["episode_ids"]["stop"]))
    seeds = opts.seeds or cfg["model_seeds"]
    conditions = opts.conditions or cfg["conditions"]
    formal = getattr(opts, "formal", False)
    stage = getattr(opts, "stage", "primary")
    parent = getattr(opts, "parent_run", None)
    if formal:
        assert tasks == cfg["tasks"] and ids == list(range(cfg["episode_ids"]["start"], cfg["episode_ids"]["stop"]))
        if stage == "primary":
            assert seeds == cfg["model_seeds"] and conditions == cfg["conditions"] and parent is None
        else:
            assert parent is not None and seeds == cfg["extension_seeds"]
            primary = verify(parent)
            assert primary["formal"] and primary["experiment_stage"] == "primary" and primary["config"] == cfg
            assert conditions == read_json(parent / "extension_decision.json")["conditions"]
    run = getattr(opts, "out", None) or (ECM / "results" / f"V_A_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}{'_' + opts.tag if opts.tag else ''}")
    run = run.resolve()
    run.mkdir(parents=True)
    plan = build_plan(cfg, tasks, ids, seeds, conditions)
    atomic_json(run / "plan.json", plan)
    rt = cfg["runtime"]
    servers = [(s, r) for s in seeds for r in range(opts.servers_per_seed or rt["servers_per_seed"])]
    for k, (seed, replica) in enumerate(servers):
        d = run / f"server{k}"
        d.mkdir()
        atomic_json(d / "config.json", {"model_seed": seed, "replica": replica, "port": rt["base_port"] + k,
                                        "policy_gpu": rt["gpus"][k % len(rt["gpus"])],
                                        "sim_gpu": rt["gpus"][(k + 2) % len(rt["gpus"])],
                                        "checkpoint_path": str(CHECKPOINT), "policy_config": cfg["policy_config"],
                                        "openpi_data_home": str(OPENPI_DATA_HOME)})
    (run / "code_snapshot").mkdir()
    for path in CODE + [opts.config.resolve()]:
        (run / "code_snapshot" / str(path.relative_to(ECM)).replace("/", "__")).write_bytes(path.read_bytes())
    atomic_json(run / "protocol.json", {
        "run_id": run.name, "created_utc": timestamp(), "stage": "V-A", "split": cfg["split"], "config": cfg,
        "formal": formal, "experiment_stage": stage, "parent_run": str(parent.resolve()) if parent else None,
        "parent_protocol_sha256": sha256(parent / "protocol.json") if parent else None,
        "tasks": tasks, "episode_ids": ids, "model_seeds": seeds, "conditions": conditions,
        "planned_episodes": len(plan), "plan_sha256": sha256(run / "plan.json"), "config_sha256": sha256(opts.config),
        "code_identity": [{"path": str(p), "sha256": sha256(p)} for p in CODE],
        "dependency_identity": [{"path": str(p), "sha256": sha256(p)} for p in dependencies(tasks)],
        "server_config_sha256": [sha256(run / f"server{k}/config.json") for k in range(len(servers))],
        "policy_sha": git_head(POLICY), "sim_sha": git_head(SIM), "checkpoint_path": str(CHECKPOINT),
        "pairing": "conditions of one (task, ID, seed) share env seed and policy RNG reset at every episode",
        "attempt_policy": "all outcomes, errors and timeouts recorded; no automatic retries"})
    return run


def verify(run):
    p = read_json(run / "protocol.json")
    assert sha256(run / "plan.json") == p["plan_sha256"]
    for row in p["code_identity"] + p.get("dependency_identity", []):
        assert sha256(row["path"]) == row["sha256"], f"code changed since seal: {row['path']}"
    for k, digest in enumerate(p["server_config_sha256"]):
        assert sha256(run / f"server{k}/config.json") == digest
    return p


def done_keys(run):
    rows = [r for path in run.glob("server*/attempts.jsonl") for r in read_jsonl(path)]
    keys = [(r["task"], r["episode_id"], r["model_seed"], r["condition"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate attempts"
    return rows, set(keys)


def stop_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()


def start_server(run, k):
    d = run / f"server{k}"
    cfg = read_json(d / "config.json")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["policy_gpu"]), "JAX_PLATFORMS": "cuda",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONDONTWRITEBYTECODE": "1",
           "OPENPI_DATA_HOME": cfg["openpi_data_home"], "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
           "PYTHONPATH": f"{POLICY}/src:{POLICY}/packages/openpi-client/src"}
    command = [str(POLICY_PYTHON), "-u", str(ECM / "scripts/serve_v_a_policy.py"),
               "--config", str(d / "config.json"), "--context", str(d / "context.json")]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", cfg["port"]))
    log = (d / f"server_{int(time.time())}.log").open("x", buffering=1)
    child = subprocess.Popen(command, cwd=POLICY, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    manifest = {"command": command, "server_pid": child.pid, "supervisor_pid": os.getpid(), "status": "loading",
                "server_ready": False, "started_utc": timestamp(), "log": log.name,
                **{k_: cfg[k_] for k_ in ("checkpoint_path", "model_seed", "port", "policy_gpu")}}
    atomic_json(d / "server_manifest.json", manifest)
    return child, log, manifest


def run_episode(run, k, item, timeout_s):
    cfg = read_json(run / f"server{k}/config.json")
    directory = (run / "episodes" / item["condition"] / item["task"] /
                 f"ep{item['episode_id']:02d}_seed{item['model_seed']}")
    row = {**item, "run_id": run.name, "server": k, "outcome": "process_error",
           "error": "interrupted before durable result", "elapsed_seconds": 0, "attempt_directory": str(directory)}
    if not directory.exists():
        directory.mkdir(parents=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["sim_gpu"]), "PYTHONDONTWRITEBYTECODE": "1",
               "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json", "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
               "PYTHONPATH": f"{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src"}
        command = [str(SIM_PYTHON), "-u", str(ECM / "scripts/eval_v_a_episode.py"), "--run-dir", str(run),
                   "--server", str(k), "--task", item["task"], "--episode-id", str(item["episode_id"]),
                   "--condition", item["condition"], "--attempt-dir", str(directory)]
        atomic_json(directory / "started.json", {"command": command, "time_utc": timestamp()})
        start = time.monotonic()
        with (directory / "console.log").open("x", buffering=1) as log:
            child = subprocess.Popen(command, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                code = child.wait(timeout=timeout_s)
                row["error"] = f"process exit={code} without durable result"
            except subprocess.TimeoutExpired:
                row.update(outcome="wall_timeout", error=f"{timeout_s}s watchdog")
            finally:
                stop_child(child)
        row["elapsed_seconds"] = round(time.monotonic() - start, 3)
        atomic_json(directory / "exit.json", {"returncode": child.returncode, "elapsed_seconds": row["elapsed_seconds"]})
    if not (directory / "result.json").exists():
        atomic_json(directory / "result.json", row)
    return {**read_json(directory / "result.json"), "block": item["block"]}


def worker(run, k, items, rt, lock):
    errors = 0
    while not (run / "stop_requested").exists():
        try:
            item = items.get_nowait()
        except queue.Empty:
            return 0
        atomic_json(run / f"server{k}/active.json", {**item, "started_utc": timestamp()})
        row = run_episode(run, k, item, rt["episode_wall_timeout_s"])
        with lock:
            append_jsonl(run / f"server{k}/attempts.jsonl", row)
        errors = errors + 1 if row["outcome"] not in VALID else 0
        if errors >= rt["max_consecutive_errors"]:
            (run / "stop_requested").touch()
            return 1
    return 2


def refresh(run, status, protocol):
    rows, _ = done_keys(run)
    by = {}
    for c in protocol["conditions"]:
        own = [r for r in rows if r["condition"] == c]
        by[c] = {"completed": len(own), "success": sum(r["outcome"] == "success" for r in own),
                 "timeout": sum(r["outcome"] == "timeout" for r in own),
                 "errors": sum(r["outcome"] not in VALID for r in own)}
    atomic_json(run / "progress.json", {"status": status, "updated_utc": timestamp(), "supervisor_pid": os.getpid(),
                                        "planned": protocol["planned_episodes"], "completed": len(rows),
                                        "conditions": by})


def controller(run):
    # Hold for the entire controller lifetime; a stale PID alone cannot protect resume.
    with (run / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run / "stop_requested").unlink(missing_ok=True)
        return _controller(run)


def _controller(run):
    protocol = verify(run)
    rt = protocol["config"]["runtime"]
    configs = sorted(run.glob("server*/config.json"), key=lambda p: int(p.parent.name[6:]))
    seeds = [read_json(p)["model_seed"] for p in configs]
    _, done = done_keys(run)
    if len(done) == protocol["planned_episodes"]:
        refresh(run, "completed", protocol)
        return 0
    queues = {s: queue.Queue() for s in set(seeds)}
    for item in read_json(run / "plan.json"):
        if (item["task"], item["episode_id"], item["model_seed"], item["condition"]) not in done:
            queues[item["model_seed"]].put(item)
    servers, pool, state = [], None, "loading_servers"
    signal.signal(signal.SIGTERM, lambda *_: (run / "stop_requested").touch())
    try:
        for k in range(len(configs)):
            servers.append(start_server(run, k))
        deadline = time.monotonic() + rt["server_load_timeout_s"]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while not all(m["server_ready"] for _, _, m in servers):
            for k, (child, log, m) in enumerate(servers):
                assert child.poll() is None, f"server{k} exited while loading; see {log.name}"
                if m["server_ready"]:
                    continue
                try:
                    with opener.open(f"http://127.0.0.1:{m['port']}/healthz", timeout=1) as resp:
                        assert resp.read() == b"OK\n"
                    m.update(server_ready=True, status="serving", ready_utc=timestamp())
                    atomic_json(run / f"server{k}/server_manifest.json", m)
                except OSError:
                    pass
            assert time.monotonic() < deadline, "server loading timeout"
            refresh(run, state, protocol)
            time.sleep(5)
        state, lock = "running", threading.Lock()
        pool = ThreadPoolExecutor(max_workers=len(servers))
        jobs = [pool.submit(worker, run, k, queues[seeds[k]], rt, lock) for k in range(len(servers))]
        while not all(j.done() for j in jobs):
            refresh(run, state, protocol)
            if any(c.poll() is not None for c, _, _ in servers):
                raise RuntimeError("a policy server exited during V-A")
            time.sleep(10)
        results = [j.result() for j in jobs]
        complete = len(done_keys(run)[1]) == protocol["planned_episodes"]
        state = "completed" if complete and all(r == 0 for r in results) else "stopped_incomplete"
    except BaseException:
        atomic_json(run / "controller_error.json", {"error": traceback.format_exc(), "time_utc": timestamp()})
        (run / "stop_requested").touch()
        state = "error"
    finally:
        if pool:
            pool.shutdown(wait=True)
        for k, (child, log, m) in enumerate(servers):
            stop_child(child)
            m.update(status="stopped", stopped_utc=timestamp(), server_exit_code=child.returncode)
            atomic_json(run / f"server{k}/server_manifest.json", m)
            log.close()
        refresh(run, state, protocol)
    return 0 if state == "completed" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path)
    p.add_argument("--controller", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--tag", default="")
    p.add_argument("--out", type=Path)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--formal", action="store_true")
    p.add_argument("--stage", choices=("primary", "extension"), default="primary")
    p.add_argument("--parent-run", type=Path)
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--episode-ids", nargs="+", type=int)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--conditions", nargs="+")
    p.add_argument("--servers-per-seed", type=int)
    opts = p.parse_args()
    if opts.controller:
        return controller(opts.controller.resolve())
    if opts.resume:
        run = opts.resume.resolve()
        verify(run)
    else:
        run = seal(opts)
    if opts.prepare_only:
        print(f"prepared={run}", flush=True)
        return 0
    log = run / f"controller_{int(time.time())}.log"
    with log.open("x", buffering=1) as f:
        child = subprocess.Popen([sys.executable, "-u", str(SCRIPT), "--controller", str(run)], cwd=ECM,
                                 env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, stdout=f,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    atomic_json(run / "controller_launch.json", {"supervisor_pid": child.pid, "log": str(log), "started_utc": timestamp()})
    print(f"run={run} controller_pid={child.pid} log={log}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
