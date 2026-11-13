#!/usr/bin/env python3
"""Diagnosis of the floor in results/fruit_finetune_kf_eval_20261025 (descriptive; no protocol, no verdict).

Starts one policy server per GPU serving the formal run's K0 checkpoint with the evaluation's serving
parameters (its protocol.json), runs scripts/diagnose_fruit_episode.py jobs and summarizes:
  replay  K0's teacher-forced 16-step predictions along train and val demonstrations versus the
          demonstrated actions, next to the error of holding the current joint state;
  closed  K0 in closed loop on train episodes (layouts it was trained on), next to the formal val result.

    python3 scripts/run_fruit_diagnosis.py --out results/fruit_diagnosis_20261027
"""
import argparse
import collections
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import urllib.request

import numpy as np

from v_a_common import CHECKPOINT, ECM, OPENPI_DATA_HOME, POLICY, POLICY_PYTHON, SIM, SIM_PYTHON, atomic_json, read_json

EVAL = ECM / "results/fruit_finetune_kf_eval_20261025"
DATA = ECM / "data/fruit_picking/rm75b_eg2"
TASKS = ("FruitCount", "FruitOccludedRipe")
BASE_PORT = 28711


def start_server(out, k, gpu, params, conditions=("K0",), base_port=BASE_PORT):
    d = out / f"server{k}"
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"model_seed": 7, "replica": 0, "port": base_port + k, "policy_gpu": gpu, "checkpoint_path": str(CHECKPOINT),
           "policy_config": "mme_vla_suite", "openpi_data_home": str(OPENPI_DATA_HOME), "tasks": list(TASKS),
           "conditions": list(conditions), "oracle": params["oracle"],
           "checkpoints": {c: params["checkpoints"][c] for c in conditions}}
    atomic_json(d / "config.json", cfg)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "PYTHONDONTWRITEBYTECODE": "1", "OPENPI_DATA_HOME": str(OPENPI_DATA_HOME),
           "MPLCONFIGDIR": str(ECM / "cache/matplotlib"), "PYTHONPATH": f"{POLICY}/src:{POLICY}/packages/openpi-client/src"}
    log = (d / f"server_{int(time.time())}.log").open("x", buffering=1)
    child = subprocess.Popen([str(POLICY_PYTHON), "-u", str(ECM / "scripts/serve_teacher_policy.py"), "--config",
                              str(d / "config.json"), "--context", str(d / "context.json")],
                             cwd=POLICY, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return child, cfg


def wait_ready(servers, timeout=1800):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    for child, cfg in servers:
        while True:
            assert child.poll() is None, f"server on port {cfg['port']} exited"
            try:
                with opener.open(f"http://127.0.0.1:{cfg['port']}/healthz", timeout=1) as resp:
                    if resp.read() == b"OK\n":
                        break
            except OSError:
                pass
            assert time.monotonic() < deadline, "server loading timeout"
            time.sleep(5)


def run_job(out, k, cfg, job):
    mode, split, task, episode = job
    d = out / mode / split / task / f"ep{episode:02d}"
    if (d / "result.json").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["policy_gpu"]), "PYTHONDONTWRITEBYTECODE": "1",
           "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json", "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
           "PYTHONPATH": f"{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src"}
    command = [str(SIM_PYTHON), "-u", str(ECM / "scripts/diagnose_fruit_episode.py"), "--port", str(cfg["port"]),
               "--context", str(out / f"server{k}/context.json"), "--root", str(DATA / split), "--task", task,
               "--episode-id", str(episode), "--condition", "K0", "--mode", mode, "--attempt-dir", str(d)]
    with (d / "console.log").open("w") as log:
        try:
            subprocess.run(command, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        except subprocess.TimeoutExpired:
            atomic_json(d / "result.json", {"outcome": "wall_timeout", "error": "3600 s"})


def summarize_replay(out):
    result = {}
    for split in ("train", "val"):
        for task in TASKS:
            files = sorted((out / "replay" / split / task).glob("ep*/replay.npz"))
            if not files:
                continue
            pred, demo, state, dist = [], [], [], []
            for f in files:
                z = np.load(f)
                pred.append(z["pred"]), demo.append(z["demo"]), state.append(z["state"]), dist.append(z["distance"])
            pred, demo, state, dist = (np.concatenate(x) for x in (pred, demo, state, dist))
            joint = np.abs(pred[..., :7] - demo[..., :7]).mean(-1)
            hold = np.abs(state[:, None, :7] - demo[..., :7]).mean(-1)
            grip_ok = np.sign(pred[..., 7]) == np.sign(demo[..., 7])
            valid = np.isfinite(demo[..., 7])
            switch = valid & (np.sign(demo[..., 7]) != np.sign(demo[:, :1, 7]))
            near, carry = dist <= 0.10, demo[:, 0, 7] < 0

            def med(x, rows=slice(None)):
                x = x[rows]
                return {f"h{h + 1}": float(np.nanmedian(x[:, h])) for h in (0, 7, 15)}

            result[f"{split}/{task}"] = {
                "episodes": len(files), "decisions": len(pred),
                "joint_abs_err_rad": med(joint), "hold_abs_err_rad": med(hold),
                "joint_abs_err_rad_near_fruit": med(joint, near), "joint_abs_err_rad_carrying": med(joint, carry),
                "gripper_sign_agreement": float(grip_ok[valid].mean()),
                "gripper_sign_agreement_at_switches": float(grip_ok[switch].mean()) if switch.any() else None}
    return result


def summarize_closed(out):
    result = {}
    groups = {f"train/{t}": sorted((out / "closed/train" / t).glob("ep*/result.json")) for t in TASKS}
    groups.update({f"val (formal K0, seed 7)/{t}": sorted((EVAL / "episodes/K0" / t).glob("ep*_seed7/result.json"))
                   for t in TASKS})
    for name, files in groups.items():
        rows = [read_json(f) for f in files]
        if not rows:
            continue
        result[name] = {"episodes": len(rows), "success": sum(r["outcome"] == "success" for r in rows),
                        "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
                        "detached_ge1": sum((r.get("final_detached") or 0) >= 1 for r in rows),
                        "in_basket_ge1": sum((r.get("final_in_basket") or 0) >= 1 for r in rows),
                        "mean_in_basket": float(np.mean([r.get("final_in_basket") or 0 for r in rows])),
                        "mean_n_pick": float(np.mean([r.get("n_pick") or 0 for r in rows]))}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--summarize-only", action="store_true")
    a = p.parse_args()
    out = a.out.resolve()
    if a.summarize_only:
        summary = {"replay": summarize_replay(out), "closed": summarize_closed(out)}
        atomic_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=1))
        return 0
    params = read_json(EVAL / "protocol.json")["params"]
    jobs = queue.Queue()
    for e in range(a.episodes):
        for task in TASKS:
            jobs.put(("closed", "train", task, e))
            for split in ("train", "val"):
                jobs.put(("replay", split, task, e))
    servers = [start_server(out, k, gpu, params) for k, gpu in enumerate(a.gpus)]
    try:
        wait_ready(servers)

        def work(k, cfg):
            while True:
                try:
                    job = jobs.get_nowait()
                except queue.Empty:
                    return
                run_job(out, k, cfg, job)

        threads = [threading.Thread(target=work, args=(k, cfg)) for k, (_, cfg) in enumerate(servers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        for child, _ in servers:
            child.terminate()
            child.wait(120)
    summary = {"replay": summarize_replay(out), "closed": summarize_closed(out)}
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
