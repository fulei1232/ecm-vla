#!/usr/bin/env python3
"""Teacher Oracle step 3 and the memory ablation: seal, run and analyze.

Protocols: reports/teacher_oracle_pilot_protocol_20261024.md (--stage pilot, PatternLock),
reports/teacher_oracle_extension_protocol_20261024.md (--stage extension, BinFill + VideoUnmaskSwap) and
reports/memory_ablation_protocol_20261024.md (--stage ablation, U0/NH/NM on all three tasks) and
reports/memory_budget_calibration_protocol_20261025.md (--stage budget, U0/B16/B8 on PatternLock val 20-49) and
reports/teacher_oracle_budget16_protocol_20261025.md (--stage budget16, the pilot with a 16-frame memory).
--stage event (event memory, event_memory.py; reports/event_memory_protocol_20261025.md).
--stage finetune (event-keyframe fine-tuning, closed loop on val 20-49; reports/finetune_kf_protocol_20261025.md):
K0/K4/K8/K12 on their fine-tuned checkpoints and B16 on 79999, one multi-checkpoint server per model seed.
--stage fruit (the same on the RM75-B fruit-picking simulator, val 0-49; reports/fruit_finetune_kf_protocol_20261025.md):
K0/K4/K8, episodes run by eval_fruit_episode.py, distances from the external module fruit_distance.RecordedDistance.
--stage fruit_dart (K0/K4/K8 fine-tuned on clean + DART demonstrations, and C0 = the fruit stage's K0 checkpoint,
val 50-99; reports/fruit_dart_protocol_20261027.md); --stage fruit_dev (its development gate: K0 vs C0 on the
dev episodes 200-219, run by scripts/run_fruit_dev_gate.py).

    python3 scripts/run_teacher_oracle.py all --smoke --out results/teacher_oracle_smoke_20261024
    python3 scripts/run_teacher_oracle.py all --out results/teacher_oracle_20261024
    python3 scripts/run_teacher_oracle.py all --stage extension --out results/teacher_oracle_ext_20261024

Subcommands: prepare | run | analyze | all. A block is one (task, episode, model seed) with its five
conditions, all on one server process. `run` resumes: complete blocks are kept; partial blocks are
moved to superseded/ and rerun whole on the new server processes.
"""
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

import numpy as np

from v_a_common import (CHECKPOINT, ECM, OPENPI_DATA_HOME, POLICY, POLICY_PYTHON, SIM, SIM_PYTHON, VALID,
                        append_jsonl, atomic_json, git_head, read_json, read_jsonl, sha256, timestamp)
import analyze_v_a as ava
import run_v_a_closed_loop as va

sys.path.insert(0, str(ECM / "src"))
from memory_management import event_keyframes as ek  # noqa: E402
from memory_management.validation import event_memory as em  # noqa: E402
from memory_management.validation import teacher_oracle as to  # noqa: E402
from memory_management.validation.partition import partition  # noqa: E402

SCRIPT = Path(__file__).resolve()
G0B = ECM / "results/V_B_cached_loss_g0b_fp32.json"
V_A_RUN = ECM / "results/V_A_formal_20261023"
CODE = [SCRIPT, ECM / "scripts/v_a_common.py", ECM / "scripts/run_v_a_closed_loop.py", ECM / "scripts/analyze_v_a.py",
        ECM / "scripts/serve_teacher_policy.py", ECM / "scripts/eval_teacher_episode.py",
        ECM / "scripts/teacher_branch_worker.py", ECM / "src/memory_management/event_keyframes.py",
        *sorted((ECM / "src/memory_management/validation").rglob("*.py"))]
FRUIT_CODE = [ECM / "scripts/eval_fruit_episode.py", ECM / "src/memory_management/fruit_eval.py",
              ECM / "src/memory_management/fruit_distance.py"]
FINETUNE_RUN = ECM / "runs/finetune_kf_20261025"
FINETUNE_STEP = 5999
FRUIT_RUN = ECM / "runs/fruit_finetune_kf_20261025"
FRUIT_STEP = 11999
FRUIT_DART_RUN = ECM / "runs/fruit_dart_kf_20261027"
FRUIT_TASKS = {"FruitCount": "count", "FruitOccludedRipe": "occluded_ripe"}
FRUIT = {"root": str(ECM / "data/fruit_picking/rm75b_eg2/val"), "robot": "rm75b_eg2"}
FRUIT_ORACLE = {"capacity": 16,
                "event": {"alpha": 0.5, "tau": 0.03, "d0": 0.0, "alphas": [0.5], "pool": "random",
                          "signal": {"kind": "external", "factory": "memory_management.fruit_distance:RecordedDistance"}},
                "keyframes": {f"K{k}": {"frames": 16, "events": k, "radius": 0.03, "tol": 0.005} for k in (0, 4, 8)}}
ORACLE_CONDITIONS = ["U0", "S0", "TO", "AO", "RS"]
PARAMS = {
    "model_seeds": [7, 8, 9], "conditions": ORACLE_CONDITIONS,
    "oracle": {"pool_size": 8, "mc_samples": 16, "batch_size": 16, "se_mult": 2.0, "rel_err_max": 1e-3},
    "expert": {"chunk_len": 20, "replan_every": 16, "max_steps": 1300, "stall_steps": 400, "stall_solves": 3},
    "runtime": {"servers_per_seed": 2, "gpus": [0, 1, 2, 3], "base_port": 28511, "episode_wall_timeout_s": 7200,
                "branch_timeout_s": 900, "max_consecutive_errors": 3, "server_load_timeout_s": 1800},
    "schedule": {"shuffle_seed": 20261024},
    "stats": {"bootstrap_draws": 20000, "bootstrap_seed": 20261024, "margin_pp": 12, "alpha": 0.05,
              "contrasts": {"C1": ["TO", "S0"], "C2": ["TO", "RS"], "C3": ["AO", "S0"]}},
    "gates": {"max_error_rate": 0.02, "expert_available_min": 0.9},
}
# smoke: one block per task at model seed 7; "first" = episode 0, "shortest" = the shortest V-A D0a seed-7 episode.
# "pooled" contrasts weight tasks equally (oracle route); "per_task" contrasts are one Holm family (ablation route).
STAGES = {
    "pilot": {"doc": ECM / "reports/teacher_oracle_pilot_protocol_20261024.md", "tasks": ["PatternLock"],
              "experts": {"PatternLock": "original"}, "episode_wall_timeout_s": 7200, "smoke_servers_per_seed": 1,
              "smoke_episode": "first"},
    "extension": {"doc": ECM / "reports/teacher_oracle_extension_protocol_20261024.md",
                  "tasks": ["BinFill", "VideoUnmaskSwap"], "experts": {"BinFill": "phase", "VideoUnmaskSwap": "phase"},
                  "episode_wall_timeout_s": 14400, "smoke_servers_per_seed": 2, "smoke_episode": "shortest"},
    "ablation": {"doc": ECM / "reports/memory_ablation_protocol_20261024.md",
                 "tasks": ["PatternLock", "BinFill", "VideoUnmaskSwap"], "experts": {}, "conditions": ["U0", "NH", "NM"],
                 "contrasts": {"A1": ["NH", "U0"], "A2": ["NM", "U0"]}, "scope": "per_task", "control": "PatternLock",
                 "episode_wall_timeout_s": 7200, "smoke_servers_per_seed": 2, "smoke_episode": "shortest"},
    "budget": {"doc": ECM / "reports/memory_budget_calibration_protocol_20261025.md", "tasks": ["PatternLock"],
               "episode_ids": list(range(20, 50)), "experts": {}, "conditions": ["U0", "B16", "B8"],
               "contrasts": {"B1": ["B16", "U0"], "B2": ["B8", "U0"]}, "scope": "per_task", "control": None,
               "route": {"kind": "budget", "order": ["B16", "B8"], "min_drop": 0.15, "min_success": 0.20},
               "episode_wall_timeout_s": 7200, "smoke_servers_per_seed": 1, "smoke_episode": "first"},
    "budget16": {"doc": ECM / "reports/teacher_oracle_budget16_protocol_20261025.md", "tasks": ["PatternLock"],
                 "experts": {"PatternLock": "original"}, "oracle": {"capacity": 16},
                 "episode_wall_timeout_s": 7200, "smoke_servers_per_seed": 1, "smoke_episode": "first"},
    "event": {"doc": ECM / "reports/event_memory_protocol_20261025.md",
              "tasks": ["PatternLock", "BinFill", "VideoUnmaskSwap"], "experts": {},
              "oracle": {"event": {"alpha": 0.5, "tau": {"PatternLock": 0.01, "BinFill": 0.05, "VideoUnmaskSwap": 0.05},
                                   "d0": 0.0, "alphas": [0.25, 0.5, 0.75, 1.0], "pool": "random",
                                   "signal": {"kind": "ground_truth"}}},
              "conditions": ["U0", "EV", "EVU", "EVR", "EVA"],
              "contrasts": {"E1": ["EV", "U0"], "E2": ["EV", "EVU"]},
              "secondary": {"E3": ["EV", "EVR"], "E4": ["EVA", "U0"]},
              "scope": "per_task", "control": None, "route": {"kind": "event"},
              "episode_wall_timeout_s": 7200, "smoke_servers_per_seed": 2, "smoke_episode": "shortest"},
    "finetune": {"doc": ECM / "reports/finetune_kf_protocol_20261025.md",
                 "tasks": ["PatternLock", "BinFill", "VideoUnmaskSwap"], "episode_ids": list(range(20, 50)),
                 "experts": {},
                 "oracle": {"capacity": 16,
                            "event": {"alpha": 0.5, "tau": 0.03, "d0": 0.0, "alphas": [0.5], "pool": "random",
                                      "signal": {"kind": "ground_truth"}},
                            "keyframes": {f"K{k}": {"frames": 16, "events": k, "radius": 0.03, "tol": 0.005}
                                          for k in (0, 4, 8, 12)}},
                 "checkpoints": {**{f"K{k}": FINETUNE_RUN / f"ckpt/mme_vla_suite/k{k}/{FINETUNE_STEP}"
                                    for k in (0, 4, 8, 12)}, "B16": CHECKPOINT},
                 "conditions": ["K0", "K4", "K8", "K12", "B16"],
                 "contrasts": {"F1": ["K4", "K0"], "F2": ["K8", "K0"], "F3": ["K12", "K0"]},
                 "secondary": {"F4": ["K0", "B16"]},
                 "scope": "per_task", "control": None, "route": {"kind": "finetune"}, "servers_per_seed": 1,
                 "episode_wall_timeout_s": 7200, "smoke_servers_per_seed": 1, "smoke_episode": "first"},
    "fruit": {"doc": ECM / "reports/fruit_finetune_kf_protocol_20261025.md", "tasks": list(FRUIT_TASKS),
              "episode_ids": list(range(50)), "experts": {}, "oracle": FRUIT_ORACLE,
              "checkpoints": {f"K{k}": FRUIT_RUN / f"ckpt/mme_vla_suite/k{k}/{FRUIT_STEP}" for k in (0, 4, 8)},
              "conditions": ["K0", "K4", "K8"], "contrasts": {"G1": ["K4", "K0"], "G2": ["K8", "K0"]},
              "scope": "per_task", "control": None, "route": {"kind": "finetune", "floor": 0.05},
              "servers_per_seed": 1, "fruit": FRUIT, "client": "eval_fruit_episode.py",
              "episode_wall_timeout_s": 7200, "smoke_servers_per_seed": 1, "smoke_episode": "first"},
    "fruit_dart": {"doc": ECM / "reports/fruit_dart_protocol_20261027.md", "tasks": list(FRUIT_TASKS),
                   "episode_ids": list(range(50, 100)), "experts": {},
                   "oracle": {**FRUIT_ORACLE, "keyframes": {**FRUIT_ORACLE["keyframes"],
                                                             "C0": FRUIT_ORACLE["keyframes"]["K0"]}},
                   "checkpoints": {**{f"K{k}": FRUIT_DART_RUN / f"ckpt/mme_vla_suite/k{k}/{FRUIT_STEP}" for k in (0, 4, 8)},
                                   "C0": FRUIT_RUN / f"ckpt/mme_vla_suite/k0/{FRUIT_STEP}"},
                   "conditions": ["K0", "K4", "K8", "C0"], "contrasts": {"G1": ["K4", "K0"], "G2": ["K8", "K0"]},
                   "secondary": {"D1": ["K0", "C0"]},
                   "scope": "per_task", "control": None, "route": {"kind": "finetune", "floor": 0.05, "effect": "D1"},
                   # four FP32 checkpoints (~43 GB) do not fit one 48 GB L20: the compared arms share one server
                   # (~32.5 GB, as in the fruit stage), the reference C0 gets its own; simulators on GPUs 6-7
                   "servers_per_seed": 1, "condition_groups": [["K0", "K4", "K8"], ["C0"]],
                   "gpus": [0, 1, 2, 3, 4, 5], "sim_gpus": [6, 7], "fruit": FRUIT, "client": "eval_fruit_episode.py",
                   "episode_wall_timeout_s": 7200, "smoke_servers_per_seed": 1, "smoke_episode": "first"},
    # development gate D of the DART study (scripts/run_fruit_dev_gate.py): DART K0 vs the clean K0 on the clean dev
    # episodes 200-219 (never trained on, never in the formal evaluation); the client wants a root named val
    "fruit_dev": {"doc": ECM / "reports/fruit_dart_protocol_20261027.md", "tasks": list(FRUIT_TASKS),
                  "episode_ids": list(range(200, 220)), "experts": {}, "model_seeds": [7],
                  "oracle": {**FRUIT_ORACLE, "keyframes": {"K0": FRUIT_ORACLE["keyframes"]["K0"],
                                                            "C0": FRUIT_ORACLE["keyframes"]["K0"]}},
                  "checkpoints": {"K0": FRUIT_DART_RUN / f"ckpt/mme_vla_suite/k0/{FRUIT_STEP}",
                                  "C0": FRUIT_RUN / f"ckpt/mme_vla_suite/k0/{FRUIT_STEP}"},
                  "conditions": ["K0", "C0"], "contrasts": {"D1": ["K0", "C0"]},
                  "scope": "per_task", "control": None, "route": {"kind": "finetune", "floor": 0.05},
                  "servers_per_seed": 4, "gpus": [0, 1, 2, 3], "sim_gpus": [4, 5, 6, 7],
                  "fruit": {"root": str(ECM / "data/fruit_picking_dev/rm75b_eg2/val"), "robot": "rm75b_eg2"},
                  "client": "eval_fruit_episode.py",
                  "episode_wall_timeout_s": 7200, "smoke_servers_per_seed": 1, "smoke_episode": "first"},
}


def smoke_episode(rule, task, ids, seed=7):
    if rule == "first":
        return ids[0]
    rows = [r for r in read_csv(V_A_RUN / "per_episode.csv")
            if r["task"] == task and r["condition"] == "D0a" and int(r["model_seed"]) == seed]
    return min((int(r["executed_steps"]), int(r["episode_id"])) for r in rows)[1]


def read_csv(path):
    import csv
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def stage_params(stage, smoke):
    s = STAGES[stage]
    params = json.loads(json.dumps(PARAMS))
    ids = s.get("episode_ids", list(range(20)))
    params.update(stage=stage, tasks=s["tasks"], experts=s["experts"], episodes={t: list(ids) for t in s["tasks"]})
    params["runtime"]["episode_wall_timeout_s"] = s["episode_wall_timeout_s"]
    if "oracle" in s:
        params["oracle"].update(s["oracle"])
    if "conditions" in s:
        params["conditions"] = s["conditions"]
        params["stats"].update(contrasts=s["contrasts"], scope=s["scope"], control=s["control"])
    if "route" in s:
        params["stats"]["route"] = s["route"]
    if "secondary" in s:
        params["stats"]["secondary"] = s["secondary"]
    if "checkpoints" in s:
        params["checkpoints"] = {c: str(p) for c, p in s["checkpoints"].items()}
    if "servers_per_seed" in s:
        params["runtime"]["servers_per_seed"] = s["servers_per_seed"]
    for key in ("condition_groups", "gpus", "sim_gpus"):
        if key in s:
            params["runtime"][key] = s[key]
    if "model_seeds" in s:
        params["model_seeds"] = s["model_seeds"]
    if "fruit" in s:
        params.update(fruit=dict(s["fruit"]), client=s["client"])
    if smoke:
        params["model_seeds"] = [7]
        params["runtime"]["servers_per_seed"] = s["smoke_servers_per_seed"]
        params["episodes"] = {t: [smoke_episode(s["smoke_episode"], t, ids)] for t in params["tasks"]}
    return params


def task_list(params):
    return params.get("tasks") or [params["task"]]


def code_files(params):
    return CODE + (FRUIT_CODE if "fruit" in params else [])


def dependencies(params):
    """va.dependencies; fruit stages bind the fruit simulator package and the planned val manifests instead of
    the RoboMME val metadata."""
    if "fruit" not in params:
        return va.dependencies(params["tasks"])
    root = Path(params["fruit"]["root"])
    manifests = [root / FRUIT_TASKS[t] / f"ep{e}_manifest.json" for t in params["tasks"] for e in params["episodes"][t]]
    package = [p for p in (ECM / "fruit_picking").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return sorted(set(va.dependencies([]) + package + manifests))


def block_key(task, episode_id, seed):
    return f"{task}_ep{episode_id:02d}_seed{seed}"


def episode_dir(out, condition, block):
    return out / "episodes" / condition / block["task"] / f"ep{block['episode_id']:02d}_seed{block['model_seed']}"


def build_plan(params):
    rng = np.random.default_rng(params["schedule"]["shuffle_seed"])
    blocks = [(t, i, s) for t in params["tasks"] for i in params["episodes"][t] for s in params["model_seeds"]]
    plan = []
    free = [c for c in params["conditions"] if c != "RS"]
    last = ["RS"] if "RS" in params["conditions"] else []
    for b in rng.permutation(len(blocks)):
        task, episode_id, seed = blocks[b]
        order = [free[c] for c in rng.permutation(len(free))] + last
        plan.append({"key": block_key(task, int(episode_id), int(seed)), "task": task,
                     "episode_id": int(episode_id), "model_seed": int(seed), "order": order})
    return plan


def check_g0b(path):
    gate = read_json(path)
    if not (gate.get("passed") is True and gate.get("compute_dtype") == "float32"
            and gate.get("matmul_precision") == "highest" and float(gate["max_relative_error"]) <= 1e-3
            and Path(gate["checkpoint_path"]).resolve() == CHECKPOINT.resolve()):
        raise SystemExit("the FP32 G0b report does not cover this checkpoint/runtime")


# --------------------------------------------------------------------------- seal

def checkpoint_identity(path, formal):
    """Hashes of a served checkpoint; fine-tuned ones (<run>/ckpt/<config>/<arm>/<step>) also name their run."""
    path = Path(path)
    files = {"params_manifest": path / "params/manifest.ocdbt", "metadata": path / "_CHECKPOINT_METADATA",
             "norm_stats": path / "assets/robomme/norm_stats.json", "history_config": path.parent / "history_config.txt"}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"checkpoint {path} is incomplete: {missing}")
    row = {"path": str(path), **{f"{k}_sha256": sha256(p) for k, p in files.items()}}
    run = path.parents[3]
    if path.resolve() != CHECKPOINT.resolve():
        train = read_json(run / "protocol.json")
        if formal and not train.get("formal"):
            raise SystemExit(f"{path} comes from a non-formal training run")
        row.update(train_protocol=str(run / "protocol.json"), train_protocol_sha256=sha256(run / "protocol.json"))
    return row


def server_configs(params):
    """One server per (model seed, replica, condition group); a lane (seed, replica) runs a block's conditions on
    the servers of their groups."""
    rt, checkpoints = params["runtime"], params.get("checkpoints")
    groups = rt.get("condition_groups") or [params["conditions"]]
    servers = [(s, r, g) for s in params["model_seeds"] for r in range(rt["servers_per_seed"]) for g in groups]
    configs = []
    for k, (seed, replica, group) in enumerate(servers):
        sim_gpu = rt["sim_gpus"][k % len(rt["sim_gpus"])] if "sim_gpus" in rt else rt["gpus"][(k + 2) % len(rt["gpus"])]
        configs.append({
            "model_seed": seed, "replica": replica, "port": rt["base_port"] + k,
            "policy_gpu": rt["gpus"][k % len(rt["gpus"])], "sim_gpu": sim_gpu,
            "checkpoint_path": str(CHECKPOINT), "policy_config": "mme_vla_suite", "openpi_data_home": str(OPENPI_DATA_HOME),
            "tasks": params["tasks"], "conditions": group, "oracle": params["oracle"],
            **({"checkpoints": checkpoints if len(groups) == 1 else {c: checkpoints[c] for c in group if c in checkpoints}}
               if checkpoints else {})})
    return configs


def prepare(out, smoke, stage, checkpoint_run=None, checkpoint_step=None):
    if (out / "protocol.json").exists():
        raise SystemExit(f"{out} is already prepared; use a new directory")
    check_g0b(G0B)
    params = stage_params(stage, smoke)
    if checkpoint_run is not None:
        if not smoke or "checkpoints" not in params:
            raise SystemExit("--checkpoint-run only replaces the fine-tuned checkpoints of a smoke run")
        params["checkpoints"].update({c: str(checkpoint_run / f"ckpt/mme_vla_suite/k{c[1:]}/{checkpoint_step}")
                                      for c in params["checkpoints"] if c in to.KEYFRAME and c.startswith("K")})
    checkpoints = params.get("checkpoints")
    identity = [checkpoint_identity(p, not smoke) for p in dict.fromkeys(checkpoints.values())] if checkpoints else []
    doc = STAGES[stage]["doc"]
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan(params)
    atomic_json(out / "plan.json", plan)
    servers = server_configs(params)
    for k, cfg in enumerate(servers):
        (out / f"server{k}").mkdir()
        atomic_json(out / f"server{k}/config.json", cfg)
    atomic_json(out / "protocol.json", {
        "name": f"teacher_oracle_{stage}", "formal": not smoke, "created_utc": timestamp(), "split": "val",
        "protocol_doc": str(doc), "protocol_doc_sha256": sha256(doc), "g0b_report": str(G0B), "g0b_sha256": sha256(G0B),
        "params": params, "planned_blocks": len(plan), "plan_sha256": sha256(out / "plan.json"),
        "server_config_sha256": [sha256(out / f"server{k}/config.json") for k in range(len(servers))],
        "code_identity": [{"path": str(p), "sha256": sha256(p)} for p in code_files(params)],
        "dependency_identity": [{"path": str(p), "sha256": sha256(p)} for p in dependencies(params)],
        "policy_sha": git_head(POLICY), "sim_sha": git_head(SIM), "checkpoint_path": str(CHECKPOINT),
        **({"checkpoint_identity": identity} if identity else {}),
        "pairing": "the conditions of a block share env seed and policy RNG reset; each condition group "
                   "(runtime.condition_groups, default all conditions) one server process",
        "attempt_policy": "all outcomes, errors and timeouts recorded; partial blocks rerun whole on resume"})


def verify(out):
    p = read_json(out / "protocol.json")
    assert sha256(out / "plan.json") == p["plan_sha256"], "plan changed since seal"
    assert sha256(p["protocol_doc"]) == p["protocol_doc_sha256"], "protocol document changed since seal"
    assert sha256(p["g0b_report"]) == p["g0b_sha256"], "G0b report changed since seal"
    for row in p["code_identity"] + p["dependency_identity"]:
        assert sha256(row["path"]) == row["sha256"], f"code changed since seal: {row['path']}"
    for k, digest in enumerate(p["server_config_sha256"]):
        assert sha256(out / f"server{k}/config.json") == digest
    for row in p.get("checkpoint_identity", []):
        assert sha256(Path(row["path"]) / "params/manifest.ocdbt") == row["params_manifest_sha256"], \
            f"checkpoint changed since seal: {row['path']}"
    return p


# --------------------------------------------------------------------------- blocks

def block_results(out, protocol, block):
    rows = {}
    for c in protocol["params"]["conditions"]:
        path = episode_dir(out, c, block) / "result.json"
        if path.exists():
            rows[c] = read_json(path)
    return rows


def block_complete(protocol, rows):
    sessions = {r.get("server_session") for r in rows.values()}
    groups = len(protocol["params"].get("runtime", {}).get("condition_groups") or [None])
    return len(rows) == len(protocol["params"]["conditions"]) and len(sessions) == groups and None not in sessions


def supersede_partial(out, protocol, plan):
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    moved = []
    for block in plan:
        rows = block_results(out, protocol, block)
        dirs = [episode_dir(out, c, block) for c in protocol["params"]["conditions"]]
        if block_complete(protocol, rows) or not any(d.exists() for d in dirs):
            continue
        for d in dirs:
            if d.exists():
                dest = out / "superseded" / stamp / d.relative_to(out / "episodes")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(dest))
                moved.append(str(dest))
    if moved:
        append_jsonl(out / "superseded.jsonl", {"time_utc": timestamp(), "moved": moved})


def start_server(out, k):
    d = out / f"server{k}"
    cfg = read_json(d / "config.json")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["policy_gpu"]), "JAX_PLATFORMS": "cuda",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONDONTWRITEBYTECODE": "1",
           "OPENPI_DATA_HOME": cfg["openpi_data_home"], "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
           "PYTHONPATH": f"{POLICY}/src:{POLICY}/packages/openpi-client/src"}
    command = [str(POLICY_PYTHON), "-u", str(ECM / "scripts/serve_teacher_policy.py"),
               "--config", str(d / "config.json"), "--context", str(d / "context.json")]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", cfg["port"]))
    log = (d / f"server_{int(time.time())}.log").open("x", buffering=1)
    child = subprocess.Popen(command, cwd=POLICY, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    manifest = {"command": command, "server_pid": child.pid, "supervisor_pid": os.getpid(), "status": "loading",
                "server_ready": False, "started_utc": timestamp(), "log": log.name,
                **{key: cfg[key] for key in ("checkpoint_path", "model_seed", "port", "policy_gpu")}}
    atomic_json(d / "server_manifest.json", manifest)
    return child, log, manifest


def run_episode(out, k, block, condition, timeout_s, schedule=None):
    cfg = read_json(out / f"server{k}/config.json")
    client = read_json(out / "protocol.json")["params"].get("client", "eval_teacher_episode.py")
    manifest = read_json(out / f"server{k}/server_manifest.json")
    directory = episode_dir(out, condition, block)
    row = {"task": block["task"], "episode_id": block["episode_id"], "model_seed": block["model_seed"],
           "condition": condition, "block": block["key"], "run_id": out.name, "server": k,
           "server_session": f"{manifest['server_pid']}@{manifest['started_utc']}", "outcome": "process_error",
           "error": "interrupted before durable result", "elapsed_seconds": 0, "attempt_directory": str(directory)}
    directory.mkdir(parents=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["sim_gpu"]), "PYTHONDONTWRITEBYTECODE": "1",
           "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json", "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
           "PYTHONPATH": f"{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src"}
    command = [str(SIM_PYTHON), "-u", str(ECM / "scripts" / client), "--run-dir", str(out),
               "--server", str(k), "--task", block["task"], "--episode-id", str(block["episode_id"]),
               "--condition", condition, "--attempt-dir", str(directory)]
    if schedule is not None:
        atomic_json(directory / "rs_schedule.json", schedule)
        command += ["--rs-schedule", str(directory / "rs_schedule.json")]
    atomic_json(directory / "started.json", {"command": command, "time_utc": timestamp()})
    start = time.monotonic()
    with (directory / "console.log").open("x", buffering=1) as log:
        child = subprocess.Popen(command, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout_s)
            row["error"] = f"process exit={code} without durable result"
        except subprocess.TimeoutExpired:
            row.update(outcome="wall_timeout", error=f"{timeout_s}s watchdog")
        finally:
            va.stop_child(child)
    row["elapsed_seconds"] = round(time.monotonic() - start, 3)
    atomic_json(directory / "exit.json", {"returncode": child.returncode, "elapsed_seconds": row["elapsed_seconds"]})
    if not (directory / "result.json").exists():
        atomic_json(directory / "result.json", row)
    return {**read_json(directory / "result.json"), "block": block["key"]}


def run_block(out, route, block, rt, lock):
    """`route`: {condition: server} of one lane (model seed, replica)."""
    rows = []
    for condition in block["order"]:
        if (out / "stop_requested").exists():
            break
        schedule = None
        if condition == "RS":
            schedule = to.rs_schedule(read_jsonl(episode_dir(out, "TO", block) / "oracle_decisions.jsonl"))
        k = route[condition]
        row = run_episode(out, k, block, condition, rt["episode_wall_timeout_s"], schedule)
        with lock:
            append_jsonl(out / f"server{k}/attempts.jsonl", row)
        rows.append(row)
    return rows


def worker(out, route, blocks, rt, lock):
    errors = 0
    while not (out / "stop_requested").exists():
        try:
            block = blocks.get_nowait()
        except queue.Empty:
            return 0
        for k in set(route.values()):
            atomic_json(out / f"server{k}/active.json", {**block, "started_utc": timestamp()})
        for row in run_block(out, route, block, rt, lock):
            errors = errors + 1 if row["outcome"] not in VALID else 0
            if errors >= rt["max_consecutive_errors"]:
                (out / "stop_requested").touch()
                return 1
    return 2


def refresh(out, status, protocol, plan):
    done = sum(block_complete(protocol, block_results(out, protocol, b)) for b in plan)
    atomic_json(out / "progress.json", {"status": status, "updated_utc": timestamp(), "supervisor_pid": os.getpid(),
                                        "planned_blocks": len(plan), "complete_blocks": done})


def run(out):
    with (out / "controller.lock").open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (out / "stop_requested").unlink(missing_ok=True)
        return _run(out)


def _run(out):
    protocol = verify(out)
    rt = protocol["params"]["runtime"]
    plan = read_json(out / "plan.json")
    supersede_partial(out, protocol, plan)
    pending = [b for b in plan if not block_complete(protocol, block_results(out, protocol, b))]
    if not pending:
        refresh(out, "completed", protocol, plan)
        return 0
    configs = sorted(out.glob("server*/config.json"), key=lambda p: int(p.parent.name[6:]))
    lanes = {}
    for k, path in enumerate(configs):
        cfg = read_json(path)
        lanes.setdefault((cfg["model_seed"], cfg["replica"]), {}).update({c: k for c in cfg["conditions"]})
    queues = {s: queue.Queue() for s, _ in lanes}
    for block in pending:
        queues[block["model_seed"]].put(block)
    servers, threads, state = [], [], "loading_servers"
    signal.signal(signal.SIGTERM, lambda *_: (out / "stop_requested").touch())
    try:
        for k in range(len(configs)):
            servers.append(start_server(out, k))
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
                    atomic_json(out / f"server{k}/server_manifest.json", m)
                except OSError:
                    pass
            assert time.monotonic() < deadline, "server loading timeout"
            refresh(out, state, protocol, plan)
            time.sleep(5)
        state, lock, results = "running", threading.Lock(), {}

        def target(lane):
            results[lane] = worker(out, lanes[lane], queues[lane[0]], rt, lock)

        threads = [threading.Thread(target=target, args=(lane,), daemon=True) for lane in lanes]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            refresh(out, state, protocol, plan)
            if any(c.poll() is not None for c, _, _ in servers):
                raise RuntimeError("a policy server exited during the run")
            time.sleep(10)
        complete = all(block_complete(protocol, block_results(out, protocol, b)) for b in plan)
        state = "completed" if complete and all(r == 0 for r in results.values()) else "stopped_incomplete"
    except BaseException:
        atomic_json(out / "controller_error.json", {"error": traceback.format_exc(), "time_utc": timestamp()})
        (out / "stop_requested").touch()
        state = "error"
    finally:
        for t in threads:
            t.join()
        for k, (child, log, m) in enumerate(servers):
            va.stop_child(child)
            m.update(status="stopped", stopped_utc=timestamp(), server_exit_code=child.returncode)
            atomic_json(out / f"server{k}/server_manifest.json", m)
            log.close()
        refresh(out, state, protocol, plan)
    return 0 if state == "completed" else 1


# --------------------------------------------------------------------------- analysis

def action_hashes(directory):
    return [r["actions_sha256"] for r in read_jsonl(Path(directory) / "policy_actions.jsonl")]


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def expert_l2(directory, branch_rows, arm=7, steps=16):
    out = []
    for r in branch_rows:
        expert = r.get("expert") or {}
        if r.get("status") != "ok" or not expert.get("chunk"):
            continue
        path = Path(directory) / "actions" / f"decision_{r['decision']:04d}.npz"
        if not path.exists():
            continue
        policy = np.load(path)["actions"][:, :arm]
        chunk = np.asarray(expert["chunk"], dtype=np.float64)[:, :arm]
        n = min(steps, len(chunk), len(policy), r.get("valid_steps") or len(chunk))
        if n:
            out.append(float(np.linalg.norm(policy[:n] - chunk[:n], axis=1).mean()))
    return out


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def judge(res, margin, alpha):
    lo95, hi95 = res["ci95"]
    lo90, hi90 = res["ci90"]
    if lo95 > 0 and res["p_holm"] < alpha:
        return "improved"
    if hi95 < 0 and res["p_holm"] < alpha:
        return "degraded"
    if lo90 > -margin and hi90 < margin:
        return "bounded"
    return "undetermined"


def per_task_family(boot, tasks, pairs, margin, alpha):
    """One Holm family: every contrast of `pairs` within every task."""
    family = {}
    for t in tasks:
        for name, (a, b) in pairs.items():
            res = boot.contrast(a, b, [t])
            if res is not None:
                family[f"{name}/{t}"] = {"name": name, "task": t, "a": a, "b": b, **res}
    adjusted = ava.holm({n: c["p_two_sided"] for n, c in family.items()})
    for n, c in family.items():
        c["p_holm"] = adjusted[n]
        c["state"] = judge(c, margin, alpha)
    return family


def route(gates, states):
    if not all(gates.values()):
        return "工程无效：有效性门槛未全部通过，结果只作工程诊断"
    c1, c2 = states.get("C1"), states.get("C2")
    if c1 == "improved" and c2 == "improved":
        return "有闭环空间"
    if c1 == "improved":
        return "只是扰动效应"
    if c1 == "bounded":
        return "没有可见空间"
    return "未定（按预先写定的扩展加种子 10、11）"


def ablation_route(gates, states, tasks, control):
    if not all(gates.values()):
        return "工程无效：有效性门槛未全部通过，结果只作工程诊断"
    if control in tasks and states.get(f"A2/{control}") != "degraded":
        return f"诊断无效：阳性对照 {control} 在只留当前帧（NM）时没有退化，先排查实现"
    verdicts = []
    for t in tasks:
        if t == control:
            continue
        a1, a2 = states.get(f"A1/{t}"), states.get(f"A2/{t}")
        if a1 == "degraded":
            verdicts.append(f"{t}：长历史被使用")
        elif a1 == "bounded" and a2 == "degraded":
            verdicts.append(f"{t}：只用近期和锚点帧，长历史无关")
        elif a2 == "bounded":
            verdicts.append(f"{t}：看不出策略使用记忆")
        else:
            verdicts.append(f"{t}：未定（按预先写定的扩展加种子 10、11）")
    return "；".join(verdicts)


def budget_route(gates, per_condition, task, rule):
    if not all(gates.values()):
        return "工程无效：有效性门槛未全部通过，结果只作工程诊断"
    rates = {c: p["success_rate"] for c, p in per_condition[task].items()}
    base = rates["U0"]
    for c in rule["order"]:
        if base - rates[c] >= rule["min_drop"] - 1e-9 and rates[c] >= rule["min_success"] - 1e-9:
            return f"选定 {to.BUDGETS[c]} 帧（{c}）：U0 {100 * base:.1f}%，{c} {100 * rates[c]:.1f}%"
    return "两个预算都不满足条件，不跑老师：缩小预算没有找到适合老师检验的设置"


def event_route(gates, states, tasks):
    if not all(gates.values()):
        return "工程无效：有效性门槛未全部通过，结果只作工程诊断"
    verdicts = []
    for t in tasks:
        e1, e2 = states.get(f"E1/{t}"), states.get(f"E2/{t}")
        if e1 == "improved" and e2 == "improved":
            verdicts.append(f"{t}：事件记忆优于官方选择，也优于同一采样器的无事件版本")
        elif e1 == "improved":
            verdicts.append(f"{t}：优于官方选择，但不优于无事件版本，收益不能归因于事件信号")
        elif e1 == "degraded":
            verdicts.append(f"{t}：事件记忆差于官方选择")
        elif e1 == "bounded":
            verdicts.append(f"{t}：没有可见提升")
        else:
            verdicts.append(f"{t}：未定")
    return "；".join(verdicts)


def finetune_route(gates, states, tasks, contrasts, per_condition=None, floor=None, effect=None):
    """Per-task verdict; with `floor`, a task whose compared conditions all succeed at most that often is not
    comparable. `effect` = {"name", "a", "b", "states": {"<name>/<task>": state}} appends a secondary contrast."""
    if not all(gates.values()):
        return "工程无效：有效性门槛未全部通过，结果只作工程诊断"
    compared = {c for pair in contrasts.values() for c in pair}
    verdicts = []
    for t in tasks:
        rates = [p["success_rate"] for c, p in (per_condition or {}).get(t, {}).items() if c in compared]
        if floor is not None and rates and all(r is not None and r <= floor + 1e-9 for r in rates):
            verdicts.append(f"{t}：地板效应，所有条件成功率都 ≤ {100 * floor:g}%，无法比较")
            continue
        named = {n: states.get(f"{n}/{t}") for n in contrasts}
        better = [contrasts[n][0] for n, s in named.items() if s == "improved"]
        worse = [contrasts[n][0] for n, s in named.items() if s == "degraded"]
        if better:
            verdicts.append(f"{t}：{'、'.join(better)} 优于同样微调的均匀记忆 K0"
                            + (f"，{'、'.join(worse)} 差于 K0" if worse else ""))
        elif worse:
            verdicts.append(f"{t}：{'、'.join(worse)} 差于 K0，没有 k 优于 K0")
        elif all(s == "bounded" for s in named.values()):
            count = {2: "两", 3: "三"}.get(len(named), str(len(named)))
            verdicts.append(f"{t}：{count}个 k 与 K0 的差都小于 δ，事件关键帧没有可见作用")
        else:
            verdicts.append(f"{t}：未定")
    text = "；".join(verdicts)
    if effect:
        named = "，".join(f"{t} {STATE_LABELS.get(effect['states'].get(effect['name'] + '/' + t), '无结果')}"
                          for t in tasks)
        text += f"。次要对比 {effect['name']} = {effect['a']} − {effect['b']}：{named}"
    return text


def effect_of(stats, secondary):
    name = stats.get("route", {}).get("effect")
    if not name:
        return None
    a, b = stats["secondary"][name]
    return {"name": name, "a": a, "b": b, "states": {k: c["state"] for k, c in secondary.items()}}


def event_diagnostics(records, se_mult):
    """Per condition: frames changed vs U and frames near the target (median per decision); for ES/TE the
    paired gain of each event memory over U (median ḡ, ḡ/se, share above se_mult·se) and TE's choices."""
    out = {}
    for c in sorted({r["condition"] for r in records}):
        rows = [d for r in records if r["condition"] == c for d in r["oracle"] if d.get("event")]
        if not rows:
            continue
        entry = out[c] = {k: median(d["event"][k] for d in rows) for k in ("changed", "near", "near_uniform")}
        gains = collections.defaultdict(list)
        for d in rows:
            for name, g in (d.get("gains") or {}).items():
                gains[name].append(g)
        if gains:
            entry["gains"] = {n: {"decisions": len(v), "median_gain": median(g["mean_gain"] for g in v),
                                  "median_gain_over_se": median(g["mean_gain"] / g["se"] for g in v if g["se"]),
                                  "share_above": sum(g["mean_gain"] > se_mult * g["se"] for g in v) / len(v)}
                              for n, v in sorted(gains.items())}
        choices = collections.Counter(d["choice"]["name"] for d in rows if d.get("swapped") and "name" in d["choice"])
        if choices:
            entry["choices"] = dict(choices)
    return out


def selection_audit(records, capacity=32, event=None, keyframes=None):
    """Every decision fed the model exactly the frames its condition defines: U (`capacity` frames) for U0,
    the ablation memory for NH/NM/B16/B8, U with the logged swap applied for S0/TO/AO/RS, and for
    EV/EVR/EVA/TE and K0-K12 the event or event-keyframe memory recomputed from the logged event signal."""
    cfg = em.EventMemoryConfig.from_params(event) if event else None
    checked, bad = 0, []
    for r in records:
        oracle = {int(o["decision"]): o for o in r["oracle"]}
        signal = read_jsonl(Path(r["attempt_directory"]) / "event_signal.jsonl") if cfg else []
        for row in read_jsonl(Path(r["attempt_directory"]) / "policy_decisions.jsonl"):
            decision, max_id = int(row["decision_idx"]), int(row["available_max_frame_id"])
            part = partition(max_id, capacity)
            expected = list(part.uniform)
            o = oracle.get(decision) or {}
            memories = distances = None
            if cfg and (r["condition"] in to.EVENT | to.KEYFRAME or o.get("swapped") and r["condition"] in to.EVENT_SCORED):
                distances = em.decode_distances([s for s in signal if s["decision"] <= decision], max_id)
            if distances is not None and r["condition"] not in to.KEYFRAME:
                memories = em.strategy_memories(part, distances, cfg.for_task(r["task"]), to.stable_seed(
                    r["task"], r["episode_id"], r["model_seed"], decision, "evr"))
            if r["condition"] in to.ABLATIONS:
                expected = to.ablate(r["condition"], part)
            elif r["condition"] in to.KEYFRAME:
                expected = ek.select(distances, max_id, ek.KeyframeConfig(**keyframes[r["condition"]]))[0]
            elif r["condition"] in to.EVENT:
                expected = memories[r["condition"]]
            elif r["condition"] != "U0":
                if o.get("uniform") != expected:
                    expected = None
                elif o.get("swapped") and r["condition"] in to.EVENT_SCORED:
                    expected = memories[o["choice"]["name"]]
                elif o.get("swapped"):
                    expected = sorted((set(expected) - {o["choice"]["replaced"]}) | {o["choice"]["frame"]})
            checked += 1
            if (expected is None or row["selected_frame_ids"] != expected
                    or row["valid_tokens"] != to.TOKENS_PER_FRAME * len(expected)):
                bad.append({"block": r["block"], "condition": r["condition"], "decision": row["decision_idx"]})
    return {"decisions": checked, "mismatches": len(bad), "examples": bad[:5]}


def collect(out, protocol, plan):
    blocks, records = [], []
    for block in plan:
        rows = block_results(out, protocol, block)
        complete = block_complete(protocol, rows)
        blocks.append({"key": block["key"], "complete": complete, "conditions": sorted(rows)})
        if not complete:
            continue
        for c, r in rows.items():
            d = Path(r["attempt_directory"])
            records.append({**r, "oracle": read_jsonl(d / "oracle_decisions.jsonl"),
                            "branch": read_jsonl(d / "teacher_branch.jsonl"), "hashes": action_hashes(d)})
    return blocks, records


def analyze(out):
    protocol = read_json(out / "protocol.json")
    params, plan = protocol["params"], read_json(out / "plan.json")
    tasks, stats, gates_cfg = task_list(params), params["stats"], params["gates"]
    blocks, records = collect(out, protocol, plan)
    by = {(r["task"], r["episode_id"], r["model_seed"], r["condition"]): r for r in records}
    scored = [r for r in records if r["condition"] in to.SCORED]
    decisions = [d for r in scored for d in r["oracle"]]
    queries = [q for r in scored for q in r["branch"]]
    audits = [d["audits"] for d in decisions if d.get("audits")]
    errors = [r for r in records if r["outcome"] not in VALID]
    g0_pairs, g0_bad = 0, []
    for (t, i, s, c), r in by.items():
        if c not in to.SHADOWS or (t, i, s, "U0") not in by:
            continue
        u = by[(t, i, s, "U0")]
        if r["outcome"] not in VALID or u["outcome"] not in VALID:
            continue
        g0_pairs += 1
        if (r["hashes"], r["outcome"], r.get("executed_steps")) != (u["hashes"], u["outcome"], u.get("executed_steps")):
            g0_bad.append({"task": t, "episode_id": i, "model_seed": s, **({"condition": c} if c != "S0" else {}),
                           "first_divergence": first_divergence(r["hashes"], u["hashes"])})
    available = [d.get("expert_status") == "ok" for d in decisions]
    rates = {"error_rate": len(errors) / len(records) if records else None,
             "expert_available": sum(available) / len(available) if available else None}
    conditions = set(params["conditions"])
    scope = stats.get("scope", "pooled")
    gates = {}
    if conditions & to.SHADOWS:
        gates["G0_noninvasive"] = g0_pairs > 0 and not g0_bad
    if conditions & to.SCORED:
        gates["G1_branch_start"] = bool(queries) and all(q.get("start_ok") for q in queries)
        gates["G2_scoring_audit"] = bool(audits) and all(a["passed"] for a in audits)
    gates["G3_errors"] = rates["error_rate"] is not None and rates["error_rate"] <= gates_cfg["max_error_rate"]
    if conditions & to.SCORED:
        gates["G4_expert_available"] = (rates["expert_available"] or 0) >= gates_cfg["expert_available_min"]
    gates["G5_same_process"] = bool(records) and all(b["complete"] for b in blocks)
    selection = None
    capacity, event = params["oracle"].get("capacity", 32), params["oracle"].get("event")
    if conditions & to.ABLATIONS or capacity != 32 or event:
        selection = selection_audit(records, capacity, event, params["oracle"].get("keyframes"))
        gates["GA_selection"] = selection["decisions"] > 0 and selection["mismatches"] == 0
    outcomes = {(r["task"], r["episode_id"], r["model_seed"], r["condition"]): r["outcome"] for r in records
                if r["outcome"] in VALID}
    margin = stats["margin_pp"] / 100
    contrasts, states, per_task_contrasts, secondary = {}, {}, {}, {}
    if all(len({i for (tt, i, _, _) in outcomes if tt == t}) > 1 for t in tasks):
        boot = ava.PairedBootstrap(outcomes, tasks, stats["bootstrap_draws"], seed=stats["bootstrap_seed"])
        if scope == "per_task":
            contrasts = per_task_family(boot, tasks, stats["contrasts"], margin, stats["alpha"])
        else:
            for name, (a, b) in stats["contrasts"].items():
                res = boot.contrast(a, b)
                if res is not None:
                    contrasts[name] = {"a": a, "b": b, **res}
            adjusted = ava.holm({n: c["p_two_sided"] for n, c in contrasts.items()})
            for n, c in contrasts.items():
                c["p_holm"] = adjusted[n]
                c["state"] = judge(c, margin, stats["alpha"])
            if len(tasks) > 1:
                per_task_contrasts = per_task_family(boot, tasks, stats["contrasts"], margin, stats["alpha"])
        states = {n: c["state"] for n, c in contrasts.items()}
        secondary = per_task_family(boot, tasks, stats.get("secondary", {}), margin, stats["alpha"])
    per_condition = {}
    for t in tasks:
        per_condition[t] = {}
        for c in params["conditions"]:
            own = [r for r in records if r["task"] == t and r["condition"] == c]
            rows = [d for r in own for d in r["oracle"]]
            p = per_condition[t][c] = {
                "episodes": len(own), "valid": sum(r["outcome"] in VALID for r in own),
                "success": sum(r["outcome"] == "success" for r in own),
                "outcomes": dict(collections.Counter(r["outcome"] for r in own)),
                "decisions": len(rows), "swapped": sum(bool(d.get("swapped")) for d in rows),
                "reasons": dict(collections.Counter(d.get("reason") for d in rows if d.get("reason"))),
                "median_steps": median(r.get("executed_steps") for r in own)}
            p["success_rate"] = p["success"] / p["valid"] if p["valid"] else None
            p["swap_rate"] = p["swapped"] / p["decisions"] if p["decisions"] else None
    accepted = collections.defaultdict(list)
    for r in records:
        for d in r["oracle"]:
            if d.get("swapped") and "frame" in (d.get("choice") or {}):
                accepted[r["condition"]].append(d)
    swaps = {c: {"n": len(ds),
                 "median_gain": median(d["choice"].get("mean_gain") for d in ds),
                 "median_gain_over_se": median(d["choice"]["mean_gain"] / d["choice"]["se"] for d in ds
                                               if d["choice"].get("se")),
                 "demo_frames": sum(d["choice"]["frame"] < d["exec_start_idx"] for d in ds),
                 "median_slot": median(d["choice"]["slot"] for d in ds)} for c, ds in accepted.items()}
    shadow = [d for r in records if r["condition"] == "S0" for d in r["oracle"] if d.get("extremes")]
    divergence = {}
    reference = next((c for c in ("S0", "U0", "K0") if c in conditions), "U0")
    for c in [c for c in params["conditions"] if c != reference and c not in ("U0", "S0")]:
        firsts = [first_divergence(by[(t, i, s, c)]["hashes"], r["hashes"]) for (t, i, s, cc), r in by.items()
                  if cc == reference and (t, i, s, c) in by]
        divergence[c] = {"identical": sum(f is None for f in firsts), "median_first": median(firsts)}
    diagnostics = {
        "swaps": swaps,
        "shadow_potential": {"decisions": len(shadow),
                             "TO_would_swap": sum(d["extremes"]["TO"]["accepted"] for d in shadow),
                             "AO_would_swap": sum(d["extremes"]["AO"]["accepted"] for d in shadow)},
        "expert_status": dict(collections.Counter(d.get("expert_status") for d in decisions)),
        "branch_seconds_median": median(q.get("seconds") for q in queries),
        "score_seconds_median": median((d.get("timing") or {}).get("score_s") for d in decisions),
        "expert_l2_median": {c: median(v for r in records if r["condition"] == c
                                       for v in expert_l2(r["attempt_directory"], r["branch"]))
                             for c in sorted(to.SCORED) if c not in to.EVENT_SCORED or c in conditions},
        f"divergence_from_{reference}": divergence,
        "selection_audit": selection,
        **({"event": event_diagnostics(records, params["oracle"]["se_mult"])} if event else {}),
        "audit_rel_err_max": max((a["cached_vs_full_rel_err"] for a in audits), default=None),
        "g0_mismatches": g0_bad, "g0_pairs": g0_pairs,
        "errors": [{"block": r["block"], "condition": r["condition"], "outcome": r["outcome"],
                    "error": (r.get("error") or "")[-600:]} for r in errors][:10],
        "branch_errors": [{"t": q.get("t"), "error": (q.get("error") or "")[-600:]} for q in queries if q.get("error")][:5],
    }
    result = {"formal": protocol["formal"], "generated_utc": timestamp(), "tasks": tasks, "gates": gates, "rates": rates,
              "contrasts": contrasts, "per_task_contrasts": per_task_contrasts,
              **({"secondary_contrasts": secondary} if "secondary" in stats else {}),
              "route": ("冒烟，不解读" if not protocol["formal"]
                        else budget_route(gates, per_condition, tasks[0], stats["route"])
                        if stats.get("route", {}).get("kind") == "budget"
                        else event_route(gates, states, tasks) if stats.get("route", {}).get("kind") == "event"
                        else finetune_route(gates, states, tasks, stats["contrasts"], per_condition,
                                            stats["route"].get("floor"), effect_of(stats, secondary))
                        if stats.get("route", {}).get("kind") == "finetune"
                        else ablation_route(gates, states, tasks, stats["control"]) if scope == "per_task"
                        else route(gates, states)),
              "per_condition": per_condition, "diagnostics": diagnostics,
              "blocks": {"planned": len(plan), "complete": sum(b["complete"] for b in blocks)}}
    atomic_json(out / "analysis.json", result)
    write_report(out, protocol, result)
    print(json.dumps({k: result[k] for k in ("gates", "rates", "route", "blocks")}, ensure_ascii=False, indent=2))
    return result


GATE_LABELS = (("G0_noninvasive", "G0 无干扰（S0 = U0）"), ("G1_branch_start", "G1 分支起点"),
               ("G2_scoring_audit", "G2 打分审计"), ("G3_errors", "G3 工程错误"),
               ("G4_expert_available", "G4 专家可用率"), ("G5_same_process", "G5 同进程"),
               ("GA_selection", "GA 选帧审计"))
STATE_LABELS = {"improved": "提升", "degraded": "退化", "bounded": "小于界限 δ", "undetermined": "未定"}


def pp(x):
    return "—" if x is None else f"{100 * x:+.1f}"


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def write_report(out, protocol, r):
    tasks = r.get("tasks") or [protocol["params"]["task"]]
    per_condition = r["per_condition"] if set(r["per_condition"]) == set(tasks) else {tasks[0]: r["per_condition"]}
    d = r["diagnostics"]
    ablation = protocol["params"].get("stats", {}).get("scope") == "per_task"
    title = {"ablation": "记忆消融诊断", "budget": "记忆预算标定",
             "budget16": "Teacher Oracle（16 帧记忆预算）", "event": "事件记忆反事实消融",
             "finetune": "事件关键帧微调", "fruit": "采摘事件关键帧微调（RM75-B）",
             "fruit_dart": "采摘事件关键帧微调，DART 示范（RM75-B）"}.get(
                 protocol["params"].get("stage"), "Teacher Oracle 第 3 步")
    lines = [f"# {title}：{' + '.join(tasks)} 闭环结果", "",
             f"状态：{'正式' if protocol['formal'] else '冒烟（非正式）'}运行；协议 `{protocol['protocol_doc']}`；"
             f"生成于 {r['generated_utc']}。完成 {r['blocks']['complete']}/{r['blocks']['planned']} 组。", "",
             f"判定：**{r['route']}**", "", "## 有效性门槛", "", "| 门槛 | 通过 |", "|---|---|"]
    lines += [f"| {label} | {'是' if r['gates'][k] else '否'} |" for k, label in GATE_LABELS if k in r["gates"]]
    if ablation:
        s = d.get("selection_audit") or {}
        lines += ["", f"工程错误率 {pct(r['rates']['error_rate'])}；选帧审计 {s.get('decisions')} 个决策，"
                  f"不一致 {s.get('mismatches')} 个。", ""]
    else:
        lines += ["", f"工程错误率 {pct(r['rates']['error_rate'])}；专家可用率 {pct(r['rates']['expert_available'])}；"
                  f"S0/U0 配对 {d['g0_pairs']} 组，不一致 {len(d['g0_mismatches'])} 组。", ""]
        if d.get("selection_audit"):
            s = d["selection_audit"]
            lines += [f"选帧审计 {s['decisions']} 个决策，不一致 {s['mismatches']} 个。", ""]
    def contrast_rows(items, label):
        rows = ["| 对比 | 差值 | 95% CI | 90% CI | Holm p | 状态 |", "|---|---|---|---|---|---|"]
        for n, c in items:
            rows.append(f"| {label(n, c)} = {c['a']} − {c['b']} | {pp(c['delta'])} | [{pp(c['ci95'][0])}, {pp(c['ci95'][1])}] | "
                        f"[{pp(c['ci90'][0])}, {pp(c['ci90'][1])}] | {c['p_holm']:.3f} | {STATE_LABELS[c['state']]} |")
        return rows + [""]

    if r["contrasts"] and ablation:
        lines += [f"## 主对比（各任务分别，{len(r['contrasts'])} 项 Holm 校正；成功率差，百分点）", ""]
        lines += contrast_rows(r["contrasts"].items(), lambda n, c: f"{c['task']} {c['name']}")
    elif r["contrasts"]:
        title = "## 主对比（成功率差，百分点）" if len(tasks) == 1 else "## 主对比（两个任务合并，任务等权；成功率差，百分点）"
        lines += [title, ""] + contrast_rows(r["contrasts"].items(), lambda n, c: n)
    if r.get("per_task_contrasts"):
        lines += ["## 各任务对比（次要结果，6 项 Holm 校正，不改变判定）", ""]
        lines += contrast_rows(r["per_task_contrasts"].items(), lambda n, c: f"{c['task']} {c['name']}")
    if r.get("secondary_contrasts"):
        lines += [f"## 次要对比（单独一族，{len(r['secondary_contrasts'])} 项 Holm 校正，不改变判定）", ""]
        lines += contrast_rows(r["secondary_contrasts"].items(), lambda n, c: f"{c['task']} {c['name']}")
    for t in tasks:
        if ablation:
            lines += [f"## 各条件：{t}", "", "| 条件 | 有效 episode | 成功率 | 决策数 | 执行步数中位数 | 结局 |",
                      "|---|---|---|---|---|---|"]
            for c, p in per_condition[t].items():
                ends = "，".join(f"{k} {v}" for k, v in sorted(p.get("outcomes", {}).items())) or "—"
                lines.append(f"| {c} | {p['valid']}/{p['episodes']} | {pct(p['success_rate'])} | {p['decisions']} | "
                             f"{p['median_steps']} | {ends} |")
        else:
            lines += [f"## 各条件：{t}", "", "| 条件 | 有效 episode | 成功率 | 决策数 | 替换率 | 执行步数中位数 | 未替换原因 |",
                      "|---|---|---|---|---|---|---|"]
            for c, p in per_condition[t].items():
                reasons = "，".join(f"{k} {v}" for k, v in sorted(p["reasons"].items())) or "—"
                lines.append(f"| {c} | {p['valid']}/{p['episodes']} | {pct(p['success_rate'])} | {p['decisions']} | "
                             f"{pct(p['swap_rate'])} | {p['median_steps']} | {reasons} |")
        lines.append("")
    lines += ["", "## 诊断（不参与判定）", ""]
    if d.get("event"):
        lines += [f"- 事件记忆（每决策中位数：相对 U 换掉的帧数、距目标 ≤ τ 的帧数；ES/TE 的配对收益）："
                  f"{json.dumps(d['event'], ensure_ascii=False)}"]
    if ablation:
        ref = next((k[len("divergence_from_"):] for k in d if k.startswith("divergence_from_")), "U0")
        lines += [f"- 选帧审计：{json.dumps(d.get('selection_audit'), ensure_ascii=False)}",
                  f"- 与 {ref} 的首次分叉：{json.dumps(d.get(f'divergence_from_{ref}'), ensure_ascii=False)}"]
    else:
        lines += [f"- 被接受的替换：{json.dumps(d['swaps'], ensure_ascii=False)}",
                  f"- S0 上的潜在替换：{json.dumps(d['shadow_potential'], ensure_ascii=False)}",
                  f"- 专家状态：{json.dumps(d['expert_status'], ensure_ascii=False)}",
                  f"- 分支耗时中位数 {d['branch_seconds_median']} s；打分耗时中位数 {d['score_seconds_median']} s；"
                  f"缓存与完整前向最大相对误差 {d['audit_rel_err_max']}",
                  f"- 策略前 16 步与专家块的关节 L2 中位数：{json.dumps(d['expert_l2_median'], ensure_ascii=False)}",
                  f"- 与 S0 的首次分叉：{json.dumps(d['divergence_from_S0'], ensure_ascii=False)}"]
    if d["errors"] or d["branch_errors"]:
        lines += ["", "## 错误", "", "```", json.dumps(d["errors"] + d["branch_errors"], ensure_ascii=False, indent=1), "```"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "analyze", "all"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, required=True)
        if name in ("prepare", "all"):
            p.add_argument("--smoke", action="store_true", help="one block per task at seed 7; not formal")
            p.add_argument("--stage", choices=sorted(STAGES), default="pilot")
            p.add_argument("--checkpoint-run", type=Path, help="smoke only: fine-tuning run serving K0-K12")
            p.add_argument("--checkpoint-step", type=int, help="smoke only: its checkpoint step")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "prepare" or (args.command == "all" and not (out / "protocol.json").exists()):
        prepare(out, args.smoke, args.stage, args.checkpoint_run and args.checkpoint_run.resolve(), args.checkpoint_step)
    elif args.command == "all":
        existing = read_json(out / "protocol.json")
        if existing["formal"] == args.smoke:
            raise SystemExit("--smoke does not match the existing protocol.json")
        if existing["params"].get("stage", "pilot") != args.stage:
            raise SystemExit("--stage does not match the existing protocol.json")
    code = 0
    if args.command in ("run", "all"):
        code = run(out)
    if args.command in ("analyze", "all"):
        analyze(out)
    return code


if __name__ == "__main__":
    sys.exit(main())
