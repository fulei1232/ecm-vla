#!/usr/bin/env python3
"""Event-keyframe fine-tuning: seal, train the arms, check.

Studies: robomme (reports/finetune_kf_protocol_20261025.md), fruit (reports/fruit_finetune_kf_protocol_20261025.md,
RM75-B fruit picking with the norm stats of its own dataset, <dataset>/assets) and fruit_dart (the same on clean +
DART demonstrations, reports/fruit_dart_protocol_20261027.md).

    python3 scripts/run_finetune_kf.py prepare --out runs/finetune_kf_20261025 --dataset data/finetune_kf/robomme3_train_gt
    python3 scripts/run_finetune_kf.py prepare --study fruit --out runs/fruit_finetune_kf_20261025 \
        --dataset data/finetune_kf/fruit_rm75b_train
    python3 scripts/run_finetune_kf.py train --out runs/finetune_kf_20261025
    python3 scripts/run_finetune_kf.py gate --out runs/finetune_kf_20261025     # training gate T1
    python3 scripts/run_finetune_kf.py status --out runs/finetune_kf_20261025

`prepare` writes one specification per arm (arms/<arm>.json) and protocol.json with the hashes of the
protocol document, the dataset build (and norm stats), the arm specifications and the code. `train` verifies them and
runs every unfinished arm on its assigned GPU group (two GPUs for earlier studies, four for fruit_dart),
the two groups in parallel and the arms of a group one after another; an arm without its final checkpoint
is moved to superseded/ and restarted from step 0 (the official loop saves no optimizer state).
`--smoke` seals a non-formal run with a few steps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_a_common import (CHECKPOINT, ECM, OPENPI_DATA_HOME, POLICY, POLICY_PYTHON, atomic_json, git_head,  # noqa: E402
                        read_json, sha256, timestamp)

SCRIPT = Path(__file__).resolve()
CODE = [SCRIPT, ECM / "scripts/finetune_event_keyframes.py", ECM / "scripts/build_finetune_dataset.py",
        ECM / "scripts/v_a_common.py", ECM / "src/memory_management/event_keyframes.py",
        ECM / "src/memory_management/validation/event_memory.py",
        POLICY / "scripts/train.py", POLICY / "src/mme_vla_suite/training/dataset.py",
        POLICY / "src/mme_vla_suite/training/dataloader.py", POLICY / "src/mme_vla_suite/training/config.py",
        POLICY / "src/mme_vla_suite/shared/mem_buffer.py", POLICY / "src/mme_vla_suite/shared/data_utils.py",
        POLICY / "src/mme_vla_suite/models/config/robomme/perceptual-framesamp-modul.yaml"]
SAMPLER = {"frames": 16, "radius": 0.03, "tol": 0.005}
TRAIN = {"steps": 6000, "warmup_steps": 500, "peak_lr": 2.5e-5, "decay_lr": 2.5e-6, "batch_size": 64,
         "fsdp_devices": 2, "num_workers": 12, "seed": 42, "ema_decay": 0.999, "log_interval": 50}
# "own_assets": train and serve with <dataset>/assets/robomme/norm_stats.json instead of the init checkpoint's.
STUDIES = {
    "robomme": {"name": "finetune_kf", "doc": ECM / "reports/finetune_kf_protocol_20261025.md",
                "arms": {"k0": 0, "k4": 4, "k8": 8, "k12": 12}, "train": {}, "own_assets": False,
                "code": [ECM / "scripts/run_finetune_kf_pipeline.sh"]},
    "fruit": {"name": "fruit_finetune_kf", "doc": ECM / "reports/fruit_finetune_kf_protocol_20261025.md",
              "arms": {"k0": 0, "k4": 4, "k8": 8}, "train": {"steps": 12000}, "own_assets": True,
              "code": [ECM / "scripts/run_fruit_finetune_pipeline.sh", ECM / "scripts/build_fruit_dataset.py",
                       ECM / "src/memory_management/fruit_distance.py"]},
    "fruit_dart": {"name": "fruit_dart_kf", "doc": ECM / "reports/fruit_dart_protocol_20261027.md",
                   "arms": {"k0": 0, "k4": 4, "k8": 8},
                   "train": {"steps": 12000, "fsdp_devices": 4},
                   "gpu_groups": ([0, 1, 2, 3], [4, 5, 6, 7]), "own_assets": True,
                   "code": [ECM / "scripts/run_fruit_dart_pipeline.sh", ECM / "scripts/build_fruit_dataset.py",
                            ECM / "scripts/run_build_fruit_dart_dataset.sh", ECM / "scripts/generate_fruit_dart.py",
                            ECM / "scripts/run_fruit_dev_gate.py",
                            ECM / "src/memory_management/fruit_distance.py"]},
}
GPU_PAIRS = ([0, 1], [2, 3])
SMOKE_STEPS = 100


def norm_stats_file(assets):
    return Path(assets) / "robomme/norm_stats.json"


def final_step(spec):
    return spec["train"]["steps"] - 1


def arm_dir(out, arm):
    return out / "ckpt/mme_vla_suite" / arm


def arm_complete(out, spec):
    step = arm_dir(out, spec["arm"]) / str(final_step(spec))
    return (step / "_CHECKPOINT_METADATA").exists() and (step / "params/manifest.ocdbt").exists() \
        and (step / "assets/robomme/norm_stats.json").exists()


def tree(root):
    return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()}


def compare_copy(source, copy, sampled=400, seed=20261025):
    """Same files and sizes; byte-identical metadata, distance traces and a random sample of the rest."""
    a, b = tree(source), tree(copy)
    problems = [f"missing in copy: {k}" for k in sorted(a.keys() - b.keys())]
    problems += [f"extra in copy: {k}" for k in sorted(b.keys() - a.keys())]
    problems += [f"size differs: {k}" for k in sorted(a.keys() & b.keys()) if a[k] != b[k]]
    common = sorted(a.keys() & b.keys())
    small = [k for k in common if k.startswith("meta/") or k.endswith("distance.npy")]
    rest = sorted(set(common) - set(small))
    rng = np.random.default_rng(seed)
    hashed = small + [rest[i] for i in sorted(rng.choice(len(rest), size=min(sampled, len(rest)), replace=False))]
    problems += [f"content differs: {k}" for k in hashed if sha256(source / k) != sha256(copy / k)]
    return {"source": str(source), "files": len(a), "bytes": sum(a.values()), "hashed_files": len(hashed),
            "sample_seed": seed, "passed": not problems, "problems": problems[:50]}


def prepare(out, dataset, smoke, source=None, study="robomme"):
    if (out / "protocol.json").exists():
        raise SystemExit(f"{out} is already prepared; use a new directory")
    s = STUDIES[study]
    build = read_json(dataset / "meta/build.json")
    if not build.get("passed"):
        raise SystemExit("the dataset build did not pass its checks")
    copy = compare_copy(source, dataset) if source else None
    if copy and not copy["passed"]:
        raise SystemExit(f"the dataset copy differs from its source: {copy['problems'][:5]}")
    assets = dataset / "assets" if s["own_assets"] else None
    if assets and not norm_stats_file(assets).exists():
        raise SystemExit(f"{study} trains with the dataset's own norm stats; {norm_stats_file(assets)} is missing")
    train = dict(TRAIN, **s["train"])
    if smoke:
        train.update(steps=SMOKE_STEPS, warmup_steps=5, log_interval=5)
    (out / "arms").mkdir(parents=True)
    specs = {}
    gpu_groups = s.get("gpu_groups", GPU_PAIRS)
    for i, (arm, k) in enumerate(s["arms"].items()):
        specs[arm] = {"arm": arm, "sampler": {**SAMPLER, "events": k}, "dataset": str(dataset),
                      "init_checkpoint": str(CHECKPOINT), "checkpoint_base_dir": str(out / "ckpt"),
                      "train": train, "gpus": gpu_groups[i % len(gpu_groups)],
                      **({"assets_dir": str(assets)} if assets else {})}
        atomic_json(out / f"arms/{arm}.json", specs[arm])
    atomic_json(out / "protocol.json", {
        "name": s["name"], "study": study, "formal": not smoke, "created_utc": timestamp(),
        "protocol_doc": str(s["doc"]), "protocol_doc_sha256": sha256(s["doc"]),
        "dataset": str(dataset), "dataset_build_sha256": sha256(dataset / "meta/build.json"),
        "dataset_episodes_sha256": sha256(dataset / "meta/episodes.json"), "dataset_copy_check": copy,
        "init_checkpoint": str(CHECKPOINT), "init_params_manifest_sha256": sha256(CHECKPOINT / "params/manifest.ocdbt"),
        "init_norm_stats_sha256": sha256(CHECKPOINT / "assets/robomme/norm_stats.json"),
        **({"assets_dir": str(assets), "assets_norm_stats_sha256": sha256(norm_stats_file(assets))} if assets else {}),
        "arms": {arm: {"spec_sha256": sha256(out / f"arms/{arm}.json"), "events": k} for arm, k in s["arms"].items()},
        "code_identity": [{"path": str(p), "sha256": sha256(p)} for p in CODE + s["code"]],
        "policy_sha": git_head(POLICY), "ecm_sha": git_head(ECM),
        "pairing": "arms share initialisation, data order (loader seed), flow-matching noise (train seed) and schedule",
        "restart_policy": "an arm without its final checkpoint is superseded and retrained from step 0"})


def verify(out):
    p = read_json(out / "protocol.json")
    assert sha256(p["protocol_doc"]) == p["protocol_doc_sha256"], "protocol document changed since seal"
    assert sha256(Path(p["dataset"]) / "meta/build.json") == p["dataset_build_sha256"], "dataset build changed"
    if p.get("dataset_copy_check"):
        check = compare_copy(Path(p["dataset_copy_check"]["source"]), Path(p["dataset"]))
        assert check["passed"], f"dataset copy differs from its source: {check['problems'][:5]}"
    if p.get("assets_dir"):
        assert sha256(norm_stats_file(p["assets_dir"])) == p["assets_norm_stats_sha256"], "norm stats changed since seal"
    for arm, row in p["arms"].items():
        assert sha256(out / f"arms/{arm}.json") == row["spec_sha256"], f"arm specification changed: {arm}"
    for row in p["code_identity"]:
        assert sha256(row["path"]) == row["sha256"], f"code changed since seal: {row['path']}"
    return p


def launch(out, spec):
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    if arm_dir(out, spec["arm"]).exists():
        dest = out / "superseded" / stamp / spec["arm"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(arm_dir(out, spec["arm"])), str(dest))
    (out / "logs").mkdir(exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, spec["gpus"])),
           "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.92",
           "OPENPI_DATA_HOME": str(OPENPI_DATA_HOME), "WANDB_MODE": "disabled", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": f"{POLICY}/src:{POLICY}/packages/openpi-client/src"}
    command = [str(POLICY_PYTHON), "-u", str(ECM / "scripts/finetune_event_keyframes.py"),
               "--spec", str(out / f"arms/{spec['arm']}.json")]
    log = (out / f"logs/{spec['arm']}_{stamp}.log").open("x", buffering=1)
    child = subprocess.Popen(command, cwd=POLICY, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return child, log


def train(out, arms):
    p = verify(out)
    specs = {arm: read_json(out / f"arms/{arm}.json") for arm in (arms or p["arms"])}
    queues = {}
    for s in specs.values():
        if not arm_complete(out, s):
            queues.setdefault(tuple(s["gpus"]), []).append(s)
    status = {"started_utc": timestamp(), "arms": {s["arm"]: "queued" for q in queues.values() for s in q}}
    lock = threading.Lock()

    def record(arm, state):
        with lock:
            status["arms"][arm] = state
            status["updated_utc"] = timestamp()
            atomic_json(out / "train_status.json", status)

    def run_queue(queue):
        for s in queue:
            child, log = launch(out, s)
            record(s["arm"], "running")
            code = child.wait()
            log.close()
            record(s["arm"], "complete" if code == 0 and arm_complete(out, s) else f"failed (exit {code})")

    atomic_json(out / "train_status.json", status)
    workers = [threading.Thread(target=run_queue, args=(q,)) for q in queues.values()]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return 0 if all(v == "complete" for v in status["arms"].values()) else 1


def same_norm_stats(a, b):
    x, y = read_json(a)["norm_stats"], read_json(b)["norm_stats"]

    def close(u, v):
        return (u is None) == (v is None) and (u is None or np.allclose(u, v, rtol=0, atol=1e-6))

    return x.keys() == y.keys() and all(x[k].keys() == y[k].keys() and all(close(x[k][q], y[k][q]) for q in x[k])
                                        for k in x)


def gate(out, only=None):
    """T1: every arm has its final checkpoint and every logged loss (interval means) of its last run is finite;
    with the dataset's own norm stats, every final checkpoint carries exactly those. `only`: check these arms
    (written to gate_t1_<arms>.json)."""
    p = verify(out)
    arms = {}
    for arm in only or p["arms"]:
        spec = read_json(out / f"arms/{arm}.json")
        logs = sorted((out / "logs").glob(f"{arm}_*.log"))
        text = logs[-1].read_text(errors="replace") if logs else ""
        losses = [float(x) for x in re.findall(r"Step \d+: .*?\bloss=([^,\s]+)", text)]
        expected = len(range(0, spec["train"]["steps"], spec["train"]["log_interval"]))
        arms[arm] = {"complete": arm_complete(out, spec), "log": str(logs[-1]) if logs else None,
                     "logged": len(losses), "expected_logged": expected,
                     "finite": len(losses) == expected and all(math.isfinite(x) for x in losses),
                     "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None}
        if spec.get("assets_dir"):
            saved = arm_dir(out, arm) / str(final_step(spec)) / "assets/robomme/norm_stats.json"
            arms[arm]["norm_stats_match"] = saved.exists() and same_norm_stats(saved, norm_stats_file(spec["assets_dir"]))
    passed = all(a["complete"] and a["finite"] and a.get("norm_stats_match", True) for a in arms.values())
    name = f"gate_t1_{'_'.join(only)}.json" if only else "gate_t1.json"
    atomic_json(out / name, {"passed": passed, "arms": arms, "time_utc": timestamp()})
    print(json.dumps({"passed": passed, **{a: (r["complete"], r["finite"], r["last_loss"]) for a, r in arms.items()}}))
    return 0 if passed else 1


def show(out):
    p = read_json(out / "protocol.json")
    for arm in p["arms"]:
        spec = read_json(out / f"arms/{arm}.json")
        logs = sorted((out / "logs").glob(f"{arm}_*.log"))
        last = [x for x in logs[-1].read_text(errors="replace").splitlines() if x.startswith("Step ")][-1:] if logs else []
        print(arm, "complete" if arm_complete(out, spec) else "incomplete", *last)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "train", "gate", "status"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-source", type=Path,
                        help="the dataset --dataset was copied from (e.g. to tmpfs); the copy is checked against it")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--study", choices=sorted(STUDIES), default="robomme", help="prepare only")
    parser.add_argument("--arms", nargs="+", choices=sorted({a for s in STUDIES.values() for a in s["arms"]}))
    a = parser.parse_args()
    out = a.out.resolve()
    if a.command == "prepare":
        prepare(out, a.dataset.resolve(), a.smoke, a.dataset_source.resolve() if a.dataset_source else None, a.study)
        return 0
    if a.command == "train":
        return train(out, a.arms)
    if a.command == "gate":
        return gate(out, a.arms)
    show(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
