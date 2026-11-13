#!/usr/bin/env python3
"""Paired, non-formal K0 versus candidate closed-loop comparison on dev 200-219."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from v_a_common import ECM, POLICY, SIM, SIM_PYTHON, atomic_json, read_json, sha256, timestamp
from run_fruit_diagnosis import start_server, wait_ready

FORMAL = ECM / "results/fruit_dart_kf_eval_20261027"
ARCHIVE = ECM / "results/fruit_dart_dev_gate_20261027/episodes/K0"
DEV = ECM / "data/fruit_picking_dev/rm75b_eg2/train"
TASKS = ("FruitCount", "FruitOccludedRipe")


def target_metrics(directory: Path):
    last = None
    with (directory / "raw_frames.jsonl").open() as f:
        for line in f:
            if line.strip():
                last = line
    if last is None:
        raise ValueError(f"missing ground-truth frames: {directory}")
    gt = json.loads(last)["gt"]
    targets = set(gt["targets"])
    inside = gt["in_basket"]
    return {"correct_in_basket": sum(bool(inside[i]) for i in targets),
            "wrong_in_basket": sum(bool(v) for i, v in enumerate(inside) if i not in targets),
            "first_target_in_basket": bool(inside[gt["targets"][0]])}


def run_one(out: Path, server, gpu: int, task: str, episode: int):
    d = out / task / f"ep{episode}"
    if (d / "result.json").is_file():
        prior = read_json(d / "result.json")
        if prior.get("outcome") != "error":
            return prior
    d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONDONTWRITEBYTECODE": "1",
           "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
           "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
           "PYTHONPATH": f"{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src"}
    cmd = [str(SIM_PYTHON), "-u", str(ECM / "scripts/diagnose_fruit_episode.py"),
           "--port", str(server[1]["port"]), "--context", str(server[2]),
           "--root", str(DEV), "--task", task, "--episode-id", str(episode),
           "--condition", "K0", "--mode", "closed", "--attempt-dir", str(d)]
    with (d / "console.log").open("w") as log:
        code = subprocess.run(cmd, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT,
                              timeout=900).returncode
    row = read_json(d / "result.json")
    if code != 0 or row.get("error"):
        raise RuntimeError(f"development rollout failed: {d}")
    print(json.dumps({"model": out.name, "task": task, "episode": episode,
                      "outcome": row["outcome"], "basket": row.get("final_in_basket")}), flush=True)
    return row


def summarize(out: Path, episodes: list[int]):
    rows = []
    for task in TASKS:
        for episode in episodes:
            archive = ARCHIVE / task / f"ep{episode}_seed7/result.json"
            baseline_path = archive if archive.is_file() else out / "baseline" / task / f"ep{episode}/result.json"
            baseline, candidate = (read_json(baseline_path),
                                   read_json(out / "candidate" / task / f"ep{episode}/result.json"))
            candidate_dir = out / "candidate" / task / f"ep{episode}"
            rows.append({"task": task, "episode": episode,
                         "baseline": {k: baseline.get(k) for k in
                                      ("outcome", "final_detached", "final_in_basket", "executed_steps")} |
                                     target_metrics(baseline_path.parent),
                         "candidate": {k: candidate.get(k) for k in
                                       ("outcome", "final_detached", "final_in_basket", "executed_steps")} |
                                      target_metrics(candidate_dir)})
    result = {"formal": False, "development": True, "episodes": len(rows),
              "baseline_success": sum(r["baseline"]["outcome"] == "success" for r in rows),
              "candidate_success": sum(r["candidate"]["outcome"] == "success" for r in rows),
              "baseline_detached": sum(r["baseline"]["final_detached"] or 0 for r in rows),
              "candidate_detached": sum(r["candidate"]["final_detached"] or 0 for r in rows),
              "baseline_in_basket": sum(r["baseline"]["final_in_basket"] or 0 for r in rows),
              "candidate_in_basket": sum(r["candidate"]["final_in_basket"] or 0 for r in rows),
              "baseline_correct_in_basket": sum(r["baseline"]["correct_in_basket"] for r in rows),
              "candidate_correct_in_basket": sum(r["candidate"]["correct_in_basket"] for r in rows),
              "baseline_wrong_in_basket": sum(r["baseline"]["wrong_in_basket"] for r in rows),
              "candidate_wrong_in_basket": sum(r["candidate"]["wrong_in_basket"] for r in rows),
              "baseline_first_target_in_basket": sum(r["baseline"]["first_target_in_basket"] for r in rows),
              "candidate_first_target_in_basket": sum(r["candidate"]["first_target_in_basket"] for r in rows),
              "rows": rows, "generated_utc": timestamp()}
    atomic_json(out / "summary.json", result)
    return result


def serve(out: Path, gpu: int, params: dict, checkpoint: Path, port: int):
    local = json.loads(json.dumps(params))
    local["checkpoints"]["K0"] = str(checkpoint)
    child, cfg = start_server(out, 0, gpu, local, conditions=("K0",), base_port=port)
    wait_ready([(child, cfg)])
    return child, cfg, out / "server0/context.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--first", type=int, default=200)
    p.add_argument("--stop", type=int, default=220)
    a = p.parse_args()
    if not 200 <= a.first < a.stop <= 220:
        p.error("only development episodes 200-219 are allowed")
    out, checkpoint = a.out.resolve(), a.checkpoint.resolve()
    if not (checkpoint / "params/manifest.ocdbt").is_file():
        p.error("candidate checkpoint incomplete")
    params = read_json(FORMAL / "protocol.json")["params"]
    baseline_checkpoint = Path(params["checkpoints"]["K0"]).resolve()
    episodes = list(range(a.first, a.stop))
    plan = {"formal": False, "source_root": str(DEV), "baseline_checkpoint": str(baseline_checkpoint),
            "candidate_checkpoint": str(checkpoint),
            "candidate_checkpoint_manifest_sha256": sha256(checkpoint / "params/manifest.ocdbt"),
            "gpu": a.gpu, "tasks": list(TASKS), "episodes": episodes}
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").is_file():
        if read_json(out / "plan.json") != plan:
            p.error("existing development comparison plan differs")
    else:
        atomic_json(out / "plan.json", plan)
    pending = [(task, ep) for task in TASKS for ep in episodes
               if not (ARCHIVE / task / f"ep{ep}_seed7/result.json").is_file() and
               not (out / "baseline" / task / f"ep{ep}/result.json").is_file()]
    if pending:
        server = serve(out / "baseline_server", a.gpu, params, baseline_checkpoint, 28971)
        try:
            for task, episode in pending:
                run_one(out / "baseline", server, a.gpu, task, episode)
        finally:
            server[0].terminate()
            server[0].wait(120)
    server = serve(out / "candidate_server", a.gpu, params, checkpoint, 28972)
    try:
        for task in TASKS:
            for episode in episodes:
                run_one(out / "candidate", server, a.gpu, task, episode)
    finally:
        server[0].terminate()
        server[0].wait(120)
    print(json.dumps(summarize(out, episodes), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
