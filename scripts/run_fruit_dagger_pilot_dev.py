#!/usr/bin/env python3
"""Small, non-formal full closed-loop check of the K0 DAgger pilot on dev ep200-203."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from v_a_common import ECM, POLICY, SIM, SIM_PYTHON, atomic_json, read_json, sha256, timestamp
from run_fruit_diagnosis import start_server, wait_ready

FORMAL = ECM / "results/fruit_dart_kf_eval_20261027"
BASELINE = ECM / "results/fruit_dart_dev_gate_20261027/episodes/K0"
DEV = ECM / "data/fruit_picking_dev/rm75b_eg2/train"
TASKS = ("FruitCount", "FruitOccludedRipe")
EPISODES = (200, 201, 202, 203)


def summarize(out):
    rows = []
    for task in TASKS:
        for episode in EPISODES:
            pilot = read_json(out / task / f"ep{episode}/result.json")
            base = read_json(BASELINE / task / f"ep{episode}_seed7/result.json")
            rows.append({"task": task, "episode_id": episode,
                         "baseline": {k: base.get(k) for k in
                                      ("outcome", "final_detached", "final_in_basket", "executed_steps")},
                         "pilot": {k: pilot.get(k) for k in
                                   ("outcome", "final_detached", "final_in_basket", "executed_steps", "error")}})
    result = {"formal": False, "development": True, "episodes": len(rows),
              "baseline_success": sum(r["baseline"]["outcome"] == "success" for r in rows),
              "pilot_success": sum(r["pilot"]["outcome"] == "success" for r in rows),
              "baseline_detached": sum(r["baseline"]["final_detached"] or 0 for r in rows),
              "pilot_detached": sum(r["pilot"]["final_detached"] or 0 for r in rows),
              "baseline_in_basket": sum(r["baseline"]["final_in_basket"] or 0 for r in rows),
              "pilot_in_basket": sum(r["pilot"]["final_in_basket"] or 0 for r in rows),
              "rows": rows, "generated_utc": timestamp()}
    atomic_json(out / "summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ECM / "results/fruit_dagger_pilot_dev_20261030")
    p.add_argument("--checkpoint", type=Path, default=ECM /
                   "runs/fruit_dagger_pilot_8gpu_v2_20261030/ckpt/mme_vla_suite/k0_dagger_pilot/199")
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()
    out, checkpoint = a.out.resolve(), a.checkpoint.resolve()
    if not (checkpoint / "params/manifest.ocdbt").is_file():
        p.error("pilot checkpoint is incomplete")
    if not all((BASELINE / task / f"ep{e}_seed7/result.json").is_file()
               for task in TASKS for e in EPISODES):
        p.error("archived K0 dev comparison is incomplete")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        p.error(f"development eval already started: {out}")
    params = read_json(FORMAL / "protocol.json")["params"]
    params["checkpoints"]["K0"] = str(checkpoint)
    atomic_json(out / "plan.json", {"formal": False, "checkpoint": str(checkpoint),
                                    "checkpoint_manifest_sha256": sha256(checkpoint / "params/manifest.ocdbt"),
                                    "baseline": str(BASELINE), "tasks": TASKS, "episodes": EPISODES,
                                    "gpu": a.gpu, "started_utc": timestamp()})
    server = start_server(out, 0, a.gpu, params, conditions=("K0",), base_port=28951)
    try:
        wait_ready([server])
        for task in TASKS:
            for episode in EPISODES:
                d = out / task / f"ep{episode}"
                d.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu),
                       "PYTHONDONTWRITEBYTECODE": "1", "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
                       "MPLCONFIGDIR": str(ECM / "cache/matplotlib"),
                       "PYTHONPATH": f"{POLICY}/examples/robomme:{POLICY}/packages/openpi-client/src:{SIM}/src"}
                cmd = [str(SIM_PYTHON), "-u", str(ECM / "scripts/diagnose_fruit_episode.py"),
                       "--port", str(server[1]["port"]), "--context", str(out / "server0/context.json"),
                       "--root", str(DEV), "--task", task, "--episode-id", str(episode),
                       "--condition", "K0", "--mode", "closed", "--attempt-dir", str(d)]
                with (d / "console.log").open("x") as log:
                    subprocess.run(cmd, cwd=ECM, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                row = read_json(d / "result.json")
                print(json.dumps({"task": task, "episode": episode, "outcome": row["outcome"],
                                  "detached": row.get("final_detached"),
                                  "basket": row.get("final_in_basket")}, ensure_ascii=False), flush=True)
    finally:
        server[0].terminate()
        server[0].wait(120)
    print(json.dumps(summarize(out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
