#!/usr/bin/env python3
"""Development gate D of the fruit DART study (reports/fruit_dart_protocol_20261027.md, section 6).

After the k0 and k4 arms are trained and before k8: the DART-trained K0 and the clean K0 (C0) on the clean dev
episodes 200-219 of both tasks (seed 7), through the evaluation harness (run_teacher_oracle.py --stage fruit_dev).
Pass: K0 succeeds in more than 5 % of the 40 episodes and at most 2 episodes (of 80) end in an engineering error.
Writes <out>/dev_gate.json; exit 0 when passed, 1 otherwise (the pipeline stops before k8).

    python3 scripts/run_fruit_dev_gate.py --out results/fruit_dart_dev_gate_20261027
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from v_a_common import ECM, VALID, atomic_json, read_json, timestamp

DEV = ECM / "data/fruit_picking_dev/rm75b_eg2"
MIN_SUCCESS, MAX_ERRORS = 0.05, 2


def ensure_val_link():
    """eval_fruit_episode.py serves episodes from a root named val: the dev episodes are linked, not copied."""
    link = DEV / "val"
    if not link.exists():
        link.symlink_to("train", target_is_directory=True)
    assert link.resolve() == (DEV / "train").resolve(), f"{link} does not point at the dev episodes"


def verdict(out):
    protocol = read_json(out / "protocol.json")
    rows = [read_json(p) for p in sorted((out / "episodes").glob("*/*/*/result.json"))]
    planned = protocol["planned_blocks"] * len(protocol["params"]["conditions"])
    per = {}
    for c in protocol["params"]["conditions"]:
        mine = [r for r in rows if r["condition"] == c]
        per[c] = {"episodes": len(mine), "success": sum(r["outcome"] == "success" for r in mine),
                  "errors": sum(r["outcome"] not in VALID for r in mine),
                  "by_task": {t: sum(r["outcome"] == "success" for r in mine if r["task"] == t)
                              for t in protocol["params"]["tasks"]}}
        per[c]["success_rate"] = per[c]["success"] / len(mine) if mine else None
    errors = sum(p["errors"] for p in per.values())
    complete = len(rows) == planned
    passed = complete and per["K0"]["success_rate"] > MIN_SUCCESS and errors <= MAX_ERRORS
    return {"passed": passed, "complete": complete, "planned_episodes": planned, "recorded_episodes": len(rows),
            "conditions": per, "engineering_errors": errors,
            "rule": f"K0 success rate > {MIN_SUCCESS:g} over the dev episodes and engineering errors <= {MAX_ERRORS}",
            "protocol_sha256_of_run": protocol.get("protocol_doc_sha256"), "time_utc": timestamp()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="one dev episode per task; not a gate decision")
    parser.add_argument("--checkpoint-run", type=Path, help="smoke only: training run serving K0 (e.g. a smoke run)")
    parser.add_argument("--checkpoint-step", type=int, help="smoke only: its checkpoint step")
    a = parser.parse_args()
    if a.checkpoint_run and not a.smoke:
        raise SystemExit("--checkpoint-run is for smoke runs only")
    out = a.out.resolve()
    ensure_val_link()
    replace = ["--checkpoint-run", str(a.checkpoint_run.resolve()), "--checkpoint-step", str(a.checkpoint_step)] \
        if a.checkpoint_run else []
    command = [sys.executable, str(ECM / "scripts/run_teacher_oracle.py"), "all", "--out", str(out),
               "--stage", "fruit_dev", *(["--smoke"] if a.smoke else []), *replace]
    code = subprocess.call(command, cwd=ECM)
    result = {**verdict(out), "harness_exit": code, "smoke": a.smoke}
    atomic_json(out / "dev_gate.json", result)
    print(json.dumps({k: result[k] for k in ("passed", "complete", "engineering_errors")}
                     | {c: p["success_rate"] for c, p in result["conditions"].items()}))
    if a.smoke:
        return code
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
