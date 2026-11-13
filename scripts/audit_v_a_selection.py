#!/usr/bin/env python3
"""Replay all V-A conditions on recorded official D0 rollouts and summarize the selections.

The rollouts are ACM M3 (test split); this only characterizes the interventions before the
closed loop, it produces no success-rate evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ECM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ECM))
from src.memory_management.validation.online_variants import CONDITIONS, EpisodeState, choose  # noqa: E402
from src.memory_management.validation.partition import partition  # noqa: E402

M3 = Path("/media/fulei/fuleivla/acm_vla/results/20261012T122037_m3_diagnostic80")


def read_lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=M3)
    p.add_argument("--out", type=Path, default=ECM / "reports/v_a_selection_audit.json")
    opts = p.parse_args()
    acc = defaultdict(lambda: defaultdict(list))
    for d in sorted((opts.run / "episodes").glob("*/ep*")):
        task = d.parent.name
        rows, decisions = read_lines(d / "raw_frames.jsonl"), read_lines(d / "policy_decisions.jsonl")
        exec_start = decisions[0]["exec_start_idx"]
        states = {c: EpisodeState(task, int(d.name[2:]), 7, exec_start) for c in CONDITIONS}
        for dec in decisions:
            F = dec["available_max_frame_id"]
            if not partition(F).full:
                continue
            now = [r for r in rows if r["frame_id"] <= F]
            for c in CONDITIONS:
                _, log = choose(c, dec["selected_frame_ids"], F, now, states[c])
                a = acc[(task, c)]
                a["replaced"].append(len(log["deleted_d0_frames"]))
                a["max_gap"].append(log["max_gap"])
                a["cur_prev_gap"].append(log["current_to_prev_gap"])
                a["exec_slots"].append(log["n_exec_slots"])
                a["demo_slots"].append(log["n_demo_slots"])
                a["overflow"].append(log["pin_overflow"])
                a["borrow"].append(log["phase_borrow"] > 0)
                a["protected"].append(log["protected_present"])
                if "n_pin_frames" in log:
                    a["pins"].append(log["n_pin_frames"])
                if c == "RNDC" and "deleted_jaccard_vs_d4b" in log and log["deleted_d0_frames"]:
                    a["jaccard_vs_d4b"].append(log["deleted_jaccard_vs_d4b"])
                if c in ("RNDB", "RNDC"):
                    reps = log.get("replacements", [])
                    a["evidence_overlap"].extend(r["evidence_overlap"] for r in reps)
                    a["between_neighbours"].extend(r["between_d4b_neighbours"] for r in reps)
                    a["same_time_bin"].extend(r["same_time_bin_now"] for r in reps)
                    a["distance_to_pin"].extend(r["distance_to_pin"] for r in reps)
                    a["redraw"].extend(r["rnd_collision_redraw"] for r in reps)
                    a["rndb_infeasible"].append(bool(log.get("rndb_infeasible")))
    summary = []
    for (task, c), a in sorted(acc.items()):
        row = {"task": task, "condition": c, "decisions": len(a["replaced"])}
        for k in ("replaced", "max_gap", "cur_prev_gap", "exec_slots", "demo_slots", "pins", "distance_to_pin",
                  "jaccard_vs_d4b"):
            if a[k]:
                row[f"{k}_median"] = float(np.median(a[k]))
                row[f"{k}_max"] = int(np.max(a[k]))
        for k in ("overflow", "borrow", "protected", "evidence_overlap", "between_neighbours", "same_time_bin",
                  "redraw", "rndb_infeasible"):
            if a[k]:
                row[f"{k}_rate"] = round(float(np.mean(a[k])), 3)
        summary.append(row)
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    opts.out.write_text(json.dumps({"source_run": str(opts.run), "split": "test (code validation only)",
                                    "summary": summary}, indent=2) + "\n")
    cols = ["task", "condition", "decisions", "replaced_median", "replaced_max", "max_gap_median",
            "cur_prev_gap_median", "exec_slots_median", "demo_slots_median", "pins_median",
            "overflow_rate", "borrow_rate", "protected_rate", "distance_to_pin_median", "between_neighbours_rate",
            "same_time_bin_rate", "evidence_overlap_rate", "redraw_rate", "rndb_infeasible_rate", "jaccard_vs_d4b_median"]
    print("\t".join(cols))
    for row in summary:
        print("\t".join(str(row.get(k, "")) for k in cols))


if __name__ == "__main__":
    main()
