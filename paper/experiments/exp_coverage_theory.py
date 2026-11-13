#!/usr/bin/env python3
"""Why uniform memory loses events: the coverage predicted by sampling geometry vs the measured coverage.

Uniform memory keeps B frames evenly spaced over 0..t (spacing t / (B - 1)). A completed pick is covered when a
memory frame lies in its key set S (+-W frames around the contact and the release keyframe, |S| <= 2 (2W + 1)).
With the grid phase independent of the event, the expected number of memory frames in S is (B - 1) |S| / t, so

    c_U(t) ~= min(1, (B - 1) |S| / t)        (coverage decays as 1 / t)

whereas PEKM with cap k covers the floor(k / 2) most recent picks at every t by construction. Measured on the val
demonstrations (same decisions and coverage rule as exp_memory_coverage.py), per budget and per decision-time bin.

    robomme_benchmark/.venv/bin/python ecm_vla/paper/experiments/exp_coverage_theory.py
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import SKIPPED, episodes, save_json
from memory_management.event_keyframes import KeyframeConfig, detect_events, select

R, TOL, W, EVERY = 0.03, 0.005, 3, 16
BINS = ((16, 200), (200, 400), (400, 600), (600, 2000))


def key_set(e, t):
    return {f for c in (e["contact"], e["release"]) for f in range(max(c - W, 0), min(c + W, t) + 1)}


def run(args):
    d, B = args
    events = detect_events(d, R, TOL)
    rows = []
    for t in range(EVERY, len(d), EVERY):
        done = [e for e in events if e["release"] is not None and e["release"] <= t]
        if not done:
            continue
        mems = {name: set(select(d, t, KeyframeConfig(frames=B, events=k, radius=R, tol=TOL))[0])
                for name, k in (("U", 0), ("PEKM-4", 4), ("PEKM-8", 8)) if k < B}
        for back, e in enumerate(reversed(done)):
            s = key_set(e, t)
            rows.append({"t": t, "back": back, "pred": min(1.0, (B - 1) * len(s) / t),
                         **{name: bool(s & m) for name, m in mems.items()}})
    return B, rows


def main():
    eps = [*episodes("val_a"), *episodes("val_b")]
    jobs = [(ep.distance, B) for B in (8, 16, 32) for ep in eps]
    pooled = defaultdict(list)
    with ProcessPoolExecutor(96) as ex:
        for B, rows in ex.map(run, jobs, chunksize=2):
            pooled[B] += rows
    report = {}
    for B, rows in sorted(pooled.items()):
        entry = {"n_pairs": len(rows), "U_measured": float(np.mean([r["U"] for r in rows])),
                 "U_predicted": float(np.mean([r["pred"] for r in rows]))}
        for name in ("PEKM-4", "PEKM-8"):
            if name in rows[0]:
                entry[name] = float(np.mean([r[name] for r in rows]))
        by_t = {}
        for lo, hi in BINS:
            sel = [r for r in rows if lo <= r["t"] < hi]
            if not sel:
                continue
            recent = [r for r in sel if r["back"] == 0]
            by_t[f"{lo}-{hi}"] = {
                "n": len(sel), "U_measured": float(np.mean([r["U"] for r in sel])),
                "U_predicted": float(np.mean([r["pred"] for r in sel])),
                **{name: float(np.mean([r[name] for r in sel])) for name in ("PEKM-4", "PEKM-8") if name in sel[0]},
                "U_last_pick": float(np.mean([r["U"] for r in recent])),
                **{f"{name}_last_pick": float(np.mean([r[name] for r in recent]))
                   for name in ("PEKM-4", "PEKM-8") if name in sel[0]},
            }
        entry["by_decision_time"] = by_t
        report[f"B={B}"] = entry
    report["skipped"] = SKIPPED
    print(save_json("coverage_theory.json", report))
    for key, e in report.items():
        if key == "skipped":
            continue
        print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k != "by_decision_time"})
        for b, v in e["by_decision_time"].items():
            print("   ", b, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})


if __name__ == "__main__":
    main()
