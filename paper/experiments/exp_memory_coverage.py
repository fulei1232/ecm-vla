#!/usr/bin/env python3
"""What a fixed-budget memory retains of the picks already made (offline, ground-truth trajectories).

At every decision (every 16 frames, t >= 16) of the val demonstrations, each pick completed before t (its event,
from the clean distance signal, has released by t) is
  covered (key)   when the memory holds a frame within +-3 frames of its contact or release keyframe,
  covered (run)   when the memory holds any frame of its near run [start, release].
Per decision: count recoverable = every completed pick covered (run); near fraction = memory frames with d <= r;
largest gap = the longest stretch of frames 0..t without a memory frame (temporal coverage); turnover = fraction of
the memory replaced since the previous decision.

Memories (all exactly B frames once t + 1 > B):
  U        uniform (official even sampling; = PEKM k = 0)
  RECENT   the last B frames (sliding window)
  RANDOM   frame 0, frame t and B - 2 random frames (5 seeds)
  PEKM-k   event keyframes of the most recent events (at most k) + uniform rest
  PEKM-S-a soft proximity-weighted density (tau = 0.05 m), anchor + last 4 uniform slots kept, alpha = a

    robomme_benchmark/.venv/bin/python ecm_vla/paper/experiments/exp_memory_coverage.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import time

import numpy as np

from common import SKIPPED, describe, episodes, save_json
from memory_management.event_keyframes import KeyframeConfig, detect_events, select
from memory_management.validation.event_memory import event_selection, proximity
from memory_management.validation.partition import partition

R, TOL, W, EVERY = 0.03, 0.005, 3, 16


def memories(d, t, B):
    out = {"U": select(d, t, KeyframeConfig(frames=B, events=0))[0]}
    if t + 1 <= B:
        frames = list(range(t + 1))
        return {name: frames for name in names(B)}
    out["RECENT"] = list(range(t - B + 1, t + 1))
    for s in range(5):
        rng = np.random.default_rng([s, t, B])
        out[f"RANDOM#{s}"] = sorted({0, t, *rng.choice(np.arange(1, t), size=B - 2, replace=False).tolist()})
    for k in ks(B):
        out[f"PEKM-{k}"] = select(d, t, KeyframeConfig(frames=B, events=k, radius=R, tol=TOL))[0]
    part = partition(t, capacity=B, recent_slots=4)
    p = proximity(d[:t + 1], 0.05)
    for a in (0.25, 0.5, 1.0):
        out[f"PEKM-S-{a:g}"] = event_selection(part, p, a)
    return out


def ks(B):
    return sorted({2, 4, 8, min(12, B - 4), B // 2} & set(range(1, B)))


def names(B):
    return ["U", "RECENT", *[f"RANDOM#{s}" for s in range(5)], *[f"PEKM-{k}" for k in ks(B)],
            *[f"PEKM-S-{a:g}" for a in (0.25, 0.5, 1.0)]]


def run(args):
    task, d, B = args
    events = detect_events(d, R, TOL)
    acc = defaultdict(lambda: defaultdict(list))
    prev = {}
    for t in range(EVERY, len(d), EVERY):
        done = [e for e in events if e["release"] is not None and e["release"] <= t]
        mem = memories(d, t, B)
        for name, frames in mem.items():
            base = name.split("#")[0]
            m = np.asarray(sorted(frames))
            a = acc[base]
            if done:
                key = [min(np.abs(m - e["contact"]).min(), np.abs(m - e["release"]).min()) <= W for e in done]
                run_ = [bool(((m >= e["start"]) & (m <= e["release"])).any()) for e in done]
                a["key"] += key
                a["run"] += run_
                a["count_recoverable"].append(all(run_))
                for back, (kk, rr) in enumerate(zip(reversed(key), reversed(run_))):
                    a[f"key_back{min(back, 3)}"].append(kk)
            a["near"].append(float(np.mean(d[m] <= R)))
            a["gap"].append(int(np.diff(np.concatenate([[-1], m, [t + 1]])).max() - 1))
            if name in prev:
                a["turnover"].append(1 - len(set(prev[name]) & set(frames)) / len(frames))
            prev[name] = frames
    return task, B, {k: dict(v) for k, v in acc.items()}


def main():
    eps = [*episodes("val_a"), *episodes("val_b")]
    jobs = [(ep.task, ep.distance, B) for B in (8, 16, 32) for ep in eps]
    pooled = defaultdict(lambda: defaultdict(list))
    with ProcessPoolExecutor(96) as ex:
        for task, B, acc in ex.map(run, jobs, chunksize=2):
            for name, metrics in acc.items():
                for m, v in metrics.items():
                    pooled[(task, B, name)][m] += v
                    pooled[("all", B, name)][m] += v
    report = {}
    for (task, B, name), metrics in sorted(pooled.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        entry = {m: (float(np.mean(v)) if m not in ("gap",) else describe(v)) for m, v in metrics.items()}
        entry["n_pick_decisions"] = len(metrics.get("key", []))
        entry["n_decisions"] = len(metrics["near"])
        report[f"{task}/B={B}/{name}"] = entry

    # latency of one PEKM selection at a long episode (t = 1000 frames, 5 events)
    rng = np.random.default_rng(0)
    d = np.abs(np.sin(np.linspace(0, 15, 1001))) * 0.4 + rng.normal(scale=0.002, size=1001)
    lat = {}
    for k in (0, 4, 8):
        cfg = KeyframeConfig(frames=16, events=k)
        times = []
        for _ in range(2000):
            t0 = time.perf_counter()
            select(d, 1000, cfg)
            times.append(time.perf_counter() - t0)
        lat[f"PEKM-{k}"] = {"median_us": float(np.median(times) * 1e6), "p99_us": float(np.percentile(times, 99) * 1e6)}
    report["latency_t1000"] = lat
    report["skipped"] = SKIPPED
    print(save_json("memory_coverage.json", report))
    for key, e in report.items():
        if key in ("latency_t1000", "skipped"):
            print(key, e)
            continue
        if key.split("/")[0] != "all":
            continue
        print(f"{key:24s} key={e.get('key', float('nan')):.3f} run={e.get('run', float('nan')):.3f} "
              f"count={e.get('count_recoverable', float('nan')):.3f} last={e.get('key_back0', float('nan')):.3f} "
              f"near={e['near']:.3f} gap_med={e['gap']['median']:.0f} turnover={e.get('turnover', float('nan')):.3f}")


if __name__ == "__main__":
    main()
