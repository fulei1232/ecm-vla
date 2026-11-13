#!/usr/bin/env python3
"""Picking-event detection from the tcp-to-fruit distance signal (event_keyframes.detect_events) against the true
detachment times of the fruits, and the stability of the event-keyframe memory when the signal is corrupted the
way a perception module would corrupt it.

Ground truth: a pick is the detachment of a fruit (first timestep with `attached`). A detected event
[start, release] matches a pick when the detachment lies in [start - w, release + w] (one-to-one, in time order).

Sweeps
  radius / tol    detect_events on the clean signal (train_clean chose r = 0.03 m; val reported alongside)
  corruption      val (0-99), r = 0.03: white Gaussian noise, temporally correlated noise, i.i.d. and burst
                  dropouts (NaN), constant bias; 5 corruption seeds each
  detectors       the plain detector against two deployment variants: a 5-frame causal median filter, and
                  hysteresis (enter at r, leave at 1.5 r) with a 3-frame minimum near run
  DART            event detection on the perturbed DART trajectories (clean signal of perturbed motion)
Memory stability: at every decision (every 16 frames, t >= 16) the K4/K8 memory computed from the corrupted signal
is compared with the memory from the clean signal (Jaccard of the frame sets, and recall of clean keyframes).

    robomme_benchmark/.venv/bin/python ecm_vla/paper/experiments/exp_event_detection.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import itertools

import numpy as np

from common import SKIPPED, describe, episodes, save_json
from memory_management.event_keyframes import KeyframeConfig, detect_events, keyframes, select, spread

R, TOL, W = 0.03, 0.005, 3
DECISION_EVERY = 16


def match(events, detaches, w=W):
    """(true positives, contact lead times, release errors) with one-to-one matching in time order."""
    used, leads, rel_err = set(), [], []
    for t in sorted(detaches):
        for j, e in enumerate(events):
            if j in used:
                continue
            end = e["release"] if e["release"] is not None else e["end"] + 1
            if e["start"] - w <= t <= end + w:
                used.add(j)
                leads.append(t - e["contact"])
                rel_err.append(end - t)
                break
    return len(used), leads, rel_err


def scores(results):
    tp = sum(r["tp"] for r in results)
    det = sum(r["detected"] for r in results)
    gt = sum(r["gt"] for r in results)
    p = tp / det if det else float("nan")
    rc = tp / gt if gt else float("nan")
    return {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
            "count_exact": float(np.mean([r["detected"] == r["gt"] for r in results])),
            "contact_lead_frames": describe(list(itertools.chain(*[r["leads"] for r in results]))),
            "release_error_frames": describe(list(itertools.chain(*[r["rel_err"] for r in results]))),
            "episodes": len(results)}


def median_filter(d, k=5):
    out = np.empty_like(d)
    for t in range(len(d)):
        win = d[max(0, t - k + 1):t + 1]
        win = win[np.isfinite(win)]
        out[t] = np.median(win) if win.size else np.nan
    return out


def detect_hysteresis(d, r_in, r_out, tol, min_run=3):
    """Near when d <= r_in, stays near until d > r_out; runs shorter than min_run frames are dropped."""
    state, near = False, np.zeros(len(d), bool)
    for t, x in enumerate(d):
        if np.isfinite(x):
            state = x <= r_in if not state else x <= r_out
        near[t] = state
    edges = np.diff(np.concatenate([[False], near, [False]]).astype(np.int8))
    events = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start < min_run:
            continue
        run = np.where(np.isfinite(d[start:stop]), d[start:stop], np.inf)
        contact = start + int(np.flatnonzero(run <= run.min() + tol)[0])
        events.append({"start": int(start), "end": int(stop - 1), "contact": int(contact),
                       "release": int(stop) if stop < len(d) else None})
    return events


def corrupt(d, kind, level, rng):
    d = d.copy()
    n = len(d)
    if kind == "gauss":
        d = d + rng.normal(scale=level, size=n)
    elif kind == "ou":  # correlation time 10 frames, stationary std = level
        x, tau, out = 0.0, 10.0, np.empty(n)
        for t in range(n):
            x += -x / tau + level * np.sqrt(2 / tau) * rng.standard_normal()
            out[t] = x
        d = d + out
    elif kind == "dropout":
        d[rng.random(n) < level] = np.nan
    elif kind == "burst":  # bursts with mean length 10 frames covering a `level` fraction of frames
        mean_len = 10
        p_start = level / (mean_len * (1 - level) + level) if level < 1 else 1
        t = 0
        while t < n:
            if rng.random() < p_start:
                ln = rng.geometric(1 / mean_len)
                d[t:t + ln] = np.nan
                t += ln
            else:
                t += 1
    elif kind == "bias":
        d = d + level
    d[np.isfinite(d)] = np.maximum(d[np.isfinite(d)], 0.0)
    return d


def memory_at(d, t, k, events_fn=None):
    if events_fn is None:
        return set(select(d, t, KeyframeConfig(frames=16, events=k, radius=R, tol=TOL))[0])
    events = events_fn(d[:t + 1])
    keys = keyframes(events, k)
    rest = [f for f in range(t + 1) if f not in set(keys)]
    return set(keys) | set(spread(rest, 16 - len(keys)))


DETECTORS = {
    "plain": lambda d: detect_events(d, R, TOL),
    "median5": lambda d: detect_events(median_filter(d), R, TOL),
    "hysteresis": lambda d: detect_hysteresis(d, R, 1.5 * R, TOL),
}


class Light:
    """What a worker needs of an episode."""

    def __init__(self, ep):
        self.task, self.episode = ep.task, ep.episode
        self.distance, self.detach_times = ep.distance, ep.detach_times


def run_episode(args):
    ep, kind, level, seed, detector, with_memory = args
    d0 = ep.distance
    rng = np.random.default_rng([seed, ep.episode, 0 if ep.task == "count" else 1])
    d = corrupt(d0, kind, level, rng) if kind != "clean" else d0
    fn = DETECTORS[detector]
    events = fn(d)
    det = ep.detach_times
    tp, leads, rel_err = match(events, det.values())
    out = {"tp": tp, "detected": len(events), "gt": len(det), "leads": leads, "rel_err": rel_err}
    if with_memory:
        jac, rec, rec3 = ({4: [], 8: []} for _ in range(3))
        clean_fn = DETECTORS["plain"]
        for t in range(DECISION_EVERY, len(d0), DECISION_EVERY):
            for k in (4, 8):
                ref = memory_at(d0, t, k)
                ref_keys = set(keyframes(clean_fn(d0[:t + 1]), k))
                got = memory_at(d, t, k, None if detector == "plain" else fn)
                jac[k].append(len(ref & got) / len(ref | got))
                if ref_keys:
                    rec[k].append(len(ref_keys & got) / len(ref_keys))
                    g = np.array(sorted(got))
                    rec3[k].append(float(np.mean([np.abs(g - f).min() <= W for f in ref_keys])))
        out["jaccard"] = {k: float(np.mean(v)) for k, v in jac.items()}
        out["key_recall"] = {k: float(np.mean(v)) if v else float("nan") for k, v in rec.items()}
        out["key_recall_w"] = {k: float(np.mean(v)) if v else float("nan") for k, v in rec3.items()}
    return out


def pool_map(jobs):
    with ProcessPoolExecutor(96) as ex:
        return list(ex.map(run_episode, jobs, chunksize=4))


def main():
    train = [Light(ep) for ep in episodes("train_clean")]
    val = [Light(ep) for ep in [*episodes("val_a"), *episodes("val_b")]]
    dart = [Light(ep) for ep in episodes("train_dart")]
    report = {"radius": {}, "tol": {}, "corruption": {}, "dart": {}, "config": {"r": R, "tol": TOL, "w": W}}

    for split, eps in (("train", train), ("val", val)):
        for r in (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
            res = []
            for ep in eps:
                ev = detect_events(ep.distance, r, TOL)
                tp, leads, rel = match(ev, ep.detach_times.values())
                res.append({"tp": tp, "detected": len(ev), "gt": len(ep.detach_times), "leads": leads, "rel_err": rel})
            report["radius"][f"{split}/r={r:g}"] = scores(res)
        for tol in (0.0, 0.0025, 0.005, 0.01, 0.02):
            res = []
            for ep in eps:
                ev = detect_events(ep.distance, R, tol)
                tp, leads, rel = match(ev, ep.detach_times.values())
                res.append({"tp": tp, "detected": len(ev), "gt": len(ep.detach_times), "leads": leads, "rel_err": rel})
            report["tol"][f"{split}/tol={tol:g}"] = scores(res)
    for task in ("count", "occluded_ripe"):
        res = [run_episode((ep, "clean", 0, 0, "plain", False)) for ep in dart if ep.task == task]
        report["dart"][task] = scores(res)
    report["dart"]["all"] = scores([run_episode((ep, "clean", 0, 0, "plain", False)) for ep in dart])

    grid = [("clean", [0.0])] + [
        ("gauss", [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03]),
        ("ou", [0.005, 0.01, 0.02, 0.03]),
        ("dropout", [0.1, 0.2, 0.3, 0.5]),
        ("burst", [0.1, 0.2, 0.3, 0.5]),
        ("bias", [-0.02, -0.01, 0.01, 0.02]),
    ]
    for detector in DETECTORS:
        for kind, levels in grid:
            for level in levels:
                seeds = [0] if kind in ("clean", "bias") else range(5)
                jobs = [(ep, kind, level, s, detector, True) for s in seeds for ep in val]
                res = pool_map(jobs)
                entry = scores(res)
                entry["memory_jaccard"] = {k: float(np.mean([r["jaccard"][k] for r in res])) for k in (4, 8)}
                entry["keyframe_recall"] = {k: float(np.nanmean([r["key_recall"][k] for r in res])) for k in (4, 8)}
                entry["keyframe_recall_w3"] = {k: float(np.nanmean([r["key_recall_w"][k] for r in res])) for k in (4, 8)}
                report["corruption"][f"{detector}/{kind}/{level:g}"] = entry
                print(f"{detector:10s} {kind:8s} {level:7g} P={entry['precision']:.3f} R={entry['recall']:.3f} "
                      f"F1={entry['f1']:.3f} exact={entry['count_exact']:.2f} "
                      f"J4={entry['memory_jaccard'][4]:.3f} J8={entry['memory_jaccard'][8]:.3f} "
                      f"KR4={entry['keyframe_recall'][4]:.3f} KR4w={entry['keyframe_recall_w3'][4]:.3f}", flush=True)
    report["skipped"] = SKIPPED
    print(save_json("event_detection.json", report))
    for key, e in report["radius"].items():
        print(f"{key:14s} P={e['precision']:.3f} R={e['recall']:.3f} F1={e['f1']:.3f} exact={e['count_exact']:.2f} "
              f"lead={e['contact_lead_frames'].get('median')} rel={e['release_error_frames'].get('median')}")
    for key, e in report["dart"].items():
        print("dart", key, f"P={e['precision']:.3f} R={e['recall']:.3f} exact={e['count_exact']:.2f}")


if __name__ == "__main__":
    main()
