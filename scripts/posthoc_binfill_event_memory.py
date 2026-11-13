#!/usr/bin/env python3
"""Post-hoc, descriptive BinFill diagnostics for the event-memory run (not pre-registered; no effect on the verdict).

1. Failure modes from the last ground-truth frame: stopped early (button pressed before enough cubes), overfilled,
   other button-stage failure (count/colour mismatch), unfinished.
2. Count evidence kept in memory, per 32-frame decision: drop moments (+-5 frames), empty-hand frames after a drop
   (+6..+20), share of history frames within 5 cm, largest gap between selected frames.
3. Redundancy: how many approach segments (contiguous frames with distance <= 5 cm) the history covers and how many
   frames it spends on each.
4. Within-condition link: post-drop coverage of older drops at decisions right after a drop, success vs stopped early.

    python3 scripts/posthoc_binfill_event_memory.py --run results/event_memory_20261025
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

CONDITIONS = ["U0", "EV", "EVU", "EVR", "EVA"]
COLORS = ("red", "blue", "green")


def jsonl(path):
    return [json.loads(l) for l in open(path)]


def failure_kind(result, last_gt):
    in_bin = sum(int(last_gt[f"{c}_cubes_in_bin"]) for c in COLORS)
    target = sum(int(last_gt[f"{c}_cubes_target_number"]) for c in COLORS)
    env_fail = bool(last_gt["info"]["fail"][0])
    phase = last_gt.get("task_phase") or last_gt.get("current_task_name") or ""
    if result["outcome"] == "success":
        kind = "success"
    elif env_fail and phase != "press the button":
        kind = "stopped_early"
    elif env_fail and in_bin > target:
        kind = "overfilled"
    elif env_fail:
        kind = "button_stage_other"
    else:
        kind = "unfinished"
    return kind, in_bin, target


def episode(d: Path):
    frames = jsonl(d / "raw_frames.jsonl")
    in_bin = np.array([sum(int(f["gt"][f"{c}_cubes_in_bin"]) for c in COLORS) for f in frames])
    dist = {}
    for s in jsonl(d / "event_signal.jsonl"):
        for k, v in enumerate(s["distance"]):
            dist[s["first_frame"] + k] = v
    n = max(dist) + 1
    near = np.array([dist.get(t, np.inf) <= 0.05 for t in range(n)])
    seg = np.full(n, -1)
    sid = -1
    for t in range(n):
        if near[t] and (t == 0 or not near[t - 1]):
            sid += 1
        if near[t]:
            seg[t] = sid
    drops = [t for t in range(1, len(in_bin)) if in_bin[t] > in_bin[t - 1]]
    return frames, drops, dist, seg, n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.run.with_name(args.run.name + "_posthoc") / "binfill.json"
    report = {"note": __doc__.splitlines()[0], "failure_modes": {}, "memory": {}, "within_condition": {}}
    link = collections.defaultdict(list)
    for cond in CONDITIONS:
        kinds = collections.Counter()
        m = collections.defaultdict(list)
        for d in sorted((args.run / "episodes" / cond / "BinFill").glob("ep*_seed*")):
            result = json.load(open(d / "result.json"))
            frames, drops, dist, seg, n = episode(d)
            kind, _, target = failure_kind(result, frames[-1]["gt"])
            kinds[kind] += 1
            after_drop = []
            for p in jsonl(d / "policy_decisions.jsonl"):
                sel, F = np.array(p["selected_frame_ids"]), p["available_max_frame_id"]
                if len(sel) < 32:
                    continue
                hist = sel[1:-4]
                hit = lambda t, lo, hi: bool(((hist >= t + lo) & (hist <= t + hi)).any())
                past = [t for t in drops if t <= F - 20]
                if past:
                    m["drop_moment_covered"].append(np.mean([hit(t, -5, 5) for t in past]))
                    m["empty_hand_after_drop_covered"].append(np.mean([hit(t, 6, 20) for t in past]))
                dd = np.array([dist.get(int(t), np.inf) for t in hist])
                m["history_share_within_5cm"].append(float((dd <= 0.05).mean()))
                m["max_gap_frames"].append(float(np.diff(sel).max()))
                ids = seg[hist[hist < n]]
                available = len(set(seg[:max(hist) + 1]) - {-1})
                if available:
                    cnt = collections.Counter(ids[ids >= 0])
                    m["approach_segments_covered"].append(len(cnt) / available)
                    m["frames_per_covered_segment"].append(float(np.mean(list(cnt.values()))) if cnt else 0.0)
                recent = [t for t in drops if F - 40 <= t <= F]
                older = [t for t in drops if t < F - 40]
                if recent and older:
                    after_drop.append(np.mean([hit(t, 6, 20) for t in older]))
            if after_drop and target >= 2 and kind in ("success", "stopped_early"):
                link[(cond, kind)].append(float(np.mean(after_drop)))
        report["failure_modes"][cond] = dict(kinds)
        report["memory"][cond] = {k: round(float(np.mean(v)), 3) for k, v in m.items()}
        report["memory"][cond]["decisions"] = len(m["max_gap_frames"])
    for (cond, kind), v in sorted(link.items()):
        report["within_condition"][f"{cond}/{kind}"] = {"episodes": len(v), "post_drop_coverage": round(float(np.mean(v)), 3)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
