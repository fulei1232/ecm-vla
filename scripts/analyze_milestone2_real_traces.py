#!/usr/bin/env python3
"""Evaluate M2 causal peaks against annotation-only phase transitions.

The gt fields are used only to diagnose the detector offline. They are never
passed to the detector. This script reports both fixed-unit and adaptive-unit
results so a flat score distribution cannot be hidden by threshold tuning.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.memory_management.types import FrameRecord
from src.memory_management.event_detection import CausalEventDetector, EventDetectorConfig


def load_trace(path):
    path = Path(path)
    rows = [json.loads(line) for line in (path / "raw_frames.jsonl").read_text().splitlines()]
    shards = {}
    frames = []
    for row in rows:
        if row["shard"] not in shards:
            shards[row["shard"]] = np.load(path / row["shard"], allow_pickle=False)
        shard = shards[row["shard"]]
        i = row["shard_offset"]
        rgb = np.asarray(shard["front"][i], dtype=np.float32).mean(axis=-1) / 255.0
        h, w = rgb.shape
        feat = np.array([rgb[y*h//2:(y+1)*h//2, x*w//4:(x+1)*w//4].mean()
                         for y in range(2) for x in range(4)], np.float32)
        image = np.broadcast_to(feat[None, None, :], (1, 16, 8)).copy()
        pos = np.zeros((1, 16, 12), np.float32)
        frames.append((row, FrameRecord(path.name, int(row["frame_id"]), 0,
                                        int(row["frame_id"]) / 20.0, image, pos,
                                        np.asarray(shard["state"][i], np.float32),
                                        vision_global_feat=feat)))
    return frames


def truth_frames(rows):
    truth = []
    for previous, current in zip(rows, rows[1:]):
        a, b = previous["gt"]["info"], current["gt"]["info"]
        changed = (a.get("simple_subgoal_online") != b.get("simple_subgoal_online") or
                   previous.get("is_demo") != current.get("is_demo") or
                   a.get("success") != b.get("success") or a.get("fail") != b.get("fail"))
        if changed:
            truth.append(int(current["frame_id"]))
    return truth


def score(trace, config, truth, tolerance):
    detector = CausalEventDetector(config)
    observations = [detector.observe(frame) for _, frame in trace]
    peaks = [o.frame_id for o in observations if o.is_peak]
    matched = {p for p in peaks if any(abs(p - t) <= tolerance for t in truth)}
    tp, fp, fn = len(matched), len(peaks) - len(matched), len([t for t in truth if not any(abs(p - t) <= tolerance for p in peaks)])
    values = np.asarray([o.saliency for o in observations], dtype=np.float64)
    return {"peaks": peaks, "truth": truth, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "saliency_median": float(np.median(values)),
            "saliency_p90": float(np.percentile(values, 90)),
            "saliency_p99": float(np.percentile(values, 99)),
            "saliency_cv": float(np.std(values) / max(abs(np.mean(values)), 1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-root", default="results/M1_acceptance_20261023/episodes")
    ap.add_argument("--out", default="results/M2_real_trace_analysis")
    ap.add_argument("--tolerance", type=int, default=2)
    args = ap.parse_args()
    root, out = Path(args.episodes_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixed = EventDetectorConfig()
    adaptive = EventDetectorConfig(adaptive_scale=True, scale_window=32, scale_floor=1e-5,
                                   saliency_threshold=1.5)
    reports = []
    for path in sorted(root.glob("*/*/*")):
        if not (path / "raw_frames.jsonl").exists():
            continue
        trace = load_trace(path)
        rows = [row for row, _ in trace]
        truth = truth_frames(rows)
        reports.append({"episode": str(path), "frames": len(trace),
                        "fixed": score(trace, fixed, truth, args.tolerance),
                        "adaptive": score(trace, adaptive, truth, args.tolerance)})
    result = {"schema": "milestone2-real-trace-analysis-v1", "formal": False,
              "annotation_only": True, "tolerance_frames": args.tolerance,
              "episodes": reports,
              "note": "gt transitions diagnose causal peaks; they are not online detector inputs. Flat score distributions require unit/frequency calibration before threshold tuning."}
    (out / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
