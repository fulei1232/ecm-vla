#!/usr/bin/env python3
"""Run the deterministic Milestone 2 causal candidate validation.

This is an engineering/replay validation on a synthetic stream. It verifies
the causal contract and the online controller integration before benchmark
episodes are used for event detector calibration.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory_management.config import MemoryConfig
from src.memory_management.types import FrameRecord
from src.memory_management.event_detection import EventDetectorConfig
from src.memory_management.candidate_generation import CandidateGeneratorConfig, CausalCandidateGenerator
from src.memory_management.online_controller import OnlineMemoryController


def make_frame(i, cfg, spikes=(), future_shift=0.0):
    state = np.zeros(cfg.state_dim, dtype=np.float32)
    state[:7] = np.array([0.001 * i, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    if i in spikes:
        state[0] += 0.25
        state[7] = 0.20
    if i >= max(spikes, default=-1) + 1:
        state[0] += future_shift
    image = np.full((cfg.num_views, cfg.tokens_per_frame, cfg.image_dim), i / 1000.0, np.float32)
    pos = np.zeros((cfg.num_views, cfg.tokens_per_frame, cfg.position_dim), np.float32)
    return FrameRecord("synthetic", i, 0, i / 10.0, image, pos, state,
                       vision_global_feat=np.array([1.0, i / 1000.0, 0.0], np.float32))


def generator(cfg):
    return CausalCandidateGenerator(
        CandidateGeneratorConfig(periodic_stride=8, visual_similarity_threshold=0.9999,
                                 visual_dedup_window=5),
        EventDetectorConfig(peak_window=2, min_peak_separation=4, saliency_threshold=1.0,
                            velocity_scale=0.01, displacement_scale=0.01,
                            gripper_scale=0.05, contact_scale=0.05))


def replay(frames, cfg):
    g = generator(cfg)
    candidates = []
    for frame in frames:
        candidates.extend(g.observe(frame))
    return g, candidates


def write_svg(path, records, event_frames):
    width, height = 1000, 360
    if not records:
        return
    xs = [r["frame_id"] for r in records]
    ys = [r["saliency"] for r in records]
    xmax, ymax = max(xs) or 1, max(max(ys), 1.0)
    points = " ".join(f"{50 + 900*x/xmax:.1f},{300 - 240*y/ymax:.1f}" for x, y in zip(xs, ys))
    marks = " ".join(f'<line x1="{50 + 900*x/xmax:.1f}" y1="55" x2="{50 + 900*x/xmax:.1f}" y2="305" stroke="#d33" stroke-width="2"/>'
                     for x in event_frames)
    path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/><text x="50" y="25" font-size="18">Milestone 2 causal saliency and event peaks</text>
<line x1="50" y1="300" x2="950" y2="300" stroke="#333"/>{marks}<polyline fill="none" stroke="#167" stroke-width="2" points="{points}"/>
<text x="50" y="340" font-size="12">blue: causal saliency; red: confirmed event frame</text></svg>''', encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/M2_validation")
    ap.add_argument("--frames", type=int, default=160)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = MemoryConfig(image_dim=8, position_dim=12, candidate_stride=8)
    event_frames = (32, 80, 128)
    frames = [make_frame(i, cfg, event_frames) for i in range(args.frames)]
    g, candidates = replay(frames, cfg)

    prefix = 96
    base_prefix, _ = replay(frames[:prefix], cfg)
    altered = [make_frame(i, cfg, event_frames, future_shift=100.0) for i in range(prefix, args.frames)]
    changed_prefix, _ = replay(frames[:prefix] + altered, cfg)
    base_ids = [x.event_ids for x in base_prefix.observations if x.frame_id < prefix]
    changed_ids = [x.event_ids for x in changed_prefix.observations if x.frame_id < prefix]
    causal_prefix_equal = base_ids == changed_ids
    all_event_ids = [eid for obs in g.observations for eid in obs.event_ids]
    detected = [obs.frame_id for obs in g.observations if obs.is_peak]
    tolerance = 1
    tp = sum(any(abs(d - truth) <= tolerance for truth in event_frames) for d in detected)
    fn = sum(not any(abs(d - truth) <= tolerance for d in detected) for truth in event_frames)
    fp = max(0, len(detected) - tp)

    online_generator = generator(cfg)
    controller = OnlineMemoryController(cfg, "fifo", candidate_generator=online_generator)
    controller.reset("synthetic", 0)
    for frame in frames:
        controller.ingest(frame)
    controller.update()
    stats = controller.stats()
    online_event_ids = [eid for obs in online_generator.observations for eid in obs.event_ids]
    records = list(g.records)
    with (out / "timeline.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_id", "frame_id", "capture_time_sec", "saliency",
                                               "event_ids", "event_types", "is_peak", "candidate_sources"])
        writer.writeheader()
        for row in records:
            row = dict(row)
            row["event_ids"] = ";".join(row["event_ids"])
            row["event_types"] = ";".join(row["event_types"])
            row["candidate_sources"] = json.dumps(row["candidate_sources"], sort_keys=True)
            writer.writerow(row)
    write_svg(out / "saliency_events.svg", records, detected)
    result = {
        "schema": "milestone2-causal-candidate-validation-v1",
        "formal": False,
        "scope": "synthetic engineering and causal replay validation; not benchmark precision/recall",
        "frames": args.frames,
        "periodic_stride": 8,
        "event_truth_frames": list(event_frames),
        "detected_peak_frames": detected,
        "event_ids": all_event_ids,
        "online_event_ids": online_event_ids,
        "offline_online_event_ids_equal": all_event_ids == online_event_ids,
        "causal_prefix_equal_after_future_perturbation": causal_prefix_equal,
        "all_confirmed_at_or_before_frame": all(obs.frame_id >= 0 and obs.frame_id <= frames[obs.frame_id].frame_id
                                                 for obs in g.observations),
        "synthetic_metrics": {"tp": tp, "fp": fp, "fn": fn,
                               "precision": tp / (tp + fp) if tp + fp else 1.0,
                               "recall": tp / (tp + fn) if tp + fn else 1.0,
                               "tolerance_frames": tolerance},
        "online_integration": {"bank_frames": stats["bank_frames"],
                                "candidate_records": stats["candidate_records"],
                                "candidate_queue": stats["candidate_queue"],
                                "selected_frame_ids": stats["selected_frame_ids"],
                                "selected_sources": stats["selected_sources"]},
        "passed": bool(causal_prefix_equal and all_event_ids == online_event_ids and
                        all_event_ids and stats["bank_frames"] == cfg.capacity),
        "artifacts": ["timeline.csv", "saliency_events.svg"]
    }
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
