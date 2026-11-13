#!/usr/bin/env python3
"""Offline causal replay of an M1 observation trace with the M2 detector."""
import argparse
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


def frame_from_npz(path, frame_id, episode, cfg):
    data = np.load(path, allow_pickle=False)
    image = np.asarray(data["observation/image"], dtype=np.float32)
    gray = image.mean(axis=-1) / 255.0
    # Eight deterministic spatial bins are enough for causal duplicate filtering
    # in this replay; the policy's 2048-D encoder is not recomputed here.
    h, w = gray.shape
    values = []
    for yi in range(2):
        for xi in range(4):
            values.append(float(gray[yi*h//2:(yi+1)*h//2, xi*w//4:(xi+1)*w//4].mean()))
    feat = np.asarray(values, np.float32)
    image_feat = np.broadcast_to(feat[None, None, :], (1, cfg.tokens_per_frame, cfg.image_dim)).copy()
    pos_feat = np.zeros((1, cfg.tokens_per_frame, cfg.position_dim), np.float32)
    state = np.asarray(data["observation/state"], dtype=np.float32)
    return FrameRecord(episode, frame_id, 0, frame_id / 16.0, image_feat, pos_feat, state,
                       vision_global_feat=feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    episode_dir, out = Path(args.episode_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = MemoryConfig(image_dim=8, position_dim=12, candidate_stride=16)
    gen_cfg = CandidateGeneratorConfig(periodic_stride=16, visual_similarity_threshold=0.995)
    evt_cfg = EventDetectorConfig()
    generator = CausalCandidateGenerator(gen_cfg, evt_cfg)
    controller = OnlineMemoryController(cfg, "fifo", candidate_generator=generator)
    files = sorted((episode_dir / "observations").glob("decision_*.npz"))
    if not files:
        raise SystemExit(f"no observation files under {episode_dir}")
    episode = episode_dir.name
    controller.reset(episode, 0)
    all_candidates = []
    for i, path in enumerate(files):
        frame = frame_from_npz(path, i, episode, cfg)
        all_candidates.extend(generator.observe(frame))
        # The controller is replayed independently below, so this pass is a
        # strict offline sequence and cannot see future observation files.
    offline_ids = [eid for obs in generator.observations for eid in obs.event_ids]
    online_generator = CausalCandidateGenerator(gen_cfg, evt_cfg)
    online = OnlineMemoryController(cfg, "fifo", candidate_generator=online_generator)
    online.reset(episode, 0)
    for i, path in enumerate(files):
        online.ingest(frame_from_npz(path, i, episode, cfg))
    online_ids = [eid for obs in online_generator.observations for eid in obs.event_ids]
    result = {
        "schema": "milestone2-real-trace-replay-v1",
        "formal": False,
        "episode_dir": str(episode_dir),
        "frames": len(files),
        "offline_event_ids": offline_ids,
        "online_event_ids": online_ids,
        "offline_online_event_ids_equal": offline_ids == online_ids,
        "peak_frames": [o.frame_id for o in generator.observations if o.is_peak],
        "candidate_count": len(all_candidates),
        "online_stats": online.stats(),
        "note": "real M1 observation trace; no human event labels, so precision/recall are not claimed",
        "passed": bool(offline_ids == online_ids and online.stats()["bank_frames"] <= cfg.capacity),
    }
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
