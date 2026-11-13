"""Periodic/event candidate generation, causal visual filtering and merging."""
from dataclasses import dataclass
import math
import numpy as np

from .event_detection import CausalEventDetector, EventDetectorConfig
from .types import Candidate, FrameRecord


@dataclass(frozen=True)
class CandidateGeneratorConfig:
    periodic_stride: int = 16
    visual_similarity_threshold: float = 0.995
    visual_dedup_window: int = 32
    max_recent_candidates: int = 256

    def __post_init__(self):
        if self.periodic_stride <= 0 or self.visual_dedup_window <= 0 or self.max_recent_candidates <= 0:
            raise ValueError("candidate generator sizes must be positive")
        if not 0 <= self.visual_similarity_threshold <= 1:
            raise ValueError("visual_similarity_threshold must lie in [0, 1]")


def _cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / (na * nb))


class CausalCandidateGenerator:
    def __init__(self, config=None, event_config=None, visual_encoder=None):
        self.config = config or CandidateGeneratorConfig()
        self.event_detector = CausalEventDetector(event_config or EventDetectorConfig())
        self.visual_encoder = visual_encoder
        self.reset()

    def reset(self, episode_id=None):
        self.episode_id = episode_id
        self.event_detector.reset(episode_id)
        self._recent = []
        self._records = []

    @property
    def observations(self):
        return self.event_detector.observations

    @property
    def records(self):
        return tuple(self._records)

    def _visual(self, frame):
        if self.visual_encoder is not None:
            value = self.visual_encoder(frame)
        elif frame.vision_global_feat is not None:
            value = frame.vision_global_feat
        elif frame.image_for_dedup is not None:
            value = frame.image_for_dedup
        else:
            value = np.asarray(frame.image_feat, dtype=np.float32).mean(axis=(0, 1))
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError("visual dedup feature must be finite and nonempty")
        return value

    def _duplicate(self, frame, current_source):
        feature = self._visual(frame)
        for old_frame, old_sources, old_feature in reversed(self._recent[-self.config.visual_dedup_window:]):
            similarity = _cosine(feature, old_feature)
            if similarity is None or similarity < self.config.visual_similarity_threshold:
                continue
            # Event candidates are retained even when a periodic frame is visually
            # redundant; periodic candidates are filtered against any prior frame.
            if current_source == "event" and "event" not in old_sources:
                continue
            return True
        return False

    def observe(self, frame: FrameRecord):
        if self.episode_id is None:
            self.episode_id = frame.episode_id
            self.event_detector.reset(frame.episode_id)
        if frame.episode_id != self.episode_id:
            raise ValueError("candidate generator received a different episode")
        event = self.event_detector.observe(frame)
        raw = []
        if frame.frame_id > 0 and frame.frame_id % self.config.periodic_stride == 0:
            raw.append(Candidate(frame, frozenset({"periodic"}), confirmed_at_frame=frame.frame_id))
        if event.is_peak:
            raw.append(Candidate(frame, frozenset({"event"}), event.event_types, event.saliency,
                                 confirmed_at_frame=frame.frame_id, event_ids=event.event_ids))
        emitted = []
        for candidate in raw:
            source = "event" if "event" in candidate.sources else "periodic"
            if self._duplicate(frame, source):
                continue
            duplicate = next((c for c in emitted if c.frame.frame_id == candidate.frame.frame_id), None)
            if duplicate is not None:
                emitted.remove(duplicate)
                candidate = Candidate(frame, duplicate.sources | candidate.sources,
                                      duplicate.event_types + candidate.event_types,
                                      max(x for x in (duplicate.event_saliency, candidate.event_saliency) if x is not None)
                                      if duplicate.event_saliency is not None or candidate.event_saliency is not None else None,
                                      frame.frame_id, duplicate.event_ids + candidate.event_ids)
            emitted.append(candidate)
        emitted.sort(key=lambda c: c.priority, reverse=True)
        feature = self._visual(frame)
        if emitted:
            self._recent.append((frame, frozenset().union(*(c.sources for c in emitted)), feature))
            self._recent = self._recent[-self.config.visual_dedup_window:]
        self._records.append({"episode_id": frame.episode_id, "frame_id": frame.frame_id,
                              "capture_time_sec": frame.capture_time_sec, "saliency": event.saliency,
                              "event_ids": list(event.event_ids), "event_types": list(event.event_types),
                              "is_peak": event.is_peak,
                              "candidate_sources": [sorted(c.sources) for c in emitted]})
        self._records = self._records[-self.config.max_recent_candidates:]
        return tuple(emitted)
