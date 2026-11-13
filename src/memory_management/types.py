"""Immutable frame ownership prevents views from retaining entire encoding batches."""
from dataclasses import dataclass, field, replace
import math
import numpy as np


def owned_array(value):
    out = np.array(value, copy=True, order="C")
    if not np.isfinite(out).all():
        raise ValueError("frame features must be finite")
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False)
class FrameRecord:
    episode_id: str
    frame_id: int
    decision_id_available: int
    capture_time_sec: float | None
    image_feat: np.ndarray
    pos_feat: np.ndarray
    raw_robot_state: np.ndarray
    image_for_dedup: np.ndarray | None = None
    vision_global_feat: np.ndarray | None = None
    contact_signal: float | None = None

    def __post_init__(self):
        if not self.episode_id or self.frame_id < 0 or self.decision_id_available < 0:
            raise ValueError("invalid frame identity")
        if self.capture_time_sec is not None and (not math.isfinite(self.capture_time_sec) or self.capture_time_sec < 0):
            raise ValueError("invalid capture time")
        if self.contact_signal is not None and not math.isfinite(float(self.contact_signal)):
            raise ValueError("contact signal must be finite")
        for name in ("image_feat", "pos_feat", "raw_robot_state", "image_for_dedup", "vision_global_feat"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, owned_array(value))
        if self.image_feat.ndim != 3 or self.pos_feat.ndim != 3 or self.raw_robot_state.ndim != 1:
            raise ValueError("expected [views,tokens,dim] image/position and [state_dim] raw state")
        if self.image_feat.shape[:2] != self.pos_feat.shape[:2]:
            raise ValueError("image and position token layouts differ")

    @property
    def nbytes(self):
        return sum(x.nbytes for x in (self.image_feat, self.pos_feat, self.raw_robot_state,
                   self.image_for_dedup, self.vision_global_feat) if x is not None)


@dataclass(frozen=True)
class Candidate:
    frame: FrameRecord
    sources: frozenset[str] = field(default_factory=lambda: frozenset({"periodic"}))
    event_types: tuple[str, ...] = ()
    event_saliency: float | None = None
    confirmed_at_frame: int = 0
    event_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.confirmed_at_frame < self.frame.frame_id:
            raise ValueError("candidate cannot be confirmed before its frame")
        object.__setattr__(self, "sources", frozenset(self.sources))
        object.__setattr__(self, "event_types", tuple(sorted(set(self.event_types))))
        object.__setattr__(self, "event_ids", tuple(sorted(set(self.event_ids))))

    @property
    def priority(self):
        """Stable causal ranking: event saliency, event source, then frame time."""
        return (float(self.event_saliency) if self.event_saliency is not None else 0.0,
                1 if "event" in self.sources else 0,
                -self.frame.frame_id)


@dataclass(frozen=True)
class MemorySlot:
    frame: FrameRecord
    role: str
    phase: str
    sources: frozenset[str] = frozenset()
    event_types: tuple[str, ...] = ()
    event_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.role not in ("anchor", "recent", "history") or self.phase not in ("demo", "exec"):
            raise ValueError("invalid slot role or phase")

    @property
    def frame_id(self):
        return self.frame.frame_id

    @property
    def is_protected(self):
        return self.role != "history"

    @property
    def is_history(self):
        return self.role == "history"

    def merged(self, candidate):
        return replace(self, sources=self.sources | candidate.sources,
                       event_types=tuple(sorted(set(self.event_types) | set(candidate.event_types))),
                       event_ids=tuple(sorted(set(self.event_ids) | set(candidate.event_ids))))
