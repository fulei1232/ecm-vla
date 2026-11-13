"""Causal kinematic event detection for Milestone 2.

The detector only consumes the frame passed to ``observe`` and frames already
seen by the detector.  It deliberately has no access to an episode archive or
future labels.  A peak is confirmed when the current score is at least as
large as the configured number of previous scores and is separated from the
last confirmed peak.
"""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np

from .types import FrameRecord


@dataclass(frozen=True)
class EventDetectorConfig:
    peak_window: int = 4
    min_peak_separation: int = 4
    saliency_threshold: float = 0.75
    velocity_weight: float = 0.45
    displacement_weight: float = 0.25
    gripper_weight: float = 0.20
    contact_weight: float = 0.10
    velocity_scale: float = 0.02
    displacement_scale: float = 0.02
    gripper_scale: float = 0.05
    contact_scale: float = 0.05
    default_dt_sec: float = 1.0
    adaptive_scale: bool = False
    scale_window: int = 32
    scale_floor: float = 1e-6

    def __post_init__(self):
        for name in ("peak_window", "min_peak_separation", "scale_window"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("velocity_scale", "displacement_scale", "gripper_scale", "contact_scale", "default_dt_sec", "scale_floor"):
            if getattr(self, name) <= 0 or not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.saliency_threshold)) or self.saliency_threshold < 0:
            raise ValueError("saliency_threshold must be finite and nonnegative")
        if sum((self.velocity_weight, self.displacement_weight, self.gripper_weight, self.contact_weight)) <= 0:
            raise ValueError("at least one saliency weight must be positive")


@dataclass(frozen=True)
class KinematicObservation:
    episode_id: str
    frame_id: int
    saliency: float
    velocity: float
    displacement: float
    gripper_change: float
    contact_change: float
    event_types: tuple[str, ...]
    event_ids: tuple[str, ...]
    is_peak: bool


class CausalEventDetector:
    def __init__(self, config=None):
        self.config = config or EventDetectorConfig()
        self.reset()

    def reset(self, episode_id=None):
        self.episode_id = episode_id
        self._previous = None
        self._scores = deque(maxlen=self.config.peak_window)
        self._component_history = deque(maxlen=self.config.scale_window)
        self._observations = []
        self._last_peak = -10**18

    @property
    def observations(self):
        return tuple(self._observations)

    @staticmethod
    def _norm(value):
        return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))

    def _score(self, velocity, displacement, gripper_change, contact_change):
        c = self.config
        raw = np.asarray((velocity, displacement, gripper_change, contact_change), dtype=np.float64)
        if c.adaptive_scale and len(self._component_history) >= min(c.peak_window, c.scale_window):
            history = np.asarray(self._component_history, dtype=np.float64)
            med = np.median(history, axis=0)
            mad = np.median(np.abs(history - med), axis=0)
            scales = np.maximum(med + 3.0 * mad, c.scale_floor)
            values = tuple(raw / scales)
        elif c.adaptive_scale:
            # A scale estimated from one or two startup deltas is unstable and
            # can create a spurious first peak.  Keep startup observations
            # causal and quiet until the configured peak history is available.
            return 0.0
        else:
            values = (velocity / c.velocity_scale, displacement / c.displacement_scale,
                      gripper_change / c.gripper_scale, contact_change / c.contact_scale)
        weights = (c.velocity_weight, c.displacement_weight, c.gripper_weight, c.contact_weight)
        return float(sum(w * min(v, 10.0) for w, v in zip(weights, values)) / sum(weights))

    def observe(self, frame: FrameRecord):
        if self.episode_id is None:
            self.episode_id = frame.episode_id
        if frame.episode_id != self.episode_id:
            raise ValueError("event detector received a different episode")
        if self._previous is not None and frame.frame_id != self._previous.frame_id + 1:
            raise ValueError("event detector frames must be contiguous")
        if self._previous is None:
            velocity = displacement = gripper_change = contact_change = 0.0
            score = 0.0
        else:
            prev = self._previous
            if frame.capture_time_sec is not None and prev.capture_time_sec is not None:
                dt = frame.capture_time_sec - prev.capture_time_sec
                if dt <= 0:
                    raise ValueError("capture times must increase")
            else:
                dt = self.config.default_dt_sec
            delta = np.asarray(frame.raw_robot_state, dtype=np.float64) - np.asarray(prev.raw_robot_state, dtype=np.float64)
            displacement = self._norm(delta[:7])
            velocity = displacement / dt
            gripper_change = abs(float(delta[7])) if delta.size > 7 else 0.0
            contact_change = (abs(float(frame.contact_signal) - float(prev.contact_signal))
                              if frame.contact_signal is not None and prev.contact_signal is not None else 0.0)
            score = self._score(velocity, displacement, gripper_change, contact_change)
        prior = tuple(self._scores)
        enough_history = len(prior) >= self.config.peak_window
        is_peak = bool(score >= self.config.saliency_threshold and
                       (not enough_history or score >= max(prior)) and
                       frame.frame_id - self._last_peak >= self.config.min_peak_separation)
        event_types = []
        if is_peak:
            if velocity >= displacement / self.config.default_dt_sec and velocity > 0:
                event_types.append("motion_peak")
            if gripper_change >= self.config.gripper_scale:
                event_types.append("gripper_change")
            if contact_change >= self.config.contact_scale:
                event_types.append("contact_change")
            if not event_types:
                event_types.append("kinematic_peak")
        event_ids = tuple(f"{frame.episode_id}:event:{frame.frame_id}:{kind}" for kind in event_types)
        obs = KinematicObservation(frame.episode_id, frame.frame_id, score, velocity, displacement,
                                    gripper_change, contact_change, tuple(event_types), event_ids, is_peak)
        self._observations.append(obs)
        self._scores.append(score)
        self._component_history.append((velocity, displacement, gripper_change, contact_change))
        if is_peak:
            self._last_peak = frame.frame_id
        self._previous = frame
        return obs
