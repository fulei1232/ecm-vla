"""Fixed memory budgets; baseline remains an explicit, unmodified upstream path."""
from dataclasses import dataclass
from enum import Enum


class ManagerMode(str, Enum):
    BASELINE = "baseline_framesamp"
    FIFO = "fifo"
    RANDOM = "random"


@dataclass(frozen=True)
class MemoryConfig:
    capacity: int = 32
    recent_slots: int = 4
    tokens_per_frame: int = 16
    num_views: int = 1
    image_dim: int = 2048
    position_dim: int = 768
    state_dim: int = 8
    rolling_capacity: int = 160
    encoding_batch: int = 8
    candidate_stride: int = 16
    phase_locked_replacement: bool = True
    recent_overflow: str = "nearest"

    def __post_init__(self):
        for name in ("capacity", "recent_slots", "tokens_per_frame", "num_views", "image_dim",
                     "position_dim", "state_dim", "rolling_capacity", "encoding_batch", "candidate_stride"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.capacity <= self.recent_slots + 1 or self.rolling_capacity < self.capacity:
            raise ValueError("capacity must leave history slots; rolling must hold at least capacity frames")
        if self.recent_overflow not in ("nearest", "error"):
            raise ValueError("recent_overflow must be nearest or error")

    @property
    def token_budget(self):
        return self.capacity * self.tokens_per_frame * self.num_views
