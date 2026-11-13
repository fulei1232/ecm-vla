"""Memory partition shared by V-A, V-B and the later bounded bank (plan §0A.2, §6.1).

For a decision at frame F the official FrameSamp indices U = even_sampling_indices(F, K) split into
anchor A = U[:1], recent window R = U[-r:], history H = U[1:-r]. R is the last r FrameSamp slots,
not the last r raw frames.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

DEMO, EXEC = "demo", "exec"


def even_sampling_indices(step_idx: int, max_size: int = 32) -> list[int]:
    """Identical to mme_vla_suite.shared.data_utils.even_sampling_indices."""
    if step_idx < max_size:
        return list(range(step_idx + 1))
    return np.linspace(0, step_idx, max_size, dtype=np.int32).tolist()


def phase_of(frame: int, exec_start_idx: int) -> str:
    return DEMO if frame < exec_start_idx else EXEC


def phase_counts(frames, exec_start_idx: int) -> dict[str, int]:
    counts = Counter(phase_of(f, exec_start_idx) for f in frames)
    return {DEMO: counts.get(DEMO, 0), EXEC: counts.get(EXEC, 0)}


@dataclass(frozen=True)
class Partition:
    max_id: int
    uniform: tuple[int, ...]
    anchor: tuple[int, ...]
    history: tuple[int, ...]
    recent: tuple[int, ...]

    @property
    def protected(self) -> frozenset[int]:
        return frozenset(self.anchor) | frozenset(self.recent)

    @property
    def full(self) -> bool:
        """True when FrameSamp already drops frames, i.e. a replacement decision exists."""
        return len(self.uniform) < self.max_id + 1


def partition(max_id: int, capacity: int = 32, recent_slots: int = 4) -> Partition:
    uniform = even_sampling_indices(max_id, capacity)
    if len(uniform) < recent_slots + 1:
        return Partition(max_id, tuple(uniform), tuple(uniform[:1]), (), tuple(uniform[1:]))
    return Partition(max_id, tuple(uniform), (uniform[0],), tuple(uniform[1:-recent_slots]),
                     tuple(uniform[-recent_slots:]))


def recent_window(max_id: int, capacity: int = 32, recent_slots: int = 4) -> list[int]:
    return list(partition(max_id, capacity, recent_slots).recent)
