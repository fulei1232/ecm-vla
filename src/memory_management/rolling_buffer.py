"""Contiguous streaming ring: no disk loader, backing history list, or unbounded pending queue."""
from collections import OrderedDict


class RollingObservationBuffer:
    def __init__(self, capacity=160):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.reset()

    def reset(self, episode_id=None):
        self.episode_id = episode_id
        self.latest_frame_id = -1
        self._frames = OrderedDict()

    def append(self, frame):
        if self.episode_id is None:
            self.episode_id = frame.episode_id
        if frame.episode_id != self.episode_id or frame.frame_id != self.latest_frame_id + 1:
            raise ValueError("frames must be contiguous, chronological and from the same episode")
        self._frames[frame.frame_id] = frame
        self.latest_frame_id = frame.frame_id
        if len(self._frames) > self.capacity:
            self._frames.popitem(last=False)

    def __len__(self):
        return len(self._frames)

    def get(self, frame_id):
        return self._frames[frame_id]

    def snapshot(self):
        return tuple(self._frames.values())

    @property
    def nbytes(self):
        return sum(f.nbytes for f in self._frames.values())
