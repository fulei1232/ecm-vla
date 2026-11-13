"""Bounded causal feature stream shared by replay and the live policy adapter."""
from .config import MemoryConfig, ManagerMode
from .types import Candidate
from .rolling_buffer import RollingObservationBuffer
from .bounded_bank import BoundedMemoryBank
from .packer import ManagedMemoryPacker


class OnlineMemoryController:
    def __init__(self, config=None, mode=ManagerMode.FIFO, seed=7, normalize_state=None,
                 candidate_generator=None):
        self.config = config or MemoryConfig()
        self.bank = BoundedMemoryBank(self.config, mode, seed)
        self.rolling = RollingObservationBuffer(self.config.rolling_capacity)
        self.packer = ManagedMemoryPacker(self.config, normalize_state)
        self.candidate_generator = candidate_generator
        self.reset()

    def reset(self, episode_id=None, exec_start_idx=0):
        self.rolling.reset(episode_id)
        self.bank.reset(episode_id, exec_start_idx)
        self.decision_id = 0
        self.updates = 0
        self._last_update_frame = -1
        self._candidate_queue = []
        if self.candidate_generator is not None:
            self.candidate_generator.reset(episode_id)

    def attach_candidate_generator(self, generator):
        self.candidate_generator = generator
        self.candidate_generator.reset(self.rolling.episode_id)
        self._candidate_queue = []

    def ingest(self, frame):
        cfg = self.config
        if (frame.image_feat.shape != (cfg.num_views, cfg.tokens_per_frame, cfg.image_dim)
                or frame.pos_feat.shape != (cfg.num_views, cfg.tokens_per_frame, cfg.position_dim)
                or frame.raw_robot_state.shape != (cfg.state_dim,)):
            raise ValueError("frame does not match the configured tensor contract")
        if frame.decision_id_available > self.decision_id:
            raise ValueError("frame not available at the current decision boundary")
        self.rolling.append(frame)
        if self.candidate_generator is not None:
            self._candidate_queue.extend(self.candidate_generator.observe(frame))
        F = frame.frame_id
        # Bound even a long demonstration burst; warmup retains every frame.
        if F < self.config.capacity or F % self.config.candidate_stride == 0:
            self.update()

    def update(self):
        F = self.rolling.latest_frame_id
        if F == self._last_update_frame:
            return
        self.bank.refresh_recent(F, self.rolling, self.decision_id)
        if F >= self.config.capacity:
            cutoff = min(self.bank.recent_frame_ids) - 1
            if self.candidate_generator is None:
                candidates = [Candidate(f, confirmed_at_frame=F) for f in self.rolling.snapshot()
                              if self.bank.last_candidate_cutoff < f.frame_id <= cutoff
                              and f.frame_id % self.config.candidate_stride == 0 and f.frame_id != 0]
            else:
                candidates = [c for c in self._candidate_queue
                              if self.bank.last_candidate_cutoff < c.frame.frame_id <= cutoff
                              and c.confirmed_at_frame <= F]
                candidates.sort(key=lambda c: c.priority, reverse=True)
                self._candidate_queue = [c for c in self._candidate_queue
                                         if c.frame.frame_id > cutoff or c.confirmed_at_frame > F]
            self.updates += sum(r["accepted"] for r in self.bank.apply_candidates(candidates))
            self.bank.last_candidate_cutoff = max(self.bank.last_candidate_cutoff, cutoff)
        self._last_update_frame = F

    def prepare(self):
        if self.rolling.latest_frame_id < 0:
            raise ValueError("add observations before inference")
        self.update()
        packed = self.packer.pack(self.bank.snapshot(), max_frame_id=self.rolling.latest_frame_id,
                                  decision_id=self.decision_id)
        self.decision_id += 1
        return packed

    def stats(self):
        bank = self.bank
        unique = {id(f): f for f in self.rolling.snapshot()}
        unique.update({id(s.frame): s.frame for s in bank.snapshot()})
        return {"bank_frames": len(bank.snapshot()), "rolling_frames": len(self.rolling),
                "retained_unique_frames": len(unique), "bank_bytes": bank.nbytes, "rolling_bytes": self.rolling.nbytes,
                "retained_unique_bytes": sum(f.nbytes for f in unique.values()), "accepted_updates": self.updates,
                "max_frame_bytes": max((f.nbytes for f in unique.values()), default=0),
                "selected_frame_ids": [s.frame_id for s in bank.snapshot()],
                "selected_sources": {str(s.frame_id): sorted(s.sources) for s in bank.snapshot()},
                "selected_event_ids": {str(s.frame_id): list(s.event_ids) for s in bank.snapshot() if s.event_ids},
                "recent_frame_ids": list(bank.recent_frame_ids), "recent_window_approx": bank.recent_window_approx,
                "phase_slot_counts": bank.phase_slot_counts(),
                "candidate_queue": len(self._candidate_queue),
                "candidate_records": len(self.candidate_generator.records) if self.candidate_generator is not None else 0}
