"""Atomic fixed-capacity bank with source-time recent refresh and phase accounting."""
from copy import deepcopy
import numpy as np

from .config import MemoryConfig, ManagerMode
from .types import MemorySlot
from .validation.partition import partition, phase_of, phase_counts
from .validation.phase_balanced_fill import balanced_subset


class BoundedMemoryBank:
    def __init__(self, config=None, mode=ManagerMode.FIFO, seed=7):
        self.config = config or MemoryConfig()
        self.mode = ManagerMode(mode)
        if self.mode == ManagerMode.BASELINE:
            raise ValueError("baseline_framesamp must use the upstream policy")
        self.seed = seed
        self.reset()

    @property
    def capacity(self):
        return self.config.capacity

    def reset(self, episode_id=None, exec_start_idx=0):
        if exec_start_idx < 0:
            raise ValueError("negative execution boundary")
        self.episode_id, self.exec_start_idx = episode_id, exec_start_idx
        self._slots = ()
        self.latest_frame_id = -1
        self.decision_id = 0
        self.recent_frame_ids = ()
        self.recent_window_approx = False
        self.last_candidate_cutoff = -1
        self.rng = np.random.default_rng(self.seed)

    def snapshot(self):
        return self._slots

    def sorted_valid_slots(self):
        return list(self._slots)

    def phase_slot_counts(self):
        return phase_counts([s.frame_id for s in self._slots], self.exec_start_idx)

    def _slot(self, frame, role, old=None):
        return MemorySlot(frame, role, phase_of(frame.frame_id, self.exec_start_idx),
                          old.sources if old else frozenset(), old.event_types if old else (),
                          old.event_ids if old else ())

    def _validate(self, slots, latest=None):
        F = self.latest_frame_id if latest is None else latest
        ids = [s.frame_id for s in slots]
        if len(ids) != min(F + 1, self.capacity) or ids != sorted(set(ids)):
            raise ValueError("bank must be chronologically sorted, unique and full after warmup")
        if not ids or ids[0] != 0 or any(s.frame.episode_id != self.episode_id for s in slots):
            raise ValueError("missing anchor or mixed episodes")
        if any(s.frame_id > F or s.frame.decision_id_available > self.decision_id for s in slots):
            raise ValueError("noncausal bank")
        if any(s.phase != phase_of(s.frame_id, self.exec_start_idx) for s in slots):
            raise ValueError("incorrect phase label")
        cfg = self.config
        if any(s.frame.image_feat.shape != (cfg.num_views, cfg.tokens_per_frame, cfg.image_dim)
               or s.frame.pos_feat.shape != (cfg.num_views, cfg.tokens_per_frame, cfg.position_dim)
               or s.frame.raw_robot_state.shape != (cfg.state_dim,) for s in slots):
            raise ValueError("slot feature shape differs from the configured tensor contract")
        if self.config.phase_locked_replacement:
            expected = phase_counts(partition(F, self.capacity, self.config.recent_slots).uniform, self.exec_start_idx)
            if phase_counts(ids, self.exec_start_idx) != expected:
                raise ValueError("phase slot quotas differ from current D0")

    def refresh_recent(self, F, rolling, decision_id=None):
        # A rejected refresh must not consume selection RNG or change availability.
        before = (self.episode_id, self.decision_id, deepcopy(self.rng.bit_generator.state))
        try:
            self._refresh_recent(F, rolling, decision_id)
        except Exception:
            self.episode_id, self.decision_id, state = before
            self.rng.bit_generator.state = state
            raise

    def _refresh_recent(self, F, rolling, decision_id=None):
        """Keep previous history where possible; rebalance only required roles/phase quotas.

        Exact D0 recent frames come from the ring. Beyond its supported time span,
        nearest same-phase available frames are explicit approximation, never disk reads.
        """
        if F != rolling.latest_frame_id or F < self.latest_frame_id:
            raise ValueError("refresh must use the latest contiguous observed frame")
        if self.episode_id is None:
            self.episode_id = rolling.episode_id
        if rolling.episode_id != self.episode_id:
            raise ValueError("ring and bank episodes differ")
        prior_decision = self.decision_id
        self.decision_id = prior_decision if decision_id is None else decision_id
        old = {s.frame_id: s for s in self._slots}
        pool = {s.frame_id: s.frame for s in self._slots}
        ring = {f.frame_id: f for f in rolling.snapshot()}
        pool.update(ring)
        part = partition(F, self.capacity, self.config.recent_slots)
        if 0 not in pool:
            raise ValueError("anchor unavailable: ingest warmup frames before evicting them")
        desired = list(part.recent)
        recent, used, approximate = [], {0}, False
        for frame_id in desired:
            chosen = frame_id
            # Once the requested window extends outside the ring, do not promote a
            # stale history slot to recent: that can strand the previous recent
            # slots after the new window and leave fewer than 32 eligible frames.
            if frame_id not in ring:
                if self.config.recent_overflow == "error":
                    raise ValueError("ring cannot cover exact recent window; increase configured capacity")
                reserved = set(desired) & set(ring)
                options = [i for i in ring if i not in used and i not in reserved
                           and phase_of(i, self.exec_start_idx) == phase_of(frame_id, self.exec_start_idx)]
                if not options:
                    # A long demonstration can have left the ring entirely. Its
                    # newest retained same-phase frames preserve enough older
                    # history to meet the decreasing D0 demonstration quota.
                    options = [i for i in pool if i not in used and i not in reserved
                               and phase_of(i, self.exec_start_idx) == phase_of(frame_id, self.exec_start_idx)]
                    if not options:
                        raise ValueError("cannot approximate recent window in the same phase")
                    chosen = max(options)
                else:
                    chosen = min(options, key=lambda i: (abs(i - frame_id), i))
                approximate = True
            # Approximate earlier entries may not consume a later exact requested frame.
            if chosen in used:
                raise ValueError("recent approximation collided")
            recent.append(chosen)
            used.add(chosen)
        recent.sort()
        if F < self.capacity:
            selected = list(range(F + 1))
        else:
            required = {0, *recent}
            selected = sorted(required)
            target = phase_counts(part.uniform, self.exec_start_idx)
            recent_start = min(recent)
            for phase in ("demo", "exec"):
                refs = [i for i in part.uniform if phase_of(i, self.exec_start_idx) == phase]
                required_phase = {i for i in required if phase_of(i, self.exec_start_idx) == phase}
                n = target[phase] - len(required_phase)
                eligible = [i for i in pool if i not in required and i < recent_start
                            and phase_of(i, self.exec_start_idx) == phase]
                kept = sorted(i for i in eligible if i in old and old[i].is_history)
                if len(kept) > n:
                    kept = (kept[-n:] if n else []) if self.mode == ManagerMode.FIFO else sorted(
                        self.rng.choice(kept, size=n, replace=False).tolist())
                # Use the validated coverage DP only to complete shortages, retaining existing history.
                candidates = sorted(required_phase | set(eligible))
                chosen = balanced_subset(candidates, required_phase | set(kept), target[phase], refs)
                selected += [i for i in chosen if i not in required]
            selected.sort()
        slots = tuple(self._slot(pool[i], "anchor" if i == 0 else "recent" if i in recent else "history", old.get(i))
                      for i in selected)
        self._validate(slots, latest=F)
        self._slots = slots
        self.latest_frame_id = F
        self.recent_frame_ids = tuple(recent)
        self.recent_window_approx = approximate

    def commit_insert(self, candidate):
        raise ValueError("warmup is filled from all observations; full banks only replace or reject")

    def commit_replace(self, slot_id, candidate):
        """slot_id is a source frame ID, not an index that can shift after sorting."""
        f = candidate.frame
        if f.episode_id != self.episode_id or f.frame_id > self.latest_frame_id:
            raise ValueError("candidate from a different episode or the future")
        if candidate.confirmed_at_frame > self.latest_frame_id or f.decision_id_available > self.decision_id:
            raise ValueError("candidate is not yet available")
        current = {s.frame_id: s for s in self._slots}
        if f.frame_id in current:
            merged = current[f.frame_id].merged(candidate)
            self._slots = tuple(merged if s.frame_id == f.frame_id else s for s in self._slots)
            return {"accepted": False, "reason": "duplicate_merged", "frame_id": f.frame_id}
        if slot_id not in current or current[slot_id].is_protected:
            raise ValueError("replacement target must be an unprotected history frame ID")
        if len(self._slots) != self.capacity or f.frame_id >= min(self.recent_frame_ids):
            raise ValueError("candidate must be older than the recent window in a full bank")
        victim = current[slot_id]
        phase = phase_of(f.frame_id, self.exec_start_idx)
        if self.config.phase_locked_replacement and phase != victim.phase:
            raise ValueError("cross-phase replacement is disabled")
        replacement = MemorySlot(f, "history", phase, candidate.sources, candidate.event_types, candidate.event_ids)
        proposed = tuple(sorted((replacement if s.frame_id == slot_id else s for s in self._slots),
                                key=lambda s: s.frame_id))
        self._validate(proposed)
        self._slots = proposed
        return {"accepted": True, "reason": "replaced", "frame_id": f.frame_id, "deleted_frame_id": slot_id}

    def apply_candidates(self, candidates):
        results = []
        for candidate in candidates:
            phase = phase_of(candidate.frame.frame_id, self.exec_start_idx)
            eligible = [s.frame_id for s in self._slots if s.is_history
                        and (not self.config.phase_locked_replacement or s.phase == phase)]
            if not eligible:
                results.append({"accepted": False, "reason": "no_same_phase_history", "frame_id": candidate.frame.frame_id})
                continue
            victim = min(eligible) if self.mode == ManagerMode.FIFO else int(self.rng.choice(eligible))
            results.append(self.commit_replace(victim, candidate))
        return results

    @property
    def nbytes(self):
        return sum(s.frame.nbytes for s in self._slots)
