"""Server-side V-A instrumentation: ACM's PolicyLogger with the selector routed through online_variants.

The official selector is still called exactly once per decision; its D0 result is passed to `choose`.
Ground-truth rows are read from the client's raw_frames.jsonl and only pick frame indices.
"""
from __future__ import annotations

import json
from pathlib import Path

from .acm_copy.diagnostic_logging import PolicyLogger
from .online_variants import CONDITIONS, EpisodeState, choose

NEEDS_ROWS = frozenset({"D4L", "D4B", "RNDB", "RNDC"})


class RawFrameTail:
    """Incrementally reads complete lines of an append-only JSONL file."""

    def __init__(self, path):
        self.path, self.offset, self.rows = Path(path), 0, []

    def upto(self, max_id):
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n") + 1
        self.rows.extend(json.loads(line) for line in data[:end].splitlines() if line.strip())
        self.offset += end
        rows = [r for r in self.rows if r["frame_id"] <= max_id]
        if [r["frame_id"] for r in rows] != list(range(max_id + 1)):
            raise ValueError(f"raw_frames.jsonl does not cover frames 0..{max_id}")
        return rows


class VALogger(PolicyLogger):
    def reset(self):
        result = super().reset()
        self.episode, self.selection_log, self.tail = None, {}, None
        if self.context:
            if self.context.get("condition") not in CONDITIONS:
                raise ValueError(f"unknown V-A condition {self.context.get('condition')!r}")
            self.tail = RawFrameTail(Path(self.context["directory"]) / "raw_frames.jsonl")
        return result

    def select(self, step_idx, token_budget, token_per_image):
        uniform = [int(i) for i in self.selector_original(step_idx, token_budget, token_per_image)]
        if not self.context:
            raise RuntimeError("V-A server received a decision without an episode context")
        condition = self.context["condition"]
        if self.episode is None:
            self.episode = EpisodeState(self.context["task"], self.context["episode_id"], self.policy._seed,
                                        self.policy.exec_start_idx)
        if self.episode.exec_start_idx != self.policy.exec_start_idx:
            raise ValueError("exec_start_idx changed within an episode")
        rows = self.tail.upto(step_idx) if condition in NEEDS_ROWS else []
        selected, log = choose(condition, uniform, step_idx, rows, self.episode)
        if len(selected) * token_per_image > token_budget:
            raise ValueError("selection exceeds the token budget")
        self.selected = selected
        self.selection_log = {"d0_frame_ids": uniform, **{k: v for k, v in log.items() if k != "selected"}}
        return selected

    def emit(self, name, row):
        if name == "policy_decisions.jsonl":
            row = {**row, "v_a": self.selection_log}
        super().emit(name, row)
