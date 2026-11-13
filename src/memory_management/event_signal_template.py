"""Template for the perception module behind the event signal (event_memory.ExternalDistance).

Select it in the stage's oracle params:
    "event": {..., "signal": {"kind": "external",
                              "factory": "memory_management.event_signal_template:EndEffectorTargetDistance",
                              "kwargs": {...}, "path": None}}
`path` is added to sys.path of the policy server (Python of POLICY_PYTHON) before importing `factory`.

The policy server builds one instance, calls reset(context) at the start of every episode, then
distance(frame) exactly once per frame, in frame order, as the policy ingests it (the demo video first).
A call only ever sees frames up to the current one. The value chooses frame indices only; it never
enters the model input.
"""
from __future__ import annotations

from .validation.event_memory import FrameInput


class EndEffectorTargetDistance:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def reset(self, context: dict) -> None:
        """context of the new episode: task, episode_id, condition, model_seed, directory, ..."""

    def distance(self, frame: FrameInput) -> float | None:
        """Metres between the end effector and the target grasp pose at frame.frame_id; None if unknown.

        frame.task: task name; frame.image: front camera (H, W, 3) uint8 exactly as the policy receives it;
        frame.state: the robot state (8,) the policy receives; frame.is_demo: True for demo-video frames.
        """
        raise NotImplementedError("plug in the perception module here")
