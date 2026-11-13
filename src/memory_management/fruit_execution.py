"""Opt-in execution controls; leave observations, history and gripper commands intact."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def load_joint_guard(path):
    value = json.loads(Path(path).read_text())
    if value.get("version") != "v1":
        raise ValueError("expected joint guard v1 statistics")
    limits = {key: np.asarray(value[key], dtype=np.float64) for key in ("lower", "upper", "v_max")}
    if any(v.shape != (7,) or not np.isfinite(v).all() for v in limits.values()):
        raise ValueError("joint guard requires seven finite limits")
    limits["lower"][6] = max(limits["lower"][6], -math.pi)
    limits["upper"][6] = min(limits["upper"][6], math.pi)
    if np.any(limits["lower"] > limits["upper"]) or np.any(limits["v_max"] <= 0):
        raise ValueError("invalid joint guard intervals or step limits")
    return limits


def guard_joint_target(proposed, measured, limits):
    """Intersect training/wrist envelope and measured-state step limits (arm only).

    An empty intersection is an engineering error, never an implicit relaxation.
    Limits are absolute joint targets; the policy has already undone DeltaActions.
    """
    proposed = np.asarray(proposed, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    if proposed.shape != (8,) or measured.ndim != 1 or len(measured) < 7:
        raise ValueError("expected 8-D action and at least 7 measured joints")
    if not np.isfinite(proposed).all() or not np.isfinite(measured[:7]).all():
        raise ValueError("non-finite action or measured state")
    lower = np.maximum(limits["lower"], measured[:7] - limits["v_max"])
    upper = np.minimum(limits["upper"], measured[:7] + limits["v_max"])
    if np.any(lower > upper):
        raise ValueError("joint guard has no feasible envelope/step intersection")
    executed = proposed.copy()
    executed[:7] = np.clip(proposed[:7], lower, upper)
    clipped = executed[:7] != proposed[:7]
    return executed, {"joint_guard": "v1", "guard_clipped": bool(clipped.any()),
                      "guard_clipped_joints": clipped.tolist(), "measured_arm_qpos": measured[:7].tolist()}


def configure_execution_horizon(evaluator, horizon=None):
    """Override the official loop's obs_horizon argument at its action-chunk boundary.

    With None, do not wrap or mutate anything. With N, the same official loop,
    get_action_chunk, pack_buffer and EpisodeState still ingest every observation;
    only the slice of the predicted 16 actions changes. No vendor file is edited.
    """
    if horizon is None:
        return evaluator.args.obs_horizon
    if not isinstance(horizon, int) or not 1 <= horizon <= 16:
        raise ValueError("execution horizon must be an integer in 1-16")
    original = evaluator.get_action_chunk

    def get_action_chunk(*args, **kwargs):
        kwargs["exec_horizon"] = horizon
        return original(*args, **kwargs)

    evaluator.get_action_chunk = get_action_chunk
    return horizon
