"""Lazy readers for verified RoboMME planner demonstrations used by V-B/M3."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_DEMO_ROOT = Path(__file__).resolve().parents[3] / "data/val_demos"


def load_demo_metadata(task: str, episode_id: int, root: Path = DEFAULT_DEMO_ROOT) -> dict:
    manifest_path = Path(root) / task / f"ep{episode_id:02d}_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing planner-demo manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("task") != task or int(manifest.get("episode", -1)) != episode_id:
        raise ValueError(f"manifest identity mismatch: {manifest_path}")
    if manifest.get("split") not in {"val", "train"} or not manifest.get("verification", {}).get("action_finite"):
        raise ValueError(f"manifest is not a verified finite train/val demo: {manifest_path}")
    h5_path = Path(manifest["h5_path"])
    if not h5_path.is_absolute():
        # Generated manifests may store paths relative to the ECM checkout.
        # Resolve them before the policy loader changes cwd to its vendor tree.
        checkout_relative = manifest_path.parents[3] / h5_path
        h5_path = checkout_relative if checkout_relative.is_file() else h5_path.resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"missing demo HDF5: {h5_path}")
    return {**manifest, "manifest_path": str(manifest_path), "h5_path": str(h5_path)}


def _step_keys(episode: h5py.Group) -> list[int]:
    indices = sorted(int(key.removeprefix("timestep_")) for key in episode if key.startswith("timestep_"))
    if indices != list(range(len(indices))):
        raise ValueError("demo timesteps must be contiguous and start at zero")
    return indices


def load_demo_arrays(task: str, episode_id: int, root: Path = DEFAULT_DEMO_ROOT) -> dict:
    """Load compact state/action metadata; image access remains lazy via read_demo_frames."""
    manifest = load_demo_metadata(task, episode_id, root)
    states, actions, boundaries, demo_flags = [], [], [], []
    with h5py.File(manifest["h5_path"], "r") as file:
        episode = file[f"episode_{episode_id}"]
        indices = _step_keys(episode)
        for idx in indices:
            step = episode[f"timestep_{idx}"]
            joints = np.asarray(step["obs"]["joint_state"][()], dtype=np.float32)
            gripper = np.asarray(step["obs"]["gripper_state"][()], dtype=np.float32)
            action = np.asarray(step["action"]["joint_action"][()], dtype=np.float32)
            if joints.shape != (7,) or gripper.size < 1 or action.shape != (8,):
                raise ValueError(f"unexpected state/action shapes at timestep {idx}: {joints.shape}, {gripper.shape}, {action.shape}")
            states.append(np.concatenate((joints, gripper.reshape(-1)[:1])).astype(np.float32))
            actions.append(action)
            info = step["info"]
            boundaries.append(bool(np.asarray(info["is_subgoal_boundary"][()]).any()))
            demo_flags.append(bool(np.asarray(info["is_video_demo"][()]).any()))
        exec_start = next((i for i, flag in enumerate(demo_flags) if not flag), None)
        if exec_start is None or any(demo_flags[exec_start:]):
            raise ValueError("demo must have an optional contiguous video prefix followed by execution frames")
    if len(indices) != int(manifest["verification"]["timesteps"]):
        raise ValueError("HDF5 timestep count differs from manifest verification")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("demo contains non-finite state/action values")
    return {
        "manifest": manifest,
        "joint_state": np.stack(states),
        "joint_action": np.stack(actions),
        "subgoal_boundaries": np.flatnonzero(boundaries).astype(int),
        "is_video_demo": np.asarray(demo_flags, dtype=np.bool_),
        "exec_start_idx": exec_start,
        "task_goal": manifest["verification"]["task_goal"],
    }


def read_demo_frames(task: str, episode_id: int, frame_ids: list[int], root: Path = DEFAULT_DEMO_ROOT) -> np.ndarray:
    """Read only requested RGB frames, preserving caller order."""
    manifest = load_demo_metadata(task, episode_id, root)
    ids = [int(i) for i in frame_ids]
    if any(i < 0 or i >= int(manifest["verification"]["timesteps"]) for i in ids):
        raise IndexError("requested demo frame is outside the trajectory")
    with h5py.File(manifest["h5_path"], "r") as file:
        episode = file[f"episode_{episode_id}"]
        frames = [np.asarray(episode[f"timestep_{i}"]["obs"]["front_rgb"][()], dtype=np.uint8) for i in ids]
    return np.stack(frames) if frames else np.empty((0, 256, 256, 3), dtype=np.uint8)


def read_demo_wrist_frames(task: str, episode_id: int, frame_ids: list[int], root: Path = DEFAULT_DEMO_ROOT) -> np.ndarray:
    """Read requested wrist RGB frames, preserving caller order."""
    manifest = load_demo_metadata(task, episode_id, root)
    ids = [int(i) for i in frame_ids]
    if any(i < 0 or i >= int(manifest["verification"]["timesteps"]) for i in ids):
        raise IndexError("requested demo frame is outside the trajectory")
    with h5py.File(manifest["h5_path"], "r") as file:
        episode = file[f"episode_{episode_id}"]
        frames = [np.asarray(episode[f"timestep_{i}"]["obs"]["wrist_rgb"][()], dtype=np.uint8) for i in ids]
    return np.stack(frames) if frames else np.empty((0, 256, 256, 3), dtype=np.uint8)


def action_chunk(actions: np.ndarray, frame_id: int, horizon: int = 20) -> tuple[np.ndarray, int]:
    """Return [horizon, 8] expert actions, padding the tail with its final action."""
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 8 or not len(values):
        raise ValueError(f"expected nonempty [frames, 8] expert actions, got {values.shape}")
    if not 0 <= frame_id < len(values) or horizon <= 0:
        raise ValueError("invalid frame id or horizon")
    end = min(len(values), frame_id + horizon)
    valid = values[frame_id:end]
    padded = np.concatenate((valid, np.repeat(valid[-1:], horizon - len(valid), axis=0)), axis=0)
    return padded, len(valid)
