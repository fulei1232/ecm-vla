"""Closed-loop evaluation of fruit-picking policies with the official MME-VLA episode loop.

`FruitEnvRunner` gives FruitPick-v1 the interface of examples/robomme/env_runner.EnvRunner, so the official
`EpisodeEvaluator.eval_each_episode` (websocket client, buffer packing, 16-step execution, timeout) runs
unchanged. Observations match the training demonstrations (fruit_picking/generate.py): front and hand camera
RGB at 256x256, state = 7 joints + first finger, absolute joint targets + gripper command as action.
Episodes are the seeds of the demonstrator's verified val episodes (<root>/<task>/ep<e>_manifest.json).

`FruitRawRecorder` is validation.acm_copy.diagnostic_logging.RawRecorder with the fruit simulator snapshot: every frame's
row in raw_frames.jsonl carries `gt` with the tcp, fruit states and `fruit_distance` (fruit_distance.from_env),
which fruit_distance.RecordedDistance serves to the policy server as the distance module's output.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from . import fruit_distance as fd
from .validation.acm_copy.diagnostic_logging import RawRecorder, append, digest, simple

TASKS = {"FruitCount": "count", "FruitOccludedRipe": "occluded_ripe"}


def pack_state(qpos, n_joints):
    return np.concatenate([qpos[:n_joints], qpos[n_joints:n_joints + 1]]).astype(np.float32)


class FruitEnvRunner:
    def __init__(self, env_id, root, max_steps, robot="rm75b_eg2", joint_guard=None):
        if env_id not in TASKS:
            raise ValueError(f"unknown fruit task {env_id!r}")
        self.env_id, self.task, self.robot, self.max_steps = env_id, TASKS[env_id], robot, max_steps
        self.root = Path(root) / self.task
        self.env = self.episode_id = self.seed = None
        self.task_goal, self.difficulty, self.info = "", "easy", {}
        self.joint_guard = joint_guard
        self.last_execution = None

    @property
    def num_episodes(self):
        return sum(1 for _ in self.root.glob("ep*_manifest.json"))

    def manifest(self, episode_id):
        return json.loads((self.root / f"ep{episode_id}_manifest.json").read_text())

    def make_env(self, episode_id):
        import gymnasium as gym
        import fruit_picking.env  # noqa: F401  (registers FruitPick-v1 and the agents)
        self.seed = int(self.manifest(episode_id)["seed"])
        self.env = gym.make("FruitPick-v1", task=self.task, robot_uids=self.robot, obs_mode="rgb+depth+segmentation",
                            control_mode="pd_joint_pos", render_mode="rgb_array", sim_backend="cpu",
                            max_episode_steps=self.max_steps + 1)
        self.episode_id = episode_id

    def _frame(self, obs):
        sd = obs["sensor_data"]
        n = len(self.env.unwrapped.agent.arm_joint_names)
        return (sd["front_camera"]["rgb"][0].cpu().numpy().astype(np.uint8),
                sd["hand_camera"]["rgb"][0].cpu().numpy().astype(np.uint8),
                pack_state(obs["agent"]["qpos"][0].cpu().numpy(), n))

    def get_init_obs(self):
        obs, self.info = self.env.reset(seed=self.seed)
        self.task_goal = self.env.unwrapped.layout.instruction
        front, wrist, state = self._frame(obs)
        return {"images": [front], "wrist_images": [wrist], "states": [state], "task_goal": self.task_goal}

    def step(self, action):
        self.last_execution = None
        if self.joint_guard is not None:
            from .fruit_execution import guard_joint_target
            measured = self.env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
            proposed = np.asarray(action).copy()
            action, audit = guard_joint_target(proposed, measured, self.joint_guard)
            self.last_execution = {"proposed_action": simple(proposed), "action": simple(action), **audit}
        try:
            obs, _, terminated, truncated, self.info = self.env.step(np.asarray(action))
        except Exception as e:  # same contract as EnvRunner.step
            print(f"Error: {e}")
            return (None, None, None), True, "error"
        success, fail = bool(self.info["success"][0]), bool(self.info["fail"][0])
        outcome = "success" if success else "fail" if fail else "timeout" if bool(truncated) else "ongoing"
        return self._frame(obs), bool(terminated) or bool(truncated), outcome

    def close_env(self):
        if self.env is not None:
            self.env.close()
            self.env = self.episode_id = None


def snapshot(env):
    u = env.unwrapped
    inside = u.in_basket()
    return {"tcp_position": simple(u.agent.tcp_pose.p[0]), "fruit_positions": simple(u.fruit_positions()),
            "detached": [bool(a) for a in u.attached], "in_basket": [bool(x) for x in inside],
            "ripe": fd.ripe_flags(u.layout), "targets": list(u.layout.targets), "n_pick": int(u.layout.n_pick),
            "n_in_basket": int(inside.sum()), "control_steps": int(u.control_steps),
            "fruit_distance": fd.from_env(u)}


class FruitRawRecorder(RawRecorder):
    def install(self, runner):
        self.runner = runner
        original_init, original_step = runner.get_init_obs, runner.step

        def details(front):
            return {"front_sha256": digest(front), "gt": snapshot(runner.env), "timestamp_ns": time.time_ns()}

        def init():
            result = original_init()
            self.exec_start = len(result["images"]) - 1
            for i, (front, wrist, state) in enumerate(zip(result["images"], result["wrist_images"], result["states"])):
                self.add(front, wrist, state, details(front), i, i - self.exec_start, i < self.exec_start)
            self.flush()
            return result

        def step(action):
            result = original_step(action)
            self.control_step += 1
            (front, wrist, state), stop, outcome = result
            if front is not None:
                self.add(front, wrist, state, details(front), len(self.rows), self.control_step, False)
            record = {"control_step": self.control_step,
                      "action": simple(action), "stop": bool(stop), "outcome": outcome,
                      "raw_frame_id": len(self.rows) - 1 if front is not None else None}
            execution = getattr(runner, "last_execution", None)
            if execution is not None:
                record.update(execution)
            elif getattr(runner, "log_execution", False):
                record.update(proposed_action=simple(action), joint_guard="off", guard_clipped=False,
                              guard_clipped_joints=[False] * 7)
            append(self.directory / "control_actions.jsonl", record)
            return result

        runner.get_init_obs, runner.step = init, step
