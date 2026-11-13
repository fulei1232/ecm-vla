"""CPU tests for the opt-in guard and official evaluation-loop horizon adapter."""
from pathlib import Path
import ast
import collections
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.memory_management.fruit_execution import configure_execution_horizon, guard_joint_target, load_joint_guard


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.limits = {"lower": np.full(7, -2.), "upper": np.full(7, 2.), "v_max": np.full(7, .1)}

    def test_intersection_arm_and_gripper(self):
        p = np.array([9., -9., 0, 0, 0, 0, 9., -.73])
        q = np.array([1.99, -1.99, 0, 0, 0, 0, 1.99, .01])
        a, log = guard_joint_target(p, q, self.limits)
        np.testing.assert_allclose(a, [2., -2., 0, 0, 0, 0, 2., -.73])
        self.assertEqual(log["guard_clipped_joints"], [True, True, False, False, False, False, True])
        self.assertEqual(p[0], 9.)
        self.assertTrue(np.all(np.abs(a[:7]-q[:7]) <= .1 + 1e-12))

    def test_step_limit_uses_measured_not_last_target(self):
        q = np.full(7, .5)
        a, _ = guard_joint_target(np.array([1.]*7 + [1.]), q, self.limits)
        np.testing.assert_allclose(a[:7], .6)

    def test_empty_intersection_is_error(self):
        with self.assertRaisesRegex(ValueError, "no feasible"):
            guard_joint_target(np.zeros(8), np.full(7, 3.), self.limits)

    def test_nonfinite_or_wrong_dimension_rejected(self):
        for p in [np.full(8, np.nan), np.zeros(7)]:
            with self.assertRaises(ValueError):
                guard_joint_target(p, np.zeros(7), self.limits)

    def test_loaded_wrist_limit_and_gripper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guard.json"
            path.write_text(json.dumps({"version": "v1", "lower": [-9]*7, "upper": [9]*7, "v_max": [9]*7}))
            a, _ = guard_joint_target([0]*6 + [8, -.42], np.zeros(7), load_joint_guard(path))
            self.assertAlmostEqual(a[6], np.pi)
            self.assertEqual(a[7], -.42)

    def test_recorder_logs_the_actual_guarded_action(self):
        from src.memory_management.fruit_eval import FruitEnvRunner, FruitRawRecorder
        class Tensor:
            def __getitem__(self, index): return self
            def cpu(self): return self
            def numpy(self): return np.zeros(9)
        sent = []
        def step(action):
            sent.append(action.copy())
            return None, 0, False, False, {"success": [False], "fail": [False]}
        env = SimpleNamespace(unwrapped=SimpleNamespace(agent=SimpleNamespace(
            robot=SimpleNamespace(get_qpos=lambda: Tensor()))), step=step)
        runner = FruitEnvRunner("FruitCount", ".", 100, joint_guard=self.limits)
        runner.env = env
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        runner._frame = lambda obs: (image, image, np.zeros(8))
        with tempfile.TemporaryDirectory() as tmp, patch("src.memory_management.fruit_eval.snapshot", return_value={}):
            recorder = FruitRawRecorder(Path(tmp), "FruitCount")
            recorder.install(runner)
            proposed = np.array([9.]*7 + [.37])
            runner.step(proposed)
            recorder.flush()
            row = json.loads((Path(tmp) / "control_actions.jsonl").read_text())
            np.testing.assert_array_equal(row["action"], sent[0])
            np.testing.assert_array_equal(row["proposed_action"], proposed)
            self.assertEqual(row["action"][7], .37)
            self.assertTrue(row["guard_clipped"])

    def test_horizon_only_changes_action_slice_not_observation_buffer(self):
        class Evaluator:
            args = SimpleNamespace(obs_horizon=16)
            def get_action_chunk(self, client, state, *, exec_horizon):
                client.extend(state)
                return list(range(16))[:exec_horizon]
        e = Evaluator()
        configure_execution_horizon(e, 4)
        observations = list(range(25))
        ingested = []
        self.assertEqual(e.get_action_chunk(ingested, observations, exec_horizon=e.args.obs_horizon), list(range(4)))
        self.assertEqual(ingested, observations)
        self.assertEqual(e.args.obs_horizon, 16)

    def test_default_does_not_wrap_method(self):
        fn = lambda **kwargs: kwargs["exec_horizon"]
        e = SimpleNamespace(args=SimpleNamespace(obs_horizon=16), get_action_chunk=fn)
        self.assertEqual(configure_execution_horizon(e), 16)
        self.assertIs(e.get_action_chunk, fn)
        for h in [0, 17, 1.5]:
            with self.assertRaises(ValueError):
                configure_execution_horizon(e, h)

    def test_real_official_loop_ingests_every_frame_at_each_horizon(self):
        # Execute the original evaluator and EpisodeState class ASTs with CPU
        # client/env/recorder doubles. No copied or reimplemented rollout loop.
        policy = Path('/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/examples/robomme')
        def class_ast(path, name):
            tree = ast.parse(path.read_text())
            return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        for horizon in (16, 8, 4):
            ingested, executed = [], []
            class Client:
                def __init__(self, *args): pass
                def reset(self): return {"reset_finished": True}
                def add_buffer(self, obs):
                    ingested.extend(int(image[0, 0, 0, 0]) for image in obs["images"])
                    return {"add_buffer_finished": True}
                def infer(self, obs):
                    return {"actions": np.repeat(np.arange(16)[:, None], 8, axis=1)}
            class Recorder:
                def __init__(self, *args, **kwargs): pass
                def record(self, **kwargs): pass
                def save_video(self, name): pass
            class Runner:
                env_id, episode_id, difficulty = "FruitCount", 200, "easy"
                def get_init_obs(self):
                    image = np.zeros((2, 2, 3), dtype=np.uint8)
                    return {"images": [image], "wrist_images": [image], "states": [np.zeros(8)], "task_goal": "pick"}
                def step(self, action):
                    executed.append(int(action[0]))
                    t = len(executed)
                    image = np.full((2, 2, 3), t, dtype=np.uint8)
                    return (image, image, np.zeros(8)), t == 32, "success" if t == 32 else "ongoing"
            ns = {"np": np, "collections": collections, "Args": object, "EnvRunner": object,
                  "SubgoalPredictorBase": object, "Path": Path, "Tuple": __import__('typing').Tuple,
                  "Optional": __import__('typing').Optional, "Any": object, "RolloutRecorder": Recorder,
                  "TASK_WITH_VIDEO_DEMO": [], "_websocket_client_policy": SimpleNamespace(MMEVLAWebsocketClientPolicy=Client),
                  "pack_buffer": lambda images, states, idx: {"images": np.stack(images)[:, None]}}
            for filename, name in (("utils.py", "EpisodeState"), ("eval.py", "EpisodeEvaluator")):
                module = ast.Module(body=[class_ast(policy/filename, name)], type_ignores=[])
                exec(compile(module, str(policy/filename), 'exec'), ns)
            args = SimpleNamespace(obs_horizon=16, host='localhost', port=1, subgoal_type=None,
                                   subgoal_keep_period=1, use_history=True, max_steps=100)
            evaluator = ns["EpisodeEvaluator"](args, Path('.'))
            configure_execution_horizon(evaluator, horizon)
            subgoal = SimpleNamespace(start_episode=lambda *a: None, step=lambda *a: None,
                                      get_subgoal=lambda *a: (None, False), end_episode=lambda *a: None)
            self.assertEqual(evaluator.eval_each_episode(Runner(), subgoal, Path('.')), 'success')
            self.assertEqual(ingested, list(range(33-horizon)))
            self.assertEqual(executed, list(range(horizon))*(32//horizon))
            self.assertEqual(args.obs_horizon, 16)


if __name__ == "__main__":
    unittest.main()
