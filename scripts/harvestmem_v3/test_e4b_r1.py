"""Focused CPU checks for the planner handoff and engineering gates."""
import json
from contextlib import contextmanager
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import eval_episode as evaluator
from e4b_r1_gates import audit
import e4b_r1_preflight as preflight
from e4b_inference_probe import install_probe


class Sink:
    def flush(self):
        pass


class FakeEpisode:
    def __init__(self, task, episode, arm, out):
        self.task, self.episode, self.arm, self.out = task, episode, arm, out
        self.u = types.SimpleNamespace(control_steps=0, layout=types.SimpleNamespace(n_pick=2),
            original_geometry=[{'ripe': True}])
        self.seed = 1
        self.budget = 100
        self.instruction = 'test'
        self.self_checks = {'slot_xy': {'0':[1, 2]}}
        self.oracle = types.SimpleNamespace(n_pick=2)
        self.latest_oracle = types.SimpleNamespace(events=[], phase='pick', subgoal='pick', target_slot=0)
        self.last_outcome = 'ongoing'
        self.last_info = {'n_in_basket':[0], 'v2_reason':'running'}
        self.planner_images, self.planner_states = [], []
        self.planner_calls = self.planner_replans = self.planner_steps = 0
        self.planner_would_clip_steps = self.planner_projection_steps = 0
        self.planner_failures, self.handoffs = [], []
        self.stop_trigger_step = self.stop_trigger_n = self.stop_arrive_step = self.stop_end_step = None
        self.stop_primitive_timeout = False
        self.actions_file = self.trace_file = self.timeline_file = Sink()
        self.recorded = []

    def reset(self):
        self.front = self.wrist = np.array([0], dtype=np.uint8)
        self.state = np.array([0] * 8, dtype=np.float32)
        return self.front, self.wrist, self.state

    def step(self, action, guard=True):
        self.u.control_steps += 1
        t = self.u.control_steps
        self.front = self.wrist = np.array([t], dtype=np.uint8)
        self.state = np.full(8, t, dtype=np.float32)
        self.latest_oracle.events = ['target_selected'] if t == 30 else []
        if guard:
            self.last_outcome = 'timeout'
        return self.front, self.wrist, self.state, guard, self.last_outcome, {'guard_clipped':guard, 'guard_projection':False}

    def should_stop(self, *args):
        return False

    def maybe_planner(self, args):
        if args.planner_reach == 'off' or self.planner_calls:
            return
        self.planner_calls = 1
        for _ in range(2):
            self.step(None, guard=False)
            self.planner_images.append(self.front.copy())
            self.planner_states.append(self.state.copy())
            self.planner_steps += 1
        self.handoffs.append({'timestep':self.u.control_steps, 'target_slot':0})

    def write_action(self, action, source, **kwargs):
        self.recorded.append((source, kwargs))

    def close(self):
        pass


class HandoffTests(unittest.TestCase):
    def test_inference_probe_preserves_output_and_records_history(self):
        class Policy:
            step_idx, exec_start_idx = 30, 30
            _rng = np.array([0, 7], dtype=np.uint32)
            def _prepare_history(self, obs):
                return {**obs, 'history': np.ones((1, 2), dtype=np.float32)}
            def infer(self, obs):
                self._prepare_history(obs)
                return {'actions':np.arange(16, dtype=np.float32).reshape(2, 8)}
        with tempfile.TemporaryDirectory() as tmp:
            policy = Policy()
            original = policy.infer({'state':np.zeros(7)})
            install_probe(policy, tmp)
            fake_jax = types.SimpleNamespace(random=types.SimpleNamespace(key_data=lambda x:x))
            with mock.patch.dict('sys.modules', {'jax':fake_jax}):
                result = policy.infer({'state':np.zeros(7)})
            np.testing.assert_array_equal(result['actions'], original['actions'])
            row = json.loads((Path(tmp) / 'infer_000.json').read_text())
            self.assertEqual(row['rng']['dtype'], 'uint32')
            self.assertEqual(row['history_inputs']['history']['shape'], [1, 2])

    def test_planner_records_guard_fields_and_frames(self):
        planner = object.__new__(evaluator.Episode)
        planner.u = types.SimpleNamespace(agent=types.SimpleNamespace(robot=types.SimpleNamespace(
            get_qpos=lambda: [types.SimpleNamespace(cpu=lambda:types.SimpleNamespace(numpy=lambda:np.zeros(9)))])))
        planner.planner_would_clip_steps = planner.planner_projection_steps = planner.planner_steps = 0
        planner.planner_images, planner.planner_states = [], []
        planner.front, planner.state = np.array([4]), np.array([5])
        planner.enforce_stop = False
        planner.write_action = mock.Mock()
        planner.step = mock.Mock(return_value=('frame',))
        planner.should_stop = mock.Mock(return_value=False)
        fake_guard = types.ModuleType('memory_management.fruit_rescue_platform')
        fake_guard.guard_v2 = lambda action, q, guard: (action, {'guard_clipped':False, 'guard_projection':True})
        with mock.patch.dict('sys.modules', {'memory_management.fruit_rescue_platform':fake_guard}):
            self.assertEqual(planner.planner_step(np.zeros(8), {'target':0,'phase':'reach'}), ('frame',))
        self.assertEqual((planner.planner_would_clip_steps, planner.planner_projection_steps), (0, 1))
        self.assertFalse(planner.write_action.call_args.kwargs['would_clip'])
        self.assertTrue(planner.write_action.call_args.kwargs['guard_projection'])
        self.assertEqual(planner.planner_images[0].tolist(), [4])

    def test_fresh_observation_and_history_with_planner_and_off(self):
        for mode, expected in [('first', 32), ('off', 30)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                received = {}
                class Client:
                    def __init__(self, *args): pass
                    def reset(self): return {'reset_finished':True}
                    def add_buffer(self, payload): received['buffer'] = payload
                    def infer(self, obs):
                        received['obs'] = obs
                        return {'actions':np.zeros((1, 8), dtype=np.float32)}
                package = types.ModuleType('openpi_client')
                package.websocket_client_policy = types.SimpleNamespace(MMEVLAWebsocketClientPolicy=Client)
                utils = types.ModuleType('utils')
                utils.pack_buffer = lambda images, states, idx: (list(images), list(states), idx)
                args = types.SimpleNamespace(task='count', episode=200, arm='DP1R', out=Path(tmp),
                    enforce_stop=False, planner_reach=mode, capture_handoff_frames=False,
                    checkpoint='harvestmem_v3_S2', port=1, context=None, memory='uniform',
                    debug_stop_at_step=None, max_steps=40)
                with mock.patch.object(evaluator, 'Episode', FakeEpisode), mock.patch.object(evaluator, '_instruction_selfcheck', return_value={}), mock.patch.dict('sys.modules', {'openpi_client':package, 'utils':utils}):
                    result = evaluator.run_episode(args)
                self.assertFalse(result['error'])
                self.assertEqual(int(received['obs']['observation/state'][0]), expected)
                self.assertEqual([int(x[0]) for x in received['buffer'][0]], list(range(expected + 1)))
                self.assertEqual(received['buffer'][2], 30)

    def test_missing_eighth_frame_fails_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            result = {'task':'occluded_ripe','slot_xy':{'0':[1,2]},'N':1,'outcome':'fail','error':'',
                      'planner_calls':1,'planner_failures':[], 'handoffs':[{'timestep':1,'target_slot':0}]}
            (p/'result.json').write_text(json.dumps(result))
            (p/'actions.jsonl').write_text('\n'.join(json.dumps(x) for x in [
                {'step':0,'source':'planner','action':[0]*8}, {'step':1,'source':'policy','action':[0]*8}]))
            fruit = {'present':True,'attached':False,'wind_detached':False,'pos':[0,0,0]}
            (p/'trace.jsonl').write_text(json.dumps({'timestep':1,'tcp_pos':[0,0,0],'fruits':[fruit]}))
            self.assertFalse(audit(p)['pass'])

    def test_preflight_stops_on_first_failed_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'preflight'
            @contextmanager
            def server(*args):
                yield root / 'context.json'
            def logged(label, *args, **kwargs):
                if label.startswith('open_loop_'):
                    (root / f'{label}.json').write_text(json.dumps({'details':[{}], 'aggregate_ratio':1}))
            def smoke(*args):
                path = root / 'smoke/DP1R/count/ep202'
                path.mkdir(parents=True)
                return path
            with mock.patch.object(preflight, 'PREFLIGHT', root), mock.patch.object(preflight, 'server', server), \
                 mock.patch.object(preflight, 'run_logged', side_effect=logged), \
                 mock.patch.object(preflight, 'smoke', side_effect=smoke) as run_smoke, \
                 mock.patch.object(preflight, 'audit', return_value={'pass':False}), \
                 mock.patch.object(preflight.subprocess, 'run'), mock.patch.object(preflight.socket, 'socket'):
                with self.assertRaisesRegex(RuntimeError, 'smoke gate failed'):
                    preflight.main()
            self.assertEqual(run_smoke.call_count, 1)
            self.assertFalse(json.loads((root / 'gates.json').read_text())['all_pass'])
            self.assertTrue((root / 'STOPPED.txt').is_file())


if __name__ == '__main__':
    unittest.main()
