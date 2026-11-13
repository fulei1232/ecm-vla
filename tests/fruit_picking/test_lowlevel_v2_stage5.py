"""Regression checks for stage5 routing, event audit and failure rules."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from fruit_lowlevel_v2_metrics import classify_failure, event_statistics, aggregate_event_statistics
from run_fruit_lowlevel_v2_stage5 import training_specs, K0


def state(i=0, basket=0, detached=False, ripe=True, distance=.2):
    return {'frame_id': i, 'gt': {'n_in_basket': basket, 'in_basket': [bool(basket)],
        'detached': [detached], 'ripe': [ripe], 'targets': [0] if ripe else [],
        'fruit_distance': distance, 'tcp_position': [0., 0., 0.]}}


class Stage5Tests(unittest.TestCase):
    def result(self, task='FruitCount', outcome='timeout'):
        return {'task': task, 'n_pick': 1, 'outcome': outcome, 'error': '', 'total_step_budget': 1300}

    def test_training_changes_only_memory_and_output_names(self):
        base = json.loads((K0/'arms/k0.json').read_text())
        for name, s in training_specs().items():
            r = deepcopy(s)
            self.assertEqual(s['sampler']['events'], int(name[1:]))
            r['sampler']['events'] = 0
            r['arm'] = 'k0'
            r['checkpoint_base_dir'] = base['checkpoint_base_dir']
            self.assertEqual(r, base)
            self.assertEqual(s['train']['seed'], 42)
            self.assertEqual(s['gpus'], list(range(8)))

    def test_more_than_quota_memory_priority_over_drop(self):
        raw = [state(), state(1, 2, True)]
        r = classify_failure(self.result(outcome='fail'), raw, [])
        self.assertEqual(r['class'], 'memory')
        self.assertIn('count_over_quota', r['memory_errors'])
        self.assertIn('dropped_fruit_outcome_fail', r['operation_tags'])

    def test_new_contact_after_quota_but_not_existing_contact(self):
        raw = [state(distance=.02), state(1, 1, True, distance=.02)]
        self.assertEqual(classify_failure(self.result(outcome='success'), raw, [])['class'], 'success')
        raw += [state(2, 1, True, distance=.2), state(3, 1, True, distance=.02)]
        self.assertIn('count_continued_picking_after_quota', classify_failure(self.result(), raw, [])['memory_errors'])

    def test_premature_stop_requires_open_still_and_time(self):
        controls = [{'action': [0.]*7+[1.]} for _ in range(1300)]
        raw = [state(i) for i in range(1301)]
        self.assertIn('count_premature_stop_proxy', classify_failure(self.result(), raw, controls)['memory_errors'])
        closed = deepcopy(controls)
        for c in closed:
            c['action'][7] = -1.
        self.assertEqual(classify_failure(self.result(), raw, closed)['class'], 'operation')
        moving = deepcopy(controls)
        for i, c in enumerate(moving):
            c['action'][0] = i*.01
        self.assertEqual(classify_failure(self.result(), raw, moving)['class'], 'operation')
        short_budget = self.result()
        short_budget['total_step_budget'] = 1299
        self.assertEqual(classify_failure(short_budget, raw, controls)['class'], 'operation')

    def test_unripe_detach_is_memory_even_if_dropped(self):
        raw = [state(ripe=False), state(1, detached=True, ripe=False)]
        r = classify_failure(self.result('FruitOccludedRipe', 'fail'), raw, [])
        self.assertEqual(r['class'], 'memory')
        self.assertIn('occluded_unripe_detached', r['memory_errors'])

    def test_engineering_error_is_not_mechanism_failure(self):
        r = self.result()
        r['error'] = 'client failure'
        self.assertEqual(classify_failure(r, [state()], [])['class'], 'invalid')

    def test_g1_k0_event_audit_on_real_logs(self):
        p = ROOT/'results/fruit_k0v2_gate_20261102/G1/FruitCount/ep220'
        r = event_statistics(p, {'frames': 16, 'events': 0, 'radius': .03, 'tol': .005})
        self.assertTrue(r['event_validity_passed'])
        self.assertEqual(r['designated_event_slot_fraction']['numerator'], 0)
        self.assertIsNotNone(r['zero_event_slot_explanation'])

    def test_event_slots_are_causal_and_release_not_true_detach(self):
        from memory_management.event_keyframes import KeyframeConfig, select
        distances = [.2]*31 + [.02]*10 + [.2]*10
        cfg = KeyframeConfig(events=4)
        raw = [state(i, distance=d) for i, d in enumerate(distances)]
        decisions, oracle = [], []
        for i, t in enumerate((30, 46)):
            frames, info = select(distances, t, cfg)
            decisions.append({'available_max_frame_id': t, 'selected_frame_ids': frames})
            oracle.append({'keyframes': info})
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'result.json').write_text(json.dumps({'task': 'FruitCount', 'episode_id': 200, 'condition': 'K4'}))
            for name, rows in [('raw_frames', raw), ('policy_decisions', decisions), ('oracle_decisions', oracle)]:
                (p/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            r = event_statistics(p, cfg.to_dict())
            self.assertTrue(r['event_validity_passed'])
            self.assertEqual(r['contact_events'], 1)
            self.assertEqual(r['detach_event_keyframes'], 1)
            self.assertEqual(r['physical_detach_events'], 0)
            self.assertEqual(r['designated_event_slot_fraction']['numerator'], 2)
            self.assertEqual(r['decisions'][0]['designated_event_frames'], [])
            self.assertEqual(aggregate_event_statistics([r])['all']['designated_event_slot_fraction']['denominator'], 32)
            decisions[1]['selected_frame_ids'] = decisions[0]['selected_frame_ids']
            (p/'policy_decisions.jsonl').write_text('\n'.join(json.dumps(v) for v in decisions)+'\n')
            self.assertFalse(event_statistics(p, cfg.to_dict())['event_validity_passed'])

    def test_client_uses_same_g1_physical_calls(self):
        def calls(file, names):
            tree = ast.parse(file.read_text())
            return [ast.dump(c) for c in ast.walk(tree) if isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name) and c.func.id in names]
        names = {'prefix_runner_class', 'prefix_recorder_class', 'install_observe_prefix'}
        self.assertEqual(calls(ROOT/'scripts/evaluate_fruit_rescue_episode.py', names),
                         calls(ROOT/'scripts/evaluate_fruit_lowlevel_v2_episode.py', names))


if __name__ == '__main__':
    unittest.main()
