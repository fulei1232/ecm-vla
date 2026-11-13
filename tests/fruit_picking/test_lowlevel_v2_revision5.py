"""Revision5 boundaries: descriptive sublabels, no-outcome retries and clustered inference."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from fruit_lowlevel_v2_metrics_r5 import classify_failure, analyze_directory
from fruit_lowlevel_v2_eval_common import terminal_outcome, retry_allowed
from fruit_lowlevel_v2_statistics import contrast, holm, ratio_contrast
from v_a_common import atomic_json, sha256


class Revision5Tests(unittest.TestCase):
    def test_structural_count_errors_are_engineering_not_memory(self):
        row = {'task': 'FruitCount', 'n_pick': 1, 'outcome': 'fail', 'error': ''}
        raw = [{'gt': {'fruit_distance': .1, 'n_in_basket': n, 'detached': [False],
                       'in_basket': [False]}} for n in (0, 2)]
        c = classify_failure(row, raw, [])
        self.assertEqual(c['class'], 'invalid')
        self.assertEqual(c['environment_anomalies'], ['count_over_quota'])
        self.assertNotIn('count_over_quota', c['memory_errors'])

    def grasp_fixture(self, distance=.03, target_detached=False):
        raw = []
        for i in range(3):
            raw.append({'frame_id': i, 'control_step': i, 'gt': {
                'fruit_distance': .1, 'n_in_basket': 0, 'detached': [i == 2, target_detached],
                'ripe': [False, True], 'in_basket': [False, False], 'targets': [1],
                'fruit_positions': [[0, 0, .5], [distance if i < 2 else .2, 0, .5]]}})
        return {'task': 'FruitOccludedRipe', 'outcome': 'fail', 'error': ''}, raw

    def test_adjacency_uses_first_successful_grasp_and_never_changes_class(self):
        row, raw = self.grasp_fixture()
        c = classify_failure(row, raw, [])
        self.assertEqual(c['class'], 'memory')
        self.assertIn('adjacent_misgrasp_possible', c['descriptive_subtags'])
        self.assertEqual(c['unripe_grasp_events'][0]['grasp_start_frame'], 1)
        self.assertAlmostEqual(c['unripe_grasp_events'][0]['distance_to_nearest_attached_target_m'], .03)
        row, raw = self.grasp_fixture(.04)
        self.assertEqual(classify_failure(row, raw, [])['descriptive_subtags'], [])
        row, raw = self.grasp_fixture(.03, target_detached=True)
        self.assertEqual(classify_failure(row, raw, [])['descriptive_subtags'], [])

    def test_never_retry_terminal_even_if_result_or_audit_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertTrue(retry_allowed(d, 0))
            self.assertTrue(retry_allowed(d, 1))
            self.assertFalse(retry_allowed(d, 2))
            (d/'control_actions.jsonl').write_text(json.dumps({'outcome': 'fail'})+'\n')
            self.assertEqual(terminal_outcome(d), 'fail')
            self.assertFalse(retry_allowed(d, 0))
            (d/'control_actions.jsonl').unlink()
            atomic_json(d/'result.json', {'outcome': 'timeout', 'error': 'cleanup crashed'})
            self.assertFalse(retry_allowed(d, 0))

    def test_terminal_physics_blocks_retry_between_log_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp);atomic_json(d/'started.json', {'task': 'FruitCount'})
            gt = {'n_pick': 1, 'in_basket': [True], 'detached': [True], 'fruit_positions': [[0, 0, .2]]}
            (d/'raw_frames.jsonl').write_text(json.dumps({'gt': gt})+'\n')
            self.assertEqual(terminal_outcome(d), 'success')
            self.assertFalse(retry_allowed(d, 0))

    def test_holm_monotone_mapping(self):
        out = holm({'a': .01, 'b': .03, 'c': .04, 'd': .2})
        np.testing.assert_allclose(list(out.values()), [.04, .09, .09, .2])

    def test_paired_ties_and_seed_clustering(self):
        zeros = np.zeros((50, 3));ones = np.ones((50, 3))
        tied = contrast(zeros, zeros, 'tie', True, 300, 1000)
        self.assertEqual(tied['p_one_sided'], 1.)
        self.assertEqual(tied['ci95'], [0., 0.])
        positive = contrast(ones, zeros, 'positive', True, 300, 1000)
        self.assertEqual(positive['difference'], 1.)
        self.assertEqual(positive['ci95'], [1., 1.])
        self.assertLess(positive['p_one_sided'], .01)
        a = np.arange(50)[:, None] % 2
        x = contrast(a, np.zeros_like(a), 'same-name', False, 500)
        y = contrast(np.repeat(a, 3, axis=1), np.zeros((50, 3)), 'same-name', False, 500)
        self.assertEqual(x['ci95'], y['ci95'])

    def test_undefined_conditional_rates_are_not_zero(self):
        v = ratio_contrast(np.zeros((5, 3)), np.zeros((5, 3)), np.zeros((5, 3)), np.ones((5, 3)), 'undefined')
        self.assertIsNone(v['difference']);self.assertIsNone(v['ci95'])

    def test_seed_server_routing_keeps_three_conditions(self):
        import run_fruit_lowlevel_v2_eval as runner
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'OUT', Path(tmp)), \
             patch.object(runner.subprocess, 'Popen', return_value=MagicMock()), \
             patch('run_fruit_diagnosis.wait_ready'):
            seal = {'checkpoints': {c: '/checkpoint/'+c for c in ('K0', 'K4', 'K8')}}
            for lane, seed in enumerate((7, 8, 9)):
                server = runner.start_server(seed, lane, seal)
                self.assertEqual(server['config']['model_seed'], seed)
                self.assertEqual(server['config']['conditions'], ['K0', 'K4', 'K8'])
                self.assertEqual(server['config']['checkpoints'], seal['checkpoints'])

    def test_readiness_cli_writes_status_without_starting_servers(self):
        import run_fruit_lowlevel_v2_eval as runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);out = root/'out';out.mkdir();(root/'reports').mkdir()
            atomic_json(root/'reports/lowlevel_rescue_status.json', {'links': {}})
            with patch.object(runner, 'OUT', out), patch.object(runner, 'ECM', root), \
                 patch.object(runner, 'verify_seal', return_value={'revision': 5}), \
                 patch.object(runner, 'start_server') as start, patch.object(sys, 'argv', ['runner']):
                runner.main()
            start.assert_not_called()
            self.assertEqual(json.loads((out/'status.json').read_text())['state'], 'sealed_ready_for_gpu_eval')

    def test_report_end_to_end_on_synthetic_rows_without_val(self):
        import report_fruit_lowlevel_v2_eval as report
        import fruit_lowlevel_v2_statistics as stats
        templates = {}
        for task in ('FruitCount', 'FruitOccludedRipe'):
            _, m, _ = analyze_directory(ROOT/f'results/fruit_lowlevel_v2_stage5_20261103/validity/K4/{task}/ep200', -1.5764960646629333)
            templates[task] = m
        data = {}
        for stage, seeds in (('main', (7, 8, 9)), ('sensitivity', (7,))):
            rows = []
            for c in ('K0', 'K4', 'K8'):
                for task in templates:
                    for episode in range(100, 150):
                        for seed in seeds:
                            success = (episode+(c != 'K0')) % 3 == 0
                            m = deepcopy(templates[task]);m.update(outcome='success' if success else 'fail', basket_unsuccessful=not success)
                            rows.append({'stage': stage, 'condition': c, 'task': task, 'episode': episode, 'model_seed': seed,
                                'n_pick': 3, 'outcome': m['outcome'], 'server_session': f'{task}/{episode}/{seed}',
                                'metrics': m, 'guard_projection_steps': 0, 'source': 'synthetic',
                                'classification': {'class': 'success' if success else 'operation', 'memory_errors': [],
                                    'operation_tags': [], 'descriptive_subtags': [], 'environment_anomalies': [], 'unripe_grasp_events': []},
                                'event_recall_proxy': {'numerator': 1, 'denominator': 1}, 'contact_events': 1,
                                'event_slot_fraction': {'numerator': 1, 'denominator': 16}})
            data[stage] = rows
        original_contrast = contrast
        def inexpensive(a, b, name, primary=False):
            return original_contrast(a, b, name, primary, bootstrap_draws=50, permutation_draws=300)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);out = root/'out';(out/'groups').mkdir(parents=True);(root/'reports').mkdir()
            sealpath = root/'seal.json';atomic_json(sealpath, {'synthetic': True});digest = sha256(sealpath)
            atomic_json(root/'reports/lowlevel_rescue_status.json', {'links': {}})
            for i in range(400):atomic_json(out/f'groups/g{i}.json', {'complete': True, 'seal_sha256': digest})
            for seed in (7, 8, 9):
                ledger = []
                for rows in data.values():
                    for r in rows:
                        if r['model_seed'] == seed:
                            ledger.append({**{k:r[k] for k in ('stage', 'task', 'episode', 'model_seed', 'condition')},
                                'attempt': 0, 'terminal_task_outcome_found': r['outcome']})
                (out/f'attempts_seed{seed}.jsonl').write_text('\n'.join(json.dumps(r) for r in ledger)+'\n')
            with patch.object(report, 'OUT', out), patch.object(report, 'ECM', root), patch.object(report, 'SEAL', sealpath), \
                 patch.object(report, 'verify_seal', return_value={'synthetic': True}), \
                 patch.object(report, 'collect', side_effect=lambda stage, seal: data[stage]), \
                 patch.object(report, 'contrast', side_effect=inexpensive), patch.object(stats, 'BOOTSTRAP_DRAWS', 50):
                report.report({'synthetic': True})
            s = json.loads((out/'summary.json').read_text())
            self.assertEqual(s['episodes'], {'main': 900, 'sensitivity': 300})
            self.assertEqual(len(s['primary_holm_family']), 4)
            self.assertTrue((out/'REPORT.md').exists())


if __name__ == '__main__':
    unittest.main()
