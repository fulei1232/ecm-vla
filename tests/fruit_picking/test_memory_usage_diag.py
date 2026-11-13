"""D6 opt-in isolation, physical-clock ablations, audits and prewritten verdict boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from fruit_memory_diag_core import select_frames, retained_metrics, hidden_metrics, diagnose_rule
from memory_management.fruit_memory_diag import DiagnosticOracleLogger
from memory_management.validation.teacher_oracle import OracleLogger


class MemoryDiagnosisTests(unittest.TestCase):
    def test_normal_is_identical_to_frozen_k0_selector(self):
        from memory_management.event_keyframes import select, KeyframeConfig
        for t in (0, 7, 15, 16, 30, 46, 1260):
            self.assertEqual(select_frames(t), select(np.zeros(t+1), t, KeyframeConfig(events=0))[0])

    def test_ablation_uses_real_frame_ids_and_physical_clock(self):
        self.assertEqual(select_frames(30, 'recent16'), list(range(15, 31)))
        self.assertEqual(select_frames(30, 'no_pregust'), list(range(20, 31)))
        ids = select_frames(46, 'no_pregust')
        self.assertEqual(len(ids), 16);self.assertEqual(ids[0], 20);self.assertEqual(ids[-1], 46)
        self.assertTrue(all(i >= 20 for i in ids))

    def test_server_adapter_defaults_to_original_logger(self):
        import serve_fruit_memory_diag as entry
        original = entry.original.OracleLogger
        with patch.object(sys, 'argv', ['serve', '--config', 'dummy']), patch.object(entry.original, 'main') as main:
            entry.main()
            main.assert_called_once()
            self.assertIs(entry.original.OracleLogger, original)

    def test_disabled_logger_rejects_explicit_ablation(self):
        obj = DiagnosticOracleLogger.__new__(DiagnosticOracleLogger)
        obj.diagnostics_enabled = False;obj.memory_ablation = None
        obj.context = {'memory_ablation': 'recent16'};obj.condition = 'K0'
        with patch.object(OracleLogger, 'reset', return_value=None):
            with self.assertRaises(ValueError):obj.reset()

    def test_logger_replaces_pending_selection_before_packing(self):
        obj = DiagnosticOracleLogger.__new__(DiagnosticOracleLogger)
        obj.memory_ablation = 'no_pregust';obj._defer_oracle = False
        obj.policy = SimpleNamespace(step_idx=30, exec_start_idx=30)
        obj.signal = None;obj.context = {'task': 'FruitOccludedRipe'}
        emitted = []
        def baseline(self, obs, teacher):
            self.pending = (30, select_frames(30))
            self.oracle_log = {'uniform': select_frames(30), 'keyframes': {'events': 0, 'keyframes': []}}
            self.emit('oracle_decisions.jsonl', self.oracle_log)
        with patch.object(OracleLogger, 'decide', baseline), patch.object(OracleLogger, 'emit', side_effect=lambda name, row: emitted.append((name, row))):
            obj.decide({}, {})
        self.assertEqual(obj.pending, (30, list(range(20, 31))))
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][1]['selected_frame_ids'], list(range(20, 31)))
        self.assertEqual(emitted[0][1]['memory_ablation'], 'no_pregust')

    def test_pre_gust_is_not_policy_relative_time(self):
        raw = [{'control_step': i, 'policy_control_step': i-30, 'gt': {'targets': [0, 1], 'detached': [False, False]}} for i in range(31)]
        normal = [{'decision_idx': 0, 'available_max_frame_id': 30, 'selected_frame_ids': select_frames(30)}]
        recent = [{**normal[0], 'selected_frame_ids': select_frames(30, 'recent16')}]
        self.assertEqual(retained_metrics(raw, normal)['all']['numerator'], 1)
        self.assertEqual(retained_metrics(raw, recent)['all']['numerator'], 0)

    def test_abc_boundaries_use_signed_changes_and_exact_counts(self):
        def rows(success_removed, errors=0):
            result = []
            for mode, successes in (('K0_normal', 40), ('K0_no_pregust', success_removed)):
                for i in range(80):
                    result.append({'condition': mode, 'task': 'FruitOccludedRipe', 'outcome': 'success' if i < successes else 'fail',
                        'hidden': {'omission_plus_wrong_fruit': int(mode == 'K0_no_pregust' and i < errors)}})
            return result
        self.assertEqual(diagnose_rule(rows(32))['verdict'], 'A')
        self.assertEqual(diagnose_rule(rows(33))['verdict'], 'C')
        self.assertEqual(diagnose_rule(rows(36))['verdict'], 'C')
        self.assertEqual(diagnose_rule(rows(37))['verdict'], 'B')
        self.assertEqual(diagnose_rule(rows(50))['verdict'], 'B')
        self.assertEqual(diagnose_rule(rows(40, 8))['verdict'], 'A')

    def test_diagnostic_audit_detects_excluded_frame_and_padding_tamper(self):
        from audit_fruit_memory_diag import audit
        from v_a_common import read_json, read_jsonl
        source = ROOT/'results/fruit_k0v2_gate_20261102/G1/FruitCount/ep220'
        params = read_json(ROOT/'results/fruit_dart_kf_eval_20261027/protocol.json')['params']['oracle']
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ('result.json', 'raw_frames.jsonl', 'control_actions.jsonl', 'client_ingest.jsonl', 'policy_ingest.jsonl',
                         'policy_decisions.jsonl', 'client_decisions.jsonl', 'policy_actions.jsonl', 'oracle_decisions.jsonl', 'policy_lifecycle.jsonl', 'event_signal.jsonl'):
                shutil.copy2(source/name, d/name)
            self.assertTrue(audit(d, None, params)['passed'])
            raw = read_jsonl(d/'raw_frames.jsonl');decisions = read_jsonl(d/'policy_decisions.jsonl');oracles = read_jsonl(d/'oracle_decisions.jsonl')
            for row, oracle in zip(decisions, oracles):
                ids = select_frames(row['available_max_frame_id'], 'no_pregust')
                row.update(memory_ablation='no_pregust', selected_frame_ids=ids, source_position_frame_ids=ids,
                    selected_control_steps=[raw[k]['policy_control_step'] for k in ids],
                    selected_front_sha256=[raw[k]['front_sha256'] for k in ids],
                    valid_tokens=len(ids)*16, padding_tokens=512-len(ids)*16,
                    mask=[True]*(len(ids)*16)+[False]*(512-len(ids)*16))
                oracle.update(memory_ablation='no_pregust', selected_frame_ids=ids)
            def save():
                for name, rows in (('policy_decisions', decisions), ('oracle_decisions', oracles)):
                    (d/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            save();self.assertTrue(audit(d, 'no_pregust', params)['passed'])
            decisions[0]['selected_frame_ids'][0] = 19
            save();self.assertFalse(audit(d, 'no_pregust', params)['passed'])

    def test_repro_compare_is_numeric_bitwise_not_tolerant(self):
        from run_fruit_memory_usage_diag import bitwise_compare
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp)/'a';b = Path(tmp)/'b';a.mkdir();b.mkdir()
            for d, zero in ((a, -0.), (b, 0.)):
                (d/'result.json').write_text('{}')
                for name in ('raw_frames', 'policy_actions', 'policy_decisions'):(d/f'{name}.jsonl').write_text('')
                (d/'control_actions.jsonl').write_text(json.dumps({'action': [zero]})+'\n')
            self.assertFalse(bitwise_compare(a, b)['bitwise_equal'])

    def test_prepare_cli_never_starts_servers(self):
        import run_fruit_memory_usage_diag as runner
        from v_a_common import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);out = root/'out';out.mkdir();(root/'reports').mkdir()
            atomic_json(root/'reports/lowlevel_rescue_status.json', {'links': {}})
            state = {'state': 'preparing_diagnosis'}
            with patch.object(runner, 'OUT', out), patch.object(runner, 'ECM', root), \
                 patch.object(runner, 'prepare', return_value=(state, {'deadline_utc': '2099-01-01T00:00:00+00:00'})), \
                 patch.object(runner, 'start_server') as start, patch.object(sys, 'argv', ['runner']):
                runner.main()
            start.assert_not_called()
            self.assertEqual(json.loads((out/'status.json').read_text())['state'], 'ready_for_gpu_diagnostic')

    def test_complete_report_path_with_synthetic_cohort_never_uses_val(self):
        import report_fruit_memory_usage as reporter
        from fruit_memory_diag_core import checked_layout, hidden_metrics, retained_metrics, count_failure_metrics, MODES, TASKS, G1
        from analyze_fruit_failure_modes import analyze_episode
        from v_a_common import read_json, read_jsonl, atomic_json
        templates = {}
        for task in TASKS:
            d = G1/f'G1/{task}/ep220';result = read_json(d/'result.json');raw = read_jsonl(d/'raw_frames.jsonl')
            controls = read_jsonl(d/'control_actions.jsonl')
            t = {'metrics': analyze_episode(d), 'audited_decisions': 1, 'reused_G1': False, 'source': 'synthetic'}
            if task == 'FruitOccludedRipe':
                t.update(hidden=hidden_metrics(checked_layout(result, raw), raw),
                         retention=retained_metrics(raw, read_jsonl(d/'policy_decisions.jsonl')))
            else:t['count_failure'] = count_failure_metrics(raw, controls, result['outcome'])
            templates[task] = t
        rows = []
        for condition in MODES:
            for task in TASKS:
                for e in range(220, 260):
                    for seed in (7, 8):
                        r = deepcopy(templates[task]);outcome = 'success' if (e+seed)%2 else 'fail'
                        r.update(condition=condition, task=task, episode=e, model_seed=seed, outcome=outcome)
                        r['metrics']['outcome'] = outcome
                        rows.append(r)
        post = read_json(ROOT/'results/fruit_memory_usage_diag_20261104/posthoc/summary.json')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);out = root/'out';(out/'posthoc').mkdir(parents=True);(out/'preservation').mkdir();(root/'reports').mkdir()
            atomic_json(out/'posthoc/summary.json', post)
            atomic_json(out/'preservation/sealed_files_sha256.json', {})
            atomic_json(out/'preservation/d5_tree_before.json', {})
            atomic_json(root/'reports/lowlevel_rescue_status.json', {'links': {}})
            with patch.object(reporter, 'OUT', out), patch.object(reporter, 'ECM', root), \
                 patch.object(reporter, 'collect', return_value=rows), patch.object(reporter, 'verify_preservation'):
                reporter.deliver({'state': 'diagnostic_dev220_259'}, complete=True)
            summary = read_json(out/'summary.json')
            self.assertEqual(summary['inference_episodes'], 480)
            self.assertEqual(summary['diagnosis_rule_14_5']['verdict'], 'B')
            self.assertEqual(summary['state']['state'], 'awaiting_D6_review')
            self.assertTrue(summary['D5_confirmatory_conclusions_unchanged'])


if __name__ == '__main__':unittest.main()
