#!/usr/bin/env python3
"""CPU-only R9 semantic/default checks and the accepted occluded task's read-only audit."""
import importlib.util
import json
from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from v_a_common import ECM, read_json, read_jsonl, sha256, timestamp
from fruit_r9_common import OUT, R8, proof, publish, preserve, d5_cases, recorded_distances
sys.path[:0] = [str(ECM), str(ECM / 'src')]
from fruit_picking.task_v2_spec import TaskV2Spec, task_layout, early_start
from memory_management import event_keyframes as ek
from memory_management.policy_adapter import install_temporal_index
from fruit_picking.task_v2_audit_r9 import MEMORIES, memory_frames, estimate_onset


def baseline(name, relative):
    spec = importlib.util.spec_from_file_location(name, OUT / 'baseline_sources' / relative)
    module = importlib.util.module_from_spec(spec);sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if (OUT / 'cpu_preflight.json').exists():
        old = read_json(OUT / 'cpu_preflight.json')
        if not old['passed']:raise RuntimeError('Failed check is retained; no automatic retry')
        print(old);return
    state = {'revision': 9, 'stage': 'R0', 'state': 'r9_cpu_checks', 'training': False, 'train_data_generated': False}
    publish(state)
    result = {'revision': 9, 'created_utc': timestamp(), 'passed': False, 'checks': {}}
    try:
        old = baseline('_r9_original_keyframes', 'src/memory_management/event_keyframes.py')
        selections = []
        cases = d5_cases();assert len(cases) == 20
        for row in cases:
            directory = Path(row['source']);distances = recorded_distances(directory)
            times = np.linspace(0, len(distances) - 1, 10, dtype=int).tolist()
            assert len(set(times)) == 10
            for t in times:
                for events in (0, 4, 8):
                    before = old.select(distances, t, old.KeyframeConfig(events=events))
                    after = ek.select(distances, t, ek.KeyframeConfig(events=events))
                    assert before == after, (directory, t, events)
            selections.append({'source': str(directory), 'times': times,
                               'event_signal_sha256': sha256(directory / 'event_signal.jsonl')})
        result['checks']['D5_default_selection'] = {'passed': True, 'episodes': 20, 'times_per_episode': 10,
                                                   'conditions': ['K0', 'K4', 'K8'], 'comparisons': 600, 'sources': selections}
        distances = ([.01, .01, .2] * 14) + [.01, .01]
        frames, info = ek.select(distances, len(distances) - 1, MEMORIES['KA'])
        assert info['keyframes'] == list(range(8, 42, 3))
        assert len(frames) == 16 and len(info['keyframes']) == 12 and 0 in frames
        result['checks']['KA_completed_releases_latest12'] = {'passed': True, 'keyframes': info['keyframes']}
        policy = SimpleNamespace(mem_buffer=object(), _prepare_mem_buffer=lambda: None)
        previous = dict(vars(policy));assert install_temporal_index(policy) is None and vars(policy) == previous
        result['checks']['absolute_adapter_is_noop'] = {'passed': True}
        spec = replace(TaskV2Spec(version='v2'), count_n_max=10, budget=2800, early_start_failure=True)
        q = np.zeros(7);tcp = np.zeros(3)
        assert not early_start(q, q, tcp, tcp, False, spec)
        assert early_start([.10001] + [0.] * 6, q, tcp, tcp, False, spec)
        assert early_start(q, q, [.03001, 0, 0], tcp, False, spec)
        assert not early_start([1.] * 7, q, [1.] * 3, tcp, True, spec)
        assert not early_start([1.] * 7, q, [1.] * 3, tcp, False, TaskV2Spec())
        result['checks']['early_start_thresholds_and_default'] = {'passed': True, 'arm_rad': .10, 'TCP_m': .03}
        old_spec = baseline('fruit_picking._r9_original_spec', 'fruit_picking/task_v2_spec.py')
        sampled = []
        for e in range(200, 260):
            seed = 4200000 + 16 * e
            for task in ('count', 'occluded_ripe'):
                assert task_layout(task, seed).to_json() == old_spec.task_layout(task, seed).to_json()
                assert task_layout(task, seed, 'v2').to_json() == old_spec.task_layout(task, seed, 'v2').to_json()
            layout = task_layout('count', seed, 'v2', spec);sampled.append(layout.n_pick)
            assert 2 <= layout.n_pick <= 10
        assert set(sampled) == set(range(2, 11))
        result['checks']['v1_and_R8_defaults_retained_R9_N2to10'] = {'passed': True, 'samples': sampled}
        midpoint, evidence = estimate_onset([1.] * 100, [False] * 30 + [True] * 70, 99)
        assert midpoint == (evidence['last_red'] + evidence['first_green']) / 2
        assert estimate_onset([1.] * 100, [None] * 100, 99)[0] is None
        result['checks']['K0_onset_uses_selected_visible_lamps'] = {'passed': True}
        occluded = [];counts = {k: 0 for k in MEMORIES}
        for path in sorted((R8 / 'replays/audit/occluded_ripe/history').glob('ep*/validity.json')):
            directory = path.parent;trace = read_jsonl(directory / 'trace.jsonl')
            distances = [f['signals']['any_hanging'] for f in trace];decisions = read_json(directory / 'decisions.json')
            for condition in MEMORIES:
                selected = [memory_frames(distances, d['step'], condition) for d in decisions]
                assert all(0 in ids for ids in selected), (directory, condition)
                counts[condition] += len(selected)
            occluded.append({'source': str(directory), 'decisions': len(decisions),
                              'trace_sha256': sha256(directory / 'trace.jsonl')})
        assert len(occluded) == 50
        result['checks']['occluded_frame0_readonly'] = {'passed': True, 'episodes': 50,
                                                     'all_decisions_retain_frame0': counts, 'sources': occluded}
        preserve()
        result['passed'] = True;result['GPU_tensor_checks'] = 'pending; must pass before new R0 simulations'
        proof('cpu_preflight.json', result);publish(state, state='r9_preparing_GPU_checks')
        print({'passed': True, 'CPU_checks': len(result['checks']), 'D5_selections': 600, 'occluded_replays': 0})
    except Exception as exc:
        result['error'] = str(exc);proof('cpu_preflight.json', result)
        publish(state, state='R9_unit_failed', reason=str(exc));raise


if __name__ == '__main__':main()
