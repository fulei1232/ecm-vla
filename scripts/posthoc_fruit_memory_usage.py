#!/usr/bin/env python3
"""Read existing D5 trajectories and training experts only; exploratory, non-preregistered."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py

from fruit_memory_diag_core import (ECM, OUT, D5, G1, LABEL, TASKS, checked_layout, hidden_metrics,
    hidden_summary, retained_metrics, count_failure_metrics, paired_ci, ratio)
from v_a_common import atomic_json, read_json, read_jsonl, sha256, timestamp


def tree_state(root):
    root = Path(root)
    records = {}
    for p in root.rglob('*'):
        if p.is_symlink():
            records[str(p.relative_to(root))] = {'link': str(p.readlink())}
        elif p.is_file():
            s = p.stat();records[str(p.relative_to(root))] = [s.st_size, s.st_mtime_ns]
    return records


def begin():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'preservation').mkdir(exist_ok=True)
    statefile = OUT/'status.json'
    if not statefile.exists():
        now = datetime.now(timezone.utc)
        document_time = datetime.fromtimestamp((ECM/'reports/HANDOFF_lowlevel_rescue_20261102.md').stat().st_mtime, timezone.utc)
        start = document_time if timedelta(0) <= now-document_time < timedelta(hours=12) else now
        atomic_json(statefile, {'phase': 6, 'revision': 6, 'label': LABEL, 'state': 'preparing_diagnosis',
            'started_utc': start.isoformat(), 'deadline_utc': (start+timedelta(days=1)).isoformat()})
    f = OUT/'preservation/d5_tree_before.json'
    if not f.exists():
        atomic_json(f, tree_state(D5))
        seal = read_json(ECM/'reports/fruit_lowlevel_v2_seal_20261104.json')
        keep = {str(ECM/'reports/fruit_lowlevel_v2_protocol_20261103.md'): seal['protocol_file_sha256'],
            str(ECM/'reports/fruit_lowlevel_v2_seal_20261104.json'): sha256(ECM/'reports/fruit_lowlevel_v2_seal_20261104.json')}
        for path, digest in seal['files_sha256'].items():
            # Revision6 may inspect existing D5 logs, but never reopen val source layouts.
            if '/data/fruit_picking/rm75b_eg2/val/' not in path:
                assert sha256(path) == digest, path
                keep[path] = digest
        atomic_json(OUT/'preservation/sealed_files_sha256.json', keep)
    verify_preservation()
    return read_json(statefile)


def verify_preservation():
    for path, digest in read_json(OUT/'preservation/sealed_files_sha256.json').items():
        assert sha256(path) == digest, path
    assert tree_state(D5) == read_json(OUT/'preservation/d5_tree_before.json'), 'A sealed D5 artifact changed'


def bound_rows(path, digest):
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, str(path)
    return [json.loads(x) for x in data.splitlines() if x.strip()]


def summarize_retention(rows):
    phases = ('before_target_1', 'before_target_2', 'before_target_3', 'after_all_targets')
    return {'all': ratio(sum(r['retention']['all']['numerator'] for r in rows), sum(r['retention']['all']['denominator'] for r in rows)),
        'frame_zero_selected': ratio(sum(r['retention']['frame_zero_selected']['numerator'] for r in rows), sum(r['retention']['frame_zero_selected']['denominator'] for r in rows)),
        'by_phase': {p: ratio(sum(r['retention']['by_phase'][p]['numerator'] for r in rows),
                             sum(r['retention']['by_phase'][p]['denominator'] for r in rows)) for p in phases}}


def behavior_by_fruit(rows):
    result = {}
    for key, predicate in (('hidden_ripe', lambda f: f['hidden'] and f['ripe']),
                           ('hidden_unripe', lambda f: f['hidden'] and not f['ripe']),
                           ('visible_unripe', lambda f: not f['hidden'] and not f['ripe'])):
        fruits = [f for r in rows for f in r['hidden']['fruits'] if predicate(f)]
        result[key] = {'fruits': len(fruits),
            'approached': ratio(sum(f['approached_15cm'] for f in fruits), len(fruits)),
            'approached_then_detached': ratio(sum(f['approached_then_detached'] for f in fruits), len(fruits)),
            'approached_without_detachment': ratio(sum(f['approached_15cm'] and not f['detached'] for f in fruits), len(fruits)),
            'wrong_fruit': ratio(sum(f['wrong_fruit'] for f in fruits), len(fruits)),
            'actual_leaf_push': ratio(sum(f.get('actual_leaf_push_started', False) for f in fruits), len(fruits))}
    return result


def expert_analysis():
    cases = []
    for kind, root, ids in (('clean', ECM/'data/fruit_picking_v2/rm75b_eg2/train/occluded_ripe', range(100)),
                            ('DART', ECM/'data/fruit_picking_dart_v2/rm75b_eg2/train/occluded_ripe', range(100, 200))):
        for episode in ids:
            manifest = read_json(root/f'ep{episode}_manifest.json')
            original = read_jsonl(root/manifest['gt'])
            with h5py.File(root/manifest['h5'], 'r') as f:
                value = f[f'episode_{episode}/setup/layout_json'][()]
                layout = json.loads(value)
            raw = [{'frame_id': r['timestep'], 'control_step': r['timestep'], 'gt': {
                'tcp_position': r['tcp_pos'], 'fruit_positions': [f['pos'] for f in r['fruits']],
                'detached': [f['attached'] for f in r['fruits']], 'in_basket': [f['in_basket'] for f in r['fruits']],
                'ripe': [f['ripe'] for f in layout['fruits']], 'targets': layout['targets'],
                'n_pick': layout['n_pick'], 'n_in_basket': r['n_in_basket']}} for r in original]
            h = hidden_metrics(layout, raw)
            hidden_ripe = [f for f in h['fruits'] if f['hidden'] and f['ripe']]
            direct = []
            for fruit in hidden_ripe:
                j = fruit['fruit'];first = fruit['first_approach_step']
                direct.append({'fruit': j,
                    'explicit_targeted_reach': any(r['phase'] == 'reach' and r['target'] == j for r in original),
                    'first_15cm_approach_intended_this_fruit': first is not None and original[first]['target'] == j,
                    'first_approach_phase': original[first]['phase'] if first is not None else None})
            planned_unripe = {r['target'] for r in original if r['phase'] == 'reach' and r['target'] is not None
                              and not layout['fruits'][r['target']]['ripe']}
            cases.append({'kind': kind, 'episode': episode, 'hidden': h, 'direct_hidden_ripe': direct,
                'explicit_unripe_target_count': len(planned_unripe), 'source': str(root/manifest['gt'])})
    summary = {}
    for kind in ('clean', 'DART'):
        own = [r for r in cases if r['kind'] == kind]
        targets = [d for r in own for d in r['direct_hidden_ripe']]
        summary[kind] = {'episodes': len(own), 'hidden': hidden_summary(own), 'fruit_behavior': behavior_by_fruit(own),
            'explicit_reach_to_hidden_ripe': ratio(sum(d['explicit_targeted_reach'] for d in targets), len(targets)),
            'first_15cm_entry_target_matches': ratio(sum(d['first_15cm_approach_intended_this_fruit'] for d in targets), len(targets)),
            'explicit_unripe_targets': sum(r['explicit_unripe_target_count'] for r in own)}
    return {'label': LABEL, 'summary': summary, 'episodes': cases,
        'limitation': '15cm approach is geometric; even expert paths can pass near an unripe neighbor. Non-detachment cannot exclude transient contact or one-step grasp.'}


def posthoc():
    begin()
    destination = OUT/'posthoc';destination.mkdir(exist_ok=True)
    final = destination/'summary.json'
    if final.exists():
        verify_preservation();return read_json(final)
    frozen = read_json(D5/'summary.json')
    trusted = {(r['stage'], r['condition'], r['task'], r['episode'], r['seed']): r
               for r in frozen['independent_validation']['records']}
    index = read_jsonl(D5/'per_episode.jsonl')
    assert len(index) == 1200
    cases = []
    for i, entry in enumerate(index):
        stage, condition, task, episode, seed = (entry[k] for k in ('stage', 'condition', 'task', 'episode', 'model_seed'))
        d = Path(entry['source']);proof = trusted[stage, condition, task, episode, seed]
        assert sha256(d/'validity.json') == proof['validity_sha256']
        valid = read_json(d/'validity.json');result = read_json(d/'result.json')
        assert sha256(d/'result.json') == proof['result_sha256']
        raw = bound_rows(d/'raw_frames.jsonl', valid['inputs_sha256']['raw_frames.jsonl'])
        layout = checked_layout(result, raw)
        row = {'stage': stage, 'condition': condition, 'task': task, 'episode': episode, 'model_seed': seed,
            'outcome': result['outcome'], 'source': str(d), 'validity_sha256': proof['validity_sha256']}
        if task == 'FruitOccludedRipe':
            decisions = bound_rows(d/'policy_decisions.jsonl', valid['inputs_sha256']['policy_decisions.jsonl'])
            row.update(hidden=hidden_metrics(layout, raw), retention=retained_metrics(raw, decisions))
        else:
            controls = bound_rows(d/'control_actions.jsonl', valid['inputs_sha256']['control_actions.jsonl'])
            row['count_failure'] = count_failure_metrics(raw, controls, result['outcome'])
        cases.append(row)
        if (i+1) % 100 == 0:
            atomic_json(destination/'progress.json', {'episodes': i+1, 'expected': 1200, 'utc': timestamp(), 'label': LABEL})
            print({'posthoc_episodes': i+1, 'expected': 1200}, flush=True)
    cohorts, comparisons = {}, {}
    for stage, seeds in (('main', (7, 8, 9)), ('sensitivity', (7,))):
        cohorts[stage] = {}
        for condition in ('K0', 'K4', 'K8'):
            occluded = [r for r in cases if r['stage'] == stage and r['condition'] == condition and r['task'] == 'FruitOccludedRipe']
            count = [r for r in cases if r['stage'] == stage and r['condition'] == condition and r['task'] == 'FruitCount']
            drops = [d for r in count for d in r['count_failure']['drops']]
            tails = [r['count_failure']['timeout_tail'] for r in count if r['count_failure']['timeout_tail']]
            phases = Counter(d['phase_proxy'] for d in drops)
            cohorts[stage][condition] = {'hidden': hidden_summary(occluded), 'fruit_behavior': behavior_by_fruit(occluded),
                'pre_gust_retained': summarize_retention(occluded),
                'count_failure': {'episodes': len(count), 'failed_episodes': sum(r['outcome'] == 'fail' for r in count),
                    'dropped_fruit_events': len(drops), 'drop_phases': {p: ratio(phases[p], len(drops)) for p in ('grasp', 'pull', 'carry', 'release')},
                    'timeout_episodes': len(tails), 'timeout_tail_labels': dict(Counter(t['label'] for t in tails))}}
        occluded = [r for r in cases if r['stage'] == stage and r['task'] == 'FruitOccludedRipe']
        for a in ('K4', 'K8'):
            for metric in ('success', 'omission_hidden_ripe', 'checked_hidden_unripe', 'wrong_fruit',
                           'wrong_hidden_unripe', 'wrong_visible_unripe', 'wrong_in_basket', 'omission_plus_wrong_fruit'):
                key = f'{stage}/{a}-K0/{metric}'
                comparisons[key] = paired_ci(occluded, a, 'K0', metric, 'FruitOccludedRipe', seeds, key)
    experts = expert_analysis()
    atomic_json(destination/'experts.json', experts)
    summary = {'label': LABEL, 'part': '§14.3 posthoc', 'episodes': len(cases), 'generated_utc': timestamp(),
        'new_val_rollouts': 0, 'val_source_layouts_or_demonstrations_opened': False,
        'layout_labels_from_recorded_seed_verified_against_initial_gt': True,
        'conditions': cohorts, 'layout_cluster_bootstrap_comparisons': comparisons,
        'expert_comparison': experts['summary'], 'D5_confirmatory_conclusions_unchanged': True}
    with (destination/'episodes.jsonl').open('w') as f:
        for row in cases:f.write(json.dumps(row, ensure_ascii=False)+'\n')
    atomic_json(final, summary)
    verify_preservation()
    return summary


if __name__ == '__main__':
    posthoc()
