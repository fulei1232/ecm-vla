#!/usr/bin/env python3
"""Verify the complete exploratory D6 delivery without changing diagnostic definitions."""
from collections import Counter
import fcntl
from pathlib import Path

import numpy as np

from audit_fruit_memory_diag import audit
from fruit_memory_diag_core import OUT, G1, LABEL, MODES, ABLATIONS, TASKS, checked_layout
from posthoc_fruit_memory_usage import verify_preservation
from run_fruit_memory_usage_diag import verify_sources, bitwise_compare
from v_a_common import ECM, atomic_json, read_json, read_jsonl, sha256, timestamp


def verify():
    state = read_json(OUT/'status.json')
    summary = read_json(OUT/'summary.json')
    plan = read_json(OUT/'plan.json')
    assert state['state'] == 'awaiting_D6_review' and summary['complete']
    verify_sources(plan)
    assert sha256(G1/'guard_v2.json') == plan['guard_sha256']
    params = read_json(ECM/'results/fruit_dart_kf_eval_20261027/protocol.json')['params']['oracle']
    pilot = read_json(OUT/'pilot_comparison.json')
    assert len(pilot['comparisons']) == 5
    for item in pilot['comparisons']:
        assert bitwise_compare(Path(item['old']), Path(item['new'])) == item
    assert pilot['reuse_G1_seed7'] == all(p['bitwise_equal'] for p in pilot['comparisons'])
    expected = {(c, t, e, seed) for c in MODES for t in TASKS for e in range(220, 260) for seed in (7, 8)}
    records, seen, sources = [], set(), set()
    successes, episodes, omissions, commissions, scores = (Counter() for _ in range(5))
    retention, decision_counts = Counter(), Counter()
    plan_hash = sha256(OUT/'plan.json')
    metadata_files = list((OUT/'audits').glob('K0_*/*/*/summary.json'))
    assert len(metadata_files) == 480
    for metadata in sorted(metadata_files):
        evidence = read_json(metadata)
        c, task, e, seed = (evidence[k] for k in ('condition', 'task', 'episode', 'model_seed'))
        key = c, task, e, seed
        assert key in expected and key not in seen
        seen.add(key)
        assert evidence['passed'] and evidence['plan_sha256'] == plan_hash
        source = Path(evidence['source'])
        if evidence['reused_G1']:
            assert pilot['reuse_G1_seed7'] and c == 'K0_normal' and seed == 7
            assert source == G1/f'G1/{task}/ep{e}'
        else:
            assert source.is_relative_to(OUT) and source.resolve().is_relative_to(OUT.resolve())
        assert source.resolve() not in sources
        sources.add(source.resolve())
        canonical = OUT/'episodes'/c/task/f'ep{e}_seed{seed}'
        assert canonical.is_symlink() and canonical.resolve() == source.resolve()
        for name, digest in evidence['source_sha256'].items():
            assert sha256(source/name) == digest, str(source/name)
        result = read_json(source/'result.json')
        assert (result['condition'], result['task'], result['episode_id'], result['model_seed']) == ('K0', task, e, seed)
        assert not result.get('error') and result['outcome'] in ('success', 'fail', 'timeout')
        assert (result['joint_guard'], result['exec_horizon'], result['observe_hold_steps'], result['total_step_budget']) == ('v2', 16, 30, 1300)
        proof = audit(source, ABLATIONS[c], params)
        assert proof['passed'] and read_json(metadata.parent/'memory_audit.json')['passed']
        assert proof['decisions'] == evidence['decisions']
        episodes[c, task] += 1
        successes[c, task] += result['outcome'] == 'success'
        record = {'condition': c, 'task': task, 'episode': e, 'model_seed': seed,
            'source': str(source), 'outcome': result['outcome'], 'reused_G1': evidence['reused_G1'],
            'result_sha256': sha256(source/'result.json'), 'audit_sha256': sha256(metadata.parent/'memory_audit.json'),
            'audited_decisions': proof['decisions']}
        if task == 'FruitOccludedRipe':
            raw = read_jsonl(source/'raw_frames.jsonl')
            layout = checked_layout(result, raw)
            hidden = {o['fruit'] for o in layout['occluders']}
            omitted, wrong = False, False
            for j, fruit in enumerate(layout['fruits']):
                detached_at = next((i for i, r in enumerate(raw) if r['gt']['detached'][j]), None)
                if j in hidden and fruit['ripe']:
                    stop = len(raw)-1 if detached_at is None else detached_at
                    approached = any(np.linalg.norm(np.asarray(r['gt']['tcp_position'])-r['gt']['fruit_positions'][j]) < .15 for r in raw[:stop+1])
                    omitted |= not approached
                if not fruit['ripe']:
                    wrong |= detached_at is not None or any(r['gt']['in_basket'][j] for r in raw)
            omissions[c] += omitted
            commissions[c] += wrong
            scores[c] += int(omitted)+int(wrong)
            decisions = read_jsonl(source/'policy_decisions.jsonl')
            for d in decisions:
                retention[c] += any(raw[k]['control_step'] < 8 for k in d['selected_frame_ids'])
            decision_counts[c] += len(decisions)
            record.update(independent_omission=bool(omitted), independent_commission=bool(wrong))
        records.append(record)
    assert seen == expected and len(records) == summary['inference_episodes'] == 480
    for c in MODES:
        reported = summary['conditions'][c]
        for task in TASKS:
            assert episodes[c, task] == reported['by_task'][task]['episodes'] == 80
            assert successes[c, task] == reported['by_task'][task]['success']['numerator']
        assert omissions[c] == reported['hidden']['omission_hidden_ripe']['numerator']
        assert commissions[c] == reported['hidden']['wrong_fruit']['numerator']
        assert scores[c] == reported['hidden']['omission_plus_wrong_fruit']['score_sum']
        assert retention[c] == reported['pre_gust_retained']['all']['numerator']
        assert decision_counts[c] == reported['pre_gust_retained']['all']['denominator']
    loss = successes['K0_normal', 'FruitOccludedRipe']-successes['K0_no_pregust', 'FruitOccludedRipe']
    rise = scores['K0_no_pregust']-scores['K0_normal']
    verdict = 'A' if loss >= 8 or rise >= 8 else 'B' if loss < 4 and rise < 4 else 'C'
    rule = summary['diagnosis_rule_14_5']
    assert (verdict, loss/80*100, rise/80*100) == (rule['verdict'], rule['success_drop_pp'], rule['omission_plus_commission_rise_pp'])
    assert sum(r['audited_decisions'] for r in records) == summary['audited_decisions']
    verify_preservation()
    return {'passed': True, 'label': LABEL, 'generated_utc': timestamp(), 'records': records,
        'episodes': 480, 'paired_groups': 160, 'audited_decisions': summary['audited_decisions'],
        'independent_verdict': verdict, 'success_loss_cases': loss, 'omission_plus_commission_rise': rise,
        'D5_preservation_passed': True, 'source_sha256': sha256(Path(__file__)), 'plan_sha256': plan_hash}


def main():
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proof = verify()
        tracked = ECM/'reports/lowlevel_rescue_D6_diag_20261104'
        for directory in (OUT, tracked):
            atomic_json(directory/'independent_validation.json', proof)
        summary = read_json(OUT/'summary.json')
        summary['independent_validation'] = {'passed': True, 'path': str(OUT/'independent_validation.json'),
            'sha256': sha256(OUT/'independent_validation.json'), 'verifier_sha256': proof['source_sha256'],
            'episodes': proof['episodes'], 'paired_groups': proof['paired_groups'], 'audited_decisions': proof['audited_decisions'],
            'independent_verdict': proof['independent_verdict'], 'D5_preservation_passed': True}
        for directory in (OUT, tracked):
            atomic_json(directory/'summary.json', summary)
        root = read_json(ECM/'reports/lowlevel_rescue_status.json')
        root['links']['D6_validation'] = str(tracked/'independent_validation.json')
        atomic_json(ECM/'reports/lowlevel_rescue_status.json', root)
        print({k: v for k, v in proof.items() if k != 'records'}, flush=True)


if __name__ == '__main__':
    main()
