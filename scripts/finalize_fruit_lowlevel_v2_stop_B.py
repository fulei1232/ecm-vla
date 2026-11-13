#!/usr/bin/env python3
"""Independent read-only validation of completed D5; publish stop-B evidence.

No sealed source, protocol, model, raw trajectory or statistical definition is
changed. Primary statistics are recomputed directly from result.json matrices.
"""
from copy import deepcopy
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from analyze_fruit_failure_modes import aggregate
from audit_fruit_rescue_memory import audit
from fruit_lowlevel_v2_eval_common import OUT, SEAL, DOC, TASKS, CONDITIONS, verify_seal
from fruit_lowlevel_v2_metrics_r5 import analyze_directory, event_statistics, class_counts
from v_a_common import ECM, atomic_json, read_json, read_jsonl, sha256, timestamp


def independent_contrast(a, b, name, reported, primary=False):
    delta = (np.asarray(a, dtype=float)-np.asarray(b, dtype=float)).mean(axis=1)
    seed_for = lambda stream: int.from_bytes(hashlib.sha256(f'20261103|{name}|{stream}'.encode()).digest()[:8], 'big')
    rng = np.random.default_rng(seed_for('bootstrap'))
    indices = rng.integers(0, len(delta), size=(10000, len(delta)))
    draws = delta[indices].mean(axis=1)
    assert np.isclose(delta.mean(), reported['difference'], rtol=0, atol=1e-14)
    assert np.allclose(np.quantile(draws, [.025, .975]), reported['ci95'], rtol=0, atol=1e-14)
    assert reported['bootstrap_draws'] == 10000 and reported['bootstrap_seed'] == seed_for('bootstrap')
    match = hashlib.sha256(draws.astype('<f8').tobytes()).hexdigest() == reported['bootstrap_values_sha256']
    evidence = {'difference': float(delta.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist(),
        'bootstrap_values_sha256_matches': match, 'all_numeric_results_match': True}
    if primary:
        integer = (np.asarray(a)-np.asarray(b)).sum(axis=1).astype(np.int64)
        rng = np.random.default_rng(seed_for('randomization'))
        signs = rng.integers(0, 2, size=(100000, len(delta)), dtype=np.int8)*2-1
        greater = int((signs@integer >= integer.sum()).sum())
        p = (greater+1)/100001
        assert reported['permutation_draws'] == 100000
        assert reported['permutation_seed'] == seed_for('randomization')
        assert greater == reported['permutation_exceedances_including_ties']
        assert p == reported['p_one_sided']
        evidence.update(p_one_sided=p, permutation_exceedances=greater)
    return evidence


def main():
    seal = verify_seal()
    state = read_json(OUT/'status.json')
    assert state['state'] == 'awaiting_review_B' and not state.get('error')
    first = OUT/'summary_before_independent_review.json'
    if not first.exists():
        atomic_json(first, read_json(OUT/'summary.json'))
        (OUT/'REPORT_before_independent_review.md').write_bytes((OUT/'REPORT.md').read_bytes())
    summary = read_json(first)
    assert summary['validity_passed'] and summary['episodes'] == {'main': 900, 'sensitivity': 300}
    assert summary['seal'] == seal and summary['seal_sha256'] == sha256(SEAL)
    expected_schedule = read_json(OUT/'schedule.json')
    attempts = [r for seed in (7, 8, 9) for r in read_jsonl(OUT/f'attempts_seed{seed}.jsonl')]
    assert len(attempts) == 1200 and all(r['attempt'] == 0 and r['terminal_task_outcome_found'] in ('success', 'fail', 'timeout') for r in attempts)
    assert attempts == summary['attempts'] and summary['retry_attempt_count'] == 0
    records = []
    total_steps = total_decisions = 0
    allrows = {}
    for stage, seeds in (('main', (7, 8, 9)), ('sensitivity', (7,))):
        rows = []
        for condition in CONDITIONS:
            for task in TASKS:
                for episode in range(100, 150):
                    for seed in seeds:
                        d = OUT/stage/condition/task/f'ep{episode}_seed{seed}'
                        assert d.is_symlink() and d.resolve().parts[-1] == 'attempt0'
                        valid = read_json(d/'validity.json')
                        assert valid['passed'] and valid['seal_sha256'] == sha256(SEAL)
                        for name, digest in valid['inputs_sha256'].items():
                            assert sha256(d/name) == digest, str(d/name)
                        result, metrics, cls = analyze_directory(d, seal['q7_center'])
                        assert not result.get('error') and result['client_exit_code'] == 0
                        assert (result['condition'], result['task'], result['episode_id'], result['model_seed']) == (condition, task, episode, seed)
                        assert result['joint_guard'] == ('v2' if stage == 'main' else 'off')
                        assert result['exec_horizon'] == 16 and result['observe_hold_steps'] == 30 and result['total_step_budget'] == 1300
                        started = read_json(d/'started.json')
                        command = started['command']
                        context = Path(command[command.index('--context')+1])
                        cfg = read_json(context.parent/'config.json')
                        assert sha256(context.parent/'config.json') == started['server_config_sha256']
                        assert cfg['model_seed'] == seed and cfg['checkpoints'] == seal['checkpoints']
                        evidence = audit(d, cfg['oracle'])
                        assert evidence['passed'] and evidence == read_json(d/'memory_audit.json')
                        events = event_statistics(d, cfg['oracle']['keyframes'][condition])
                        assert events == read_json(d/'event_statistics.json') and events['event_validity_passed']
                        assert cls == read_json(d/'classification.json') and cls['class'] != 'invalid'
                        assert not cls['environment_anomalies']
                        lifecycle = [r for r in read_jsonl(d/'policy_lifecycle.jsonl') if r.get('kind') == 'reset']
                        assert len(lifecycle) == 1 and lifecycle[0]['model_seed'] == seed
                        raw, controls = read_jsonl(d/'raw_frames.jsonl'), read_jsonl(d/'control_actions.jsonl')
                        assert len(raw) == len(controls)+1 and len(controls) == result['executed_steps'] <= 1300
                        assert all(r['frame_id'] == i and r['control_step'] == i and r['policy_control_step'] == i-30 for i, r in enumerate(raw))
                        assert all(r['is_demo'] == (i < 30) for i, r in enumerate(raw))
                        total_steps += len(controls);total_decisions += evidence['decisions']
                        rows.append({'condition': condition, 'task': task, 'episode': episode, 'seed': seed,
                            'result': result, 'metrics': metrics, 'class': cls, 'events': events})
                        records.append({'stage': stage, 'condition': condition, 'task': task, 'episode': episode, 'seed': seed,
                            'outcome': result['outcome'], 'steps': len(controls), 'audited_decisions': evidence['decisions'],
                            'validity_sha256': sha256(d/'validity.json'), 'result_sha256': sha256(d/'result.json'),
                            'accepted_attempt': str(d.resolve())})
        allrows[stage] = rows
        for condition in CONDITIONS:
            for task in ('all', *TASKS):
                selected = [r for r in rows if r['condition'] == condition and (task == 'all' or r['task'] == task)]
                target = summary['conditions'][stage][condition][task]
                assert aggregate([r['metrics'] for r in selected]) == target['metrics']
                assert class_counts([r['class'] for r in selected]) == target['failure_classes']
                assert sum(r['events']['detach_event_keyframes'] for r in selected) == target['event_recall_proxy']['numerator']
                assert sum(r['events']['physical_detach_events'] for r in selected) == target['event_recall_proxy']['denominator']
    for planned in expected_schedule:
        stage, task, episode, seed = (planned[k] for k in ('stage', 'task', 'episode', 'seed'))
        group = read_json(OUT/'groups'/f'{stage}_{task}_ep{episode}_seed{seed}.json')
        assert group['group'] == planned and group['complete']
        assert [r['condition'] for r in group['accepted']] == planned['condition_order']
        assert len({r['server_session'] for r in group['accepted']}) == 1
        assert group['seal_sha256'] == sha256(SEAL)
    checks = {}
    for stage, seeds, reported in (('main', (7, 8, 9), summary['primary_holm_family']),
                                   ('sensitivity', (7,), summary['sensitivity_nonconfirmatory'])):
        for name, value in reported.items():
            _, task, pair = name.split('/')
            a, b = pair.split('-')
            lookup = {(r['condition'], r['task'], r['episode'], r['seed']): r['result']['outcome'] == 'success' for r in allrows[stage]}
            mat = lambda c: np.array([[float(lookup[c, task, e, seed]) for seed in seeds] for e in range(100, 150)])
            checks[name] = independent_contrast(mat(a), mat(b), name, value, primary=stage == 'main')
    ordered = sorted(summary['primary_holm_family'], key=lambda k: (checks[k]['p_one_sided'], k))
    adjusted = 0.
    for i, key in enumerate(ordered):
        adjusted = min(1., max(adjusted, checks[key]['p_one_sided']*(4-i)))
        assert summary['primary_holm_family'][key]['p_holm'] == adjusted
        assert summary['primary_holm_family'][key]['supported'] == (checks[key]['difference'] > 0 and adjusted <= .05)
        checks[key]['p_holm'] = adjusted
    review = {'verified_utc': timestamp(), 'source_script_sha256': sha256(Path(__file__)),
        'seal_verified_unchanged': True, 'protocol_file_sha256': sha256(DOC),
        'accepted_episodes': len(records), 'complete_paired_groups': len(expected_schedule),
        'all_attempts': len(attempts), 'retries': 0, 'engineering_errors': 0,
        'memory_event_audits_recomputed': True, 'audited_decisions': total_decisions,
        'total_physical_control_steps': total_steps, 'physical_clock_prefix_and_budget_verified': True,
        'classification_and_full_unified_metrics_match': True,
        'all_count_structural_anomalies_zero': True, 'all_primary_and_sensitivity_statistics_match': checks,
        'records': records}
    assert len(records) == 1200 and len(expected_schedule) == 400
    atomic_json(OUT/'independent_validation.json', review)
    summary['independent_validation'] = review
    summary['interpretation'] = {
        'primary': 'None of four Holm-corrected one-sided tests supports improved success; this is not evidence of equivalence.',
        'memory_failure': 'Count:0/150 in every condition. Occluded:K0 12/150,K4 21/150,K8 20/150; all pairwise descriptive difference CIs include0.',
        'sensitivity': 'Count K4 negative/K8 positive directions persist; occluded K8 point direction reverses. All six no-guard CIs contain0; no envelope-independence claim.'}
    atomic_json(OUT/'summary.json', summary)
    tracked = ECM/'reports/lowlevel_rescue_D5_B_20261104'
    atomic_json(tracked/'summary.json', summary)
    lines = (OUT/'REPORT_before_independent_review.md').read_text().splitlines()
    lines[:5] = ['阶段5停点B完成：正式900条和无包络300条全部有效，已停止等待审阅。',
        '4项主假设均未获支持，Holm校正p均为1；不能据此声称等效或事件记忆无效。',
        'K0/K4/K8成功：计数101/150、93/150、106/150；遮挡45/150、44/150、47/150。',
        '记忆类失败：计数均0/150；遮挡12/150、21/150、20/150，描述性差异区间均包含0。',
        '工程错误0/1200、重跑0，1200条记忆审计通过；协议和模型保持封存，不写入论文。']
    lines += ['', '## 独立复核与结论的边界', '',
        f'独立核验来源：`{OUT}/independent_validation.json`。1200条接受回合、400组配对、{total_decisions}次记忆/事件决策审计和{total_steps}个物理控制步均核对通过；已从原始result.json重建主分析与敏感性矩阵，重新计算所有对应bootstrap区间、置换p及4项Holm，与封存分析产物一致。',
        '原始自动报告和summary另存REPORT_before_independent_review.md、summary_before_independent_review.json；不修改协议、分析函数、模型或任何回放输入。', '',
        '主结论：本协议没有取得事件关键帧提高整任务成功率的统计证据。4项区间均跨0；未拒绝不代表记忆方式等效，也不证明完全没有效应。',
        '机制检查：计数任务的超额和达额后继续取果受立即终止约束；实际可观察的提前停止代理在三条件都为0/150，不能据此检验计数记忆是否改善。遮挡记忆类为K0 12/150、K4 21/150、K8 20/150；K4−K0为+6.00个百分点（95%区间[−1.33,13.33]），K8−K0为+5.33（[−1.33,12.00]）。这些只是预登记的描述结果；未见记忆类失败减少的证据。所有adjacent_misgrasp_possible计数为0，但该局部子标签不能排除其他操作因素。',
        '敏感性：计数K4−K0从−5.33变为−10.00个百分点、K8−K0从+3.33变为+4.00，方向一致；遮挡K4−K0从−0.67到−6.00，K8−K0从+1.33到−6.00，后者点方向改变。无包络六项区间均跨0，不能声称比较结论独立于包络。主分析含3种子、敏感性仅种子7，点估计对照也不能解释成单一包络因果效应。',
        '诊断：无包络时执行/原始输出腕偏离分别为K0 5/100、K4 3/100、K8 2/100；主分析三条件均0/300。它们是预登记的诊断结果，不改变主检验或既定平台取舍。', '',
        '按§13.4在停点B交付D5，等待用户审阅和决定后续用途。没有执行额外实验或写入论文。']
    body = '\n'.join(lines)+'\n'
    (OUT/'REPORT.md').write_text(body)
    (ECM/'reports/lowlevel_rescue_D5_B_20261104.md').write_text(body)
    root = read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase=5, instruction_revision=5, state='awaiting_review_B', updated_utc=timestamp(),
        decision='D5 complete and independently verified; stop B for user review. No primary hypothesis supported.',
        next='Await user review of D5 and decision on subsequent use; no further experiments or paper edits authorized.')
    root['links'].update(D5_B_validation=str(OUT/'independent_validation.json'),
        D5_B_versioned_report='reports/lowlevel_rescue_D5_B_20261104.md',
        D5_B_versioned_summary='reports/lowlevel_rescue_D5_B_20261104/summary.json')
    atomic_json(ECM/'reports/lowlevel_rescue_status.json', root)
    print({'verified': True, 'stop': 'B', 'episodes': len(records), 'paired_groups': len(expected_schedule),
           'audited_decisions': total_decisions, 'physical_steps': total_steps, 'primary_supported': 0}, flush=True)


if __name__ == '__main__':
    main()
