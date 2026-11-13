#!/usr/bin/env python3
"""Revision5 additions; preserve the revision4 classifier and training provenance.

The base classification still executes fruit_lowlevel_v2_metrics.classify_failure.
Only the authorized structural-error handling and descriptive adjacency label
are added. There are no fitted parameters or performance-based changes.
"""
import argparse
import ast
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_fruit_failure_modes import analyze_episode
from fruit_lowlevel_v2_metrics import classify_failure as classify_v4, event_statistics, fraction
from v_a_common import ECM, atomic_json, read_json, read_jsonl, sha256, timestamp

ENV = ECM/'fruit_picking/env.py'


def detach_steps():
    for node in ast.parse(ENV.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DETACH_STEPS' for t in node.targets):
            return int(ast.literal_eval(node.value))
    raise ValueError('DETACH_STEPS is unavailable')


def classify_failure(result, raw, controls, near_miss=False):
    original = classify_v4(result, raw, controls, near_miss)
    structural = [tag for tag in original['memory_errors'] if tag in (
        'count_over_quota', 'count_continued_picking_after_quota')]
    answer = {**original, 'environment_anomalies': structural, 'descriptive_subtags': [],
              'unripe_grasp_events': [], 'revision': 5}
    if structural:
        answer.update(**{'class': 'invalid'}, engineering_error='count termination invariant violated')
        answer['memory_errors'] = [t for t in original['memory_errors'] if t not in structural]
        return answer
    if original['class'] == 'invalid' or result['task'] != 'FruitOccludedRipe':
        return answer
    steps = detach_steps()
    if steps != 2:
        raise ValueError('Revision5 is bound to the audited DETACH_STEPS=2 environment')
    for i in range(1, len(raw)):
        before, current = raw[i-1]['gt'], raw[i]['gt']
        for fruit, (a, b, ripe) in enumerate(zip(before['detached'], current['detached'], current['ripe'])):
            if a or not b or ripe:
                continue
            start = i-steps+1
            if start < 0 or raw[start]['control_step'] != raw[i]['control_step']-steps+1:
                raise ValueError('Cannot identify the first step of the successful continuous grasp')
            gt = raw[start]['gt']
            position = np.asarray(gt['fruit_positions'][fruit], dtype=float)
            candidates = [j for j in gt['targets'] if gt['ripe'][j] and not gt['detached'][j]]
            nearest = min(((float(np.linalg.norm(position-np.asarray(gt['fruit_positions'][j], dtype=float))), j)
                           for j in candidates), default=None)
            adjacent = nearest is not None and nearest[0] < .04
            answer['unripe_grasp_events'].append({'fruit': fruit,
                'detach_frame': raw[i]['frame_id'], 'grasp_start_frame': raw[start]['frame_id'],
                'grasp_start_control_step': raw[start]['control_step'], 'detach_steps': steps,
                'nearest_attached_target': nearest[1] if nearest else None,
                'distance_to_nearest_attached_target_m': nearest[0] if nearest else None,
                'adjacent_misgrasp_possible': adjacent,
                'timing': 'First control step of the DETACH_STEPS consecutive grasps ending in this first detachment.'})
    if any(e['adjacent_misgrasp_possible'] for e in answer['unripe_grasp_events']):
        answer['descriptive_subtags'].append('adjacent_misgrasp_possible')
    assert answer['class'] == original['class'], 'Adjacency must never change classification'
    return answer


def analyze_directory(directory, center):
    directory = Path(directory)
    result = read_json(directory/'result.json')
    raw = read_jsonl(directory/'raw_frames.jsonl')
    controls = read_jsonl(directory/'control_actions.jsonl')
    metrics = analyze_episode(directory, center)
    classification = classify_failure(result, raw, controls, bool(metrics.get('near_miss')))
    if metrics['engineering_error']:
        classification['class'] = 'invalid'
    return result, metrics, classification


def class_counts(rows):
    counts = Counter(r['class'] for r in rows)
    tags = Counter(t for r in rows for t in r['memory_errors']+r['operation_tags']+r['descriptive_subtags'])
    for tag in ('count_over_quota', 'count_continued_picking_after_quota', 'count_premature_stop_proxy',
                'occluded_wrong_in_basket', 'occluded_unripe_detached', 'adjacent_misgrasp_possible',
                'never_detached', 'near_miss', 'dropped_fruit_outcome_fail', 'timeout_stopped', 'timeout_operating_or_stalled'):
        tags.setdefault(tag, 0)
    structural = Counter(t for r in rows for t in r['environment_anomalies'])
    return {'episodes': len(rows), 'classes': {k: counts[k] for k in ('success', 'memory', 'operation', 'invalid')},
        'subtags_episode_counts': dict(tags),
        'structural_anomalies': {k: structural[k] for k in ('count_over_quota', 'count_continued_picking_after_quota')},
        'unripe_grasp_events': sum(len(r['unripe_grasp_events']) for r in rows),
        'adjacent_misgrasp_possible_fruit_events': sum(e['adjacent_misgrasp_possible'] for r in rows for e in r['unripe_grasp_events'])}


def dev_trial(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    roots = {'K0': ECM/'results/fruit_k0v2_gate_20261102/dev/v2',
             **{c: ECM/f'results/fruit_lowlevel_v2_stage5_20261103/validity/{c}' for c in ('K4', 'K8')}}
    center = read_json(ECM/'results/fruit_k0v2_gate_20261102/expert_profile.json')['q7_center']
    oracle = read_json(ECM/'results/fruit_dart_kf_eval_20261027/protocol.json')['params']['oracle']
    records = []
    for condition, root in roots.items():
        for task in ('FruitCount', 'FruitOccludedRipe'):
            for e in range(200, 220):
                d = root/task/f'ep{e}'
                result, metrics, classification = analyze_directory(d, center)
                assert result['condition'] == condition and result['model_seed'] == 7
                evidence = read_json(d/'memory_audit.json')
                assert evidence['passed']
                events = event_statistics(d, oracle['keyframes'][condition])
                assert events['event_validity_passed']
                records.append({'condition': condition, 'task': task, 'episode': e, 'source': str(d),
                    'outcome': result['outcome'], 'classification': classification,
                    'event_recall_proxy': fraction(events['detach_event_keyframes'], events['physical_detach_events']),
                    'inputs_sha256': {n: sha256(d/n) for n in ('result.json', 'raw_frames.jsonl', 'control_actions.jsonl', 'memory_audit.json')}})
    by_condition = {}
    for c in roots:
        selected = [r for r in records if r['condition'] == c]
        summary = class_counts([r['classification'] for r in selected])
        summary['outcomes'] = dict(Counter(r['outcome'] for r in selected))
        summary['by_task'] = {t: class_counts([r['classification'] for r in selected if r['task'] == t])
                              for t in ('FruitCount', 'FruitOccludedRipe')}
        for task, counts in summary['by_task'].items():
            own = [r for r in selected if r['task'] == task]
            counts['event_recall_proxy'] = fraction(sum(r['event_recall_proxy']['numerator'] for r in own),
                                                   sum(r['event_recall_proxy']['denominator'] for r in own))
        summary['event_recall_proxy'] = fraction(sum(r['event_recall_proxy']['numerator'] for r in selected),
                                               sum(r['event_recall_proxy']['denominator'] for r in selected))
        by_condition[c] = summary
    passed = all(v['classes']['invalid'] == 0 and not any(v['structural_anomalies'].values()) for v in by_condition.values())
    summary = {'revision': 5, 'generated_utc': timestamp(), 'passed': passed, 'episodes': len(records),
        'definitions_changed_from_trial_results': False, 'val_layouts_used': False, 'conditions': by_condition,
        'analysis_source_sha256': {str(p): sha256(p) for p in (Path(__file__).resolve(), ECM/'scripts/fruit_lowlevel_v2_metrics.py', ENV)},
        'grasp_time_inference': 'DETACH_STEPS=2; first step of the successful grasp streak is detach frame minus1. Tree-fruit positions and attached-target set come from that frame.',
        'records': records}
    atomic_json(out/'summary.json', summary)
    lines = ['# 修订5分类代码开发集试运行', '', '只使用既有dev200–219回放，不读val；不基于这些结果改变定义或阈值。', '',
        '| 条件 | 成功类 | 记忆类 | 操作类 | 无效 | 结构异常 | 邻近误抓子标签（回合/果实） | 代理脱离/真值脱果 |', '|---|---:|---:|---:|---:|---:|---|---|']
    for c, v in by_condition.items():
        k = v['classes'];r = v['event_recall_proxy']
        lines.append(f'| {c} | {k["success"]}/40 | {k["memory"]}/40 | {k["operation"]}/40 | {k["invalid"]}/40 | {sum(v["structural_anomalies"].values())} | {v["subtags_episode_counts"]["adjacent_misgrasp_possible"]}/{v["adjacent_misgrasp_possible_fruit_events"]} | {r["numerator"]}/{r["denominator"]} |')
    lines += ['', '邻近误抓列是回合计数/果实事件计数，不是比例；所有分类子标签及逐回合证据在summary.json。', '',
        '## 全部分类子标签（回合计数，不改变分类）', '', '| 子标签 | K0 | K4 | K8 |', '|---|---:|---:|---:|']
    tags = sorted({tag for v in by_condition.values() for tag in v['subtags_episode_counts']})
    for tag in tags:
        lines.append('| '+tag+' | '+' | '.join(f'{by_condition[c]["subtags_episode_counts"].get(tag, 0)}/40' for c in roots)+' |')
    lines += ['', '## 分任务分类与事件召回诊断', '', '| 条件/任务 | 成功 | 记忆 | 操作 | 无效 | 代理脱离/真值脱果 |', '|---|---:|---:|---:|---:|---|']
    for c, v in by_condition.items():
        for task, counts in v['by_task'].items():
            k, r = counts['classes'], counts['event_recall_proxy']
            lines.append(f'| {c}/{task} | {k["success"]}/20 | {k["memory"]}/20 | {k["operation"]}/20 | {k["invalid"]}/20 | {r["numerator"]}/{r["denominator"]} |')
    lines += ['',
        '未成熟果实脱落由连续双指抓握产生；邻近误抓子标签只描述，不改变记忆类归属。计数超额及达额后继续取果为环境终止结构异常，不当作记忆效果。', '',
        f'试运行判定：{passed}。未使用本次结果修改定义、阈值、平台、模型或统计。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print({'passed': passed, 'conditions': {c: v['classes'] for c, v in by_condition.items()}}, flush=True)
    if not passed:
        raise SystemExit('Definition or environment validity issue in dev trial; stop before sealing')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-trial', action='store_true', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    dev_trial(args.out)
