"""Sealed D5 primary, failure-mechanism and sensitivity analyses; no policy runs."""
from collections import Counter
from pathlib import Path
import csv
import json

import numpy as np

from analyze_fruit_failure_modes import aggregate
from fruit_lowlevel_v2_eval_common import OUT, TASKS, CONDITIONS, SEAL, verify_seal
from fruit_lowlevel_v2_metrics_r5 import analyze_directory, class_counts, fraction
from fruit_lowlevel_v2_statistics import contrast, ratio_contrast, holm
from v_a_common import ECM, atomic_json, read_json, read_jsonl, sha256, timestamp


def ratio(v):
    n, d = v['numerator'], v['denominator']
    return f'{n}/{d} ({n/d:.2%})' if d else '0/0（无观测/不适用）'


def interval(r, scale=100):
    return '无定义' if r['ci95'] is None else f'[{r["ci95"][0]*scale:.2f}, {r["ci95"][1]*scale:.2f}]'


def effect(r, scale=100):
    return '无定义' if r['difference'] is None else f'{r["difference"]*scale:.2f}'


def vector(rows, condition, metric, seed_list, fn=None):
    selected = [r for r in rows if r['condition'] == condition]
    lookup = {(r['episode'], r['model_seed']): r for r in selected}
    layouts = sorted({r['episode'] for r in selected})
    return np.array([[fn(lookup[e, s]) if fn else
                      (lookup[e, s]['metrics']['outcome'] == metric if metric in ('success', 'fail', 'timeout')
                       else lookup[e, s]['metrics'][metric])
                      for s in seed_list] for e in layouts], dtype=float)


def collect(stage, seal):
    seeds = (7, 8, 9) if stage == 'main' else (7,)
    rows = []
    for condition in CONDITIONS:
        for task in TASKS:
            for episode in range(100, 150):
                for seed in seeds:
                    d = OUT/stage/condition/task/f'ep{episode}_seed{seed}'
                    valid = read_json(d/'validity.json')
                    assert valid['passed'] and valid['seal_sha256'] == sha256(SEAL)
                    for name, digest in valid['inputs_sha256'].items():
                        assert sha256(d/name) == digest, str(d/name)
                    result, metrics, cls = analyze_directory(d, seal['q7_center'])
                    assert not metrics['engineering_error'] and cls['class'] != 'invalid'
                    assert cls == read_json(d/'classification.json')
                    events = read_json(d/'event_statistics.json')
                    assert events['event_validity_passed']
                    controls = read_jsonl(d/'control_actions.jsonl')
                    rows.append({'stage': stage, 'task': task, 'episode': episode, 'model_seed': seed,
                        'condition': condition, 'n_pick': result['n_pick'], 'outcome': result['outcome'],
                        'server_session': result['server_session'], 'source': str(d), 'metrics': metrics,
                        'classification': cls, 'guard_projection_steps': sum(bool(c.get('guard_projection')) for c in controls),
                        'event_recall_proxy': fraction(events['detach_event_keyframes'], events['physical_detach_events']),
                        'contact_events': events['contact_events'],
                        'event_slot_fraction': events['designated_event_slot_fraction']})
    for task in TASKS:
        for episode in range(100, 150):
            for seed in seeds:
                group = [r for r in rows if (r['task'], r['episode'], r['model_seed']) == (task, episode, seed)]
                assert len(group) == 3 and len({r['server_session'] for r in group}) == 1
                assert len({r['n_pick'] for r in group}) == 1
    return rows


def summaries(rows):
    result = {}
    for condition in CONDITIONS:
        own = [r for r in rows if r['condition'] == condition]
        scopes = [('all', own)] + [(t, [r for r in own if r['task'] == t]) for t in TASKS]
        s = {}
        for task, selected in scopes:
            n, d = (sum(r['event_recall_proxy'][k] for r in selected) for k in ('numerator', 'denominator'))
            s[task] = {'metrics': aggregate([r['metrics'] for r in selected]),
                'failure_classes': class_counts([r['classification'] for r in selected]),
                'guard_projection_steps': sum(r['guard_projection_steps'] for r in selected),
                'event_recall_proxy': fraction(n, d), 'contact_events': sum(r['contact_events'] for r in selected),
                'event_slot_fraction': fraction(sum(r['event_slot_fraction']['numerator'] for r in selected),
                                                sum(r['event_slot_fraction']['denominator'] for r in selected)),
                'success_with_memory_error': sum(r['outcome'] == 'success' and r['classification']['class'] == 'memory' for r in selected)}
        s['by_quota_N'] = {str(n): aggregate([r['metrics'] for r in own if r['n_pick'] == n]) for n in sorted({r['n_pick'] for r in own})}
        s['by_task_and_quota_N'] = {t: {str(n): aggregate([r['metrics'] for r in own if r['task'] == t and r['n_pick'] == n])
                                       for n in sorted({r['n_pick'] for r in own if r['task'] == t})} for t in TASKS}
        result[condition] = s
    return result


def report(seal=None):
    seal = verify_seal() if seal is None else seal
    assert verify_seal() == seal
    main, sensitivity = collect('main', seal), collect('sensitivity', seal)
    assert len(main) == 900 and len(sensitivity) == 300
    groups = [read_json(p) for p in (OUT/'groups').glob('*.json')]
    assert len(groups) == 400 and all(g['complete'] and g['seal_sha256'] == sha256(SEAL) for g in groups)
    primary, diagnostic, secondary = {}, {}, {}
    for stage, data, seeds in (('main', main, (7, 8, 9)), ('sensitivity', sensitivity, (7,))):
        for task in TASKS:
            selected = [r for r in data if r['task'] == task]
            pairs = [('K4', 'K0'), ('K8', 'K0')] + ([('K8', 'K4')] if stage == 'sensitivity' else [])
            for a, b in pairs:
                name = f'{stage}/{task}/{a}-{b}'
                answer = contrast(vector(selected, a, 'success', seeds), vector(selected, b, 'success', seeds), name, primary=stage == 'main')
                (primary if stage == 'main' else diagnostic)[name] = answer
            for a, b in (('K4', 'K0'), ('K8', 'K0'), ('K8', 'K4')):
                for category in ('memory', 'operation'):
                    name = f'{stage}/{task}/{a}-{b}/{category}'
                    indicator = lambda r, category=category: r['classification']['class'] == category
                    aa, bb = (vector(selected, c, None, seeds, indicator) for c in (a, b))
                    secondary[name] = contrast(aa, bb, name)
                    failed = lambda r: r['classification']['class'] in ('memory', 'operation')
                    da, db = (vector(selected, c, None, seeds, failed) for c in (a, b))
                    secondary[name+'/within_failure'] = ratio_contrast(aa, da, bb, db, name+'/within_failure')
            if stage == 'main':
                for a, b in (('K4', 'K0'), ('K8', 'K0')):
                    for metric in ('correct_in_basket', 'wrong_in_basket', 'count_exact', 'fail', 'freeze_200_basket_unsuccessful'):
                        if metric == 'count_exact' and task != 'FruitCount':
                            continue
                        name = f'main/{task}/{a}-{b}/{metric}'
                        if metric == 'freeze_200_basket_unsuccessful':
                            eligible = lambda r: bool(r['metrics']['basket_unsuccessful'])
                            dena, denb = (vector(selected, c, None, seeds, eligible) for c in (a, b))
                            numa, numb = (vector(selected, c, None, seeds, lambda r: eligible(r) and r['metrics']['freeze_200']) for c in (a, b))
                            secondary[name] = ratio_contrast(numa, dena, numb, denb, name)
                        else:
                            secondary[name] = contrast(vector(selected, a, metric, seeds), vector(selected, b, metric, seeds), name)
    assert len(primary) == 4
    adjusted = holm({k: v['p_one_sided'] for k, v in primary.items()})
    for key, answer in primary.items():
        answer.update(p_holm=adjusted[key], supported=answer['difference'] > 0 and adjusted[key] <= .05)
    metrics = {'main': summaries(main), 'sensitivity': summaries(sensitivity)}
    attempts = [r for seed in (7, 8, 9) for r in read_jsonl(OUT/f'attempts_seed{seed}.jsonl')]
    assert len(attempts) >= 1200
    by_attempt = {}
    for r in attempts:
        key = tuple(r[k] for k in ('stage', 'task', 'episode', 'model_seed', 'condition'))
        by_attempt.setdefault(key, []).append(r)
    assert len(by_attempt) == 1200
    for records in by_attempt.values():
        records.sort(key=lambda r: r['attempt'])
        assert [r['attempt'] for r in records] == list(range(len(records))) and len(records) <= 3
        assert all(r['terminal_task_outcome_found'] is None for r in records[:-1])
        assert records[-1]['terminal_task_outcome_found'] in ('success', 'fail', 'timeout')
    retries = [r for r in attempts if r['attempt'] > 0]
    summary = {'phase': 5, 'revision': 5, 'stop': 'B', 'generated_utc': timestamp(),
        'validity_passed': True, 'episodes': {'main': len(main), 'sensitivity': len(sensitivity)},
        'seal_sha256': sha256(SEAL), 'seal': seal, 'primary_holm_family': primary,
        'secondary_descriptive_only': secondary, 'sensitivity_nonconfirmatory': diagnostic,
        'conditions': metrics, 'attempts': attempts, 'retry_attempt_count': len(retries),
        'all_attempts_retained_in_superseded': True, 'requires_review_B': True,
        'paper_modified': False, 'analysis_code_sha256': sha256(Path(__file__))}
    atomic_json(OUT/'summary.json', summary)
    with (OUT/'per_episode.jsonl').open('w') as f:
        for row in main+sensitivity:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (OUT/'per_episode.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('stage', 'task', 'episode', 'model_seed', 'condition', 'n_pick', 'outcome', 'class', 'adjacent_misgrasp_possible', 'source'))
        writer.writeheader()
        for row in main+sensitivity:
            writer.writerow({**{k: row[k] for k in ('stage', 'task', 'episode', 'model_seed', 'condition', 'n_pick', 'outcome', 'source')},
                'class': row['classification']['class'], 'adjacent_misgrasp_possible': 'adjacent_misgrasp_possible' in row['classification']['descriptive_subtags']})
    supported = sum(v['supported'] for v in primary.values())
    lines = [f'阶段5停点B：正式900条与无包络敏感性300条完成，有效性通过。',
        f'主假设族4项单侧检验，Holm校正后{supported}/4获得支持；逐项差异与区间见下表。',
        '三个条件均使用固定11999、同一统计和G1平台；敏感性仅关闭包络。',
        f'全部1200条具备有效性证据；额外重跑{len(retries)}次，全部尝试保留。',
        '记忆类/操作类及邻近误抓子标签完整报告；到停点B等待用户审阅，不写论文。', '',
        '## 主假设（两个任务，共一个4项Holm族）', '',
        '| 任务/对比 | 差异百分点 | 95%布局聚类区间 | 单侧p | Holm p | 支持 |', '|---|---:|---|---:|---:|---|']
    for key, v in primary.items():
        lines.append(f'| {key} | {effect(v)} | {interval(v)} | {v["p_one_sided"]:.6g} | {v["p_holm"]:.6g} | {v["supported"]} |')
    lines += ['', '布局聚类bootstrap10000次；随机化以布局整体交换条件100000次；种子子流和重采样摘要在summary.json。', '',
        '## 记忆类与操作类（机制描述，不进主检验族）', '',
        '| 范围/条件/任务 | 成功类 | 记忆类 | 操作类 | 邻近误抓（回合/果实） |', '|---|---:|---:|---:|---|']
    for stage in metrics:
        for c in CONDITIONS:
            for t in TASKS:
                v = metrics[stage][c][t]['failure_classes'];k = v['classes'];n = v['episodes']
                lines.append(f'| {stage}/{c}/{t} | {k["success"]}/{n} | {k["memory"]}/{n} | {k["operation"]}/{n} | {v["subtags_episode_counts"]["adjacent_misgrasp_possible"]}/{v["adjacent_misgrasp_possible_fruit_events"]} |')
    lines += ['', '邻近误抓列为回合数/果实事件数，不是比例；它不改变未成熟果实抓取的记忆类归属。所有子标签数量见summary.json，逐例见per_episode.jsonl。', '',
        '| 记忆/操作类对比 | 差异百分点 | 95%区间 |', '|---|---:|---|']
    for key, v in secondary.items():
        if key.endswith('/memory') or key.endswith('/operation') or key.endswith('/within_failure'):
            lines.append(f'| {key} | {effect(v)} | {interval(v)} |')
    lines += ['', 'within_failure为该类/全部分类失败，条件各自的分母在每次布局重采样中重算；未定义的bootstrap次数单列，未填0。', '',
        '## 不开包络敏感性（非确证性）', '', '| 对比 | 差异百分点 | 95%布局聚类区间 |', '|---|---:|---|']
    for key, v in diagnostic.items():
        lines.append(f'| {key} | {effect(v)} | {interval(v)} |')
    lines += ['', '种子7、val100–149，三个条件各100条；这6项任务内对比不进入Holm。与开启包络主分析分别列出，不据此改变主假设判定。', '',
        '## 其他预定义次要指标的配对差', '', '| 对比 | 差异（果实数或百分点） | 95%区间 |', '|---|---:|---|']
    for key, v in secondary.items():
        if not any(part in key for part in ('/memory', '/operation', '/within_failure')):
            scale = 1 if key.endswith('/correct_in_basket') or key.endswith('/wrong_in_basket') else 100
            lines.append(f'| {key} | {effect(v, scale)} | {interval(v, scale)} |')
    for stage in metrics:
        lines += ['', f'## {stage}：全部统一指标及诊断', '']
        for scope in ('all', *TASKS):
            vals = {c: metrics[stage][c][scope] for c in CONDITIONS}
            lines += ['', f'### {scope}', '', '| 指标 | K0 | K4 | K8 |', '|---|---:|---:|---:|']
            for metric in ('success', 'fail', 'timeout', 'engineering_error', 'first_target_in_basket', 'never_detached', 'near_miss',
                           'wrist_over_limit', 'wrist_deviation_executed', 'wrist_deviation_proposed', 'freeze_200', 'freeze_200_basket_unsuccessful', 'count_exact'):
                lines.append('| '+metric+' | '+' | '.join(ratio(v['metrics'][metric]) for v in vals.values())+' |')
            for metric in ('correct_in_basket', 'wrong_in_basket'):
                lines.append('| '+metric+'（果实总数/回合数） | '+' | '.join(f'{v["metrics"][metric]["total_fruits"]}/{v["metrics"][metric]["episodes_with_data"]}' for v in vals.values())+' |')
            for metric in ('start_idle', 'leave_home_step', 'post_release_resume_steps'):
                lines.append('| '+metric+'中位步 | '+' | '.join(str(v['metrics'][metric]['median']) for v in vals.values())+' |')
            lines.append('| 从未离家 | '+' | '.join(ratio(v['metrics']['leave_home_step']['never_left_home']) for v in vals.values())+' |')
            lines.append('| 入篮后未恢复（右删失） | '+' | '.join(ratio(v['metrics']['post_release_resume_steps']['unresumed_at_end']) for v in vals.values())+' |')
            lines.append('| 任一关节裁剪 | '+' | '.join(ratio(v['metrics']['guard_clip_rate']['any_joint']) for v in vals.values())+' |')
            for j in range(7):
                lines.append(f'| 关节{j+1}裁剪 | '+' | '.join(ratio(v['metrics']['guard_clip_rate']['per_joint'][j]) for v in vals.values())+' |')
            lines.append('| guard_projection步数 | '+' | '.join(str(v['guard_projection_steps']) for v in vals.values())+' |')
            lines.append('| 代理脱离/真值脱果（事件召回诊断） | '+' | '.join(ratio(v['event_recall_proxy']) for v in vals.values())+' |')
            lines.append('| result成功但归记忆类的冲突回合 | '+' | '.join(str(v['success_with_memory_error']) for v in vals.values())+' |')
            for context in ('carrying_fruit', 'closed_empty', 'open_after_basket', 'open_before_first_detach'):
                lines.append('| 偏离起始 '+context+' | '+' | '.join(ratio(v['metrics']['deviation_onset_context'][context]) for v in vals.values())+' |')
    lines += ['', '事件召回为事件总数比，不做一对一匹配或截断；它是方法诊断。离家和恢复中位数只用已观察样本；裁剪分母含30步保持。', '',
        '## 配额N分层', '', '| 范围/条件/任务/N | 成功 | 首颗入篮 | 掉果 |', '|---|---:|---:|---:|']
    for stage in metrics:
        for c in CONDITIONS:
            for task, quotas in metrics[stage][c]['by_task_and_quota_N'].items():
                for n, v in quotas.items():
                    lines.append(f'| {stage}/{c}/{task}/N={n} | {ratio(v["success"])} | {ratio(v["first_target_in_basket"])} | {ratio(v["fail"])} |')
    lines += ['', '分层不根据结果合并，所有分层统一指标见summary.json。', '', '## 有效性、全部尝试与来源', '',
        f'封存清单：`{SEAL}`；摘要 `{sha256(SEAL)}`。协议、分析代码、平台、模型和布局清单摘要通过核对。400个布局×种子组各有3条件，组内同服务会话；全部逐决策记忆审计通过。',
        f'全部尝试{len(attempts)}条、额外重跑{len(retries)}次；明细见summary.json/attempts及attempts_seed7/8/9.jsonl。所有尝试（包括采纳的最后一次）在superseded，main/sensitivity为采纳尝试的只读链接。',
        f'逐回合指标、分类和来源：`{OUT}/per_episode.jsonl`及CSV；每个尝试保留控制、原始帧、事件与审计日志。旧正式评测只作背景，未用于统计比较。', '',
        '## 决策与下一步', '', '按§13.4在停点B交付D5，等待用户审阅。没有修改平台、模型、阈值或论文。']
    body = '\n'.join(lines)+'\n'
    (OUT/'REPORT.md').write_text(body)
    tracked = ECM/'reports/lowlevel_rescue_D5_B_20261104';tracked.mkdir(exist_ok=True)
    atomic_json(tracked/'summary.json', summary)
    (ECM/'reports/lowlevel_rescue_D5_B_20261104.md').write_text(body)
    root = read_json(ECM/'reports/lowlevel_rescue_status.json')
    root['links'].update(D5_B_report=str(OUT/'REPORT.md'), D5_B_summary=str(OUT/'summary.json'))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json', root)


def failure_report(seal, state):
    partial = {}
    for stage in ('main', 'sensitivity'):
        partial[stage] = {}
        for c in CONDITIONS:
            rows = []
            for f in (OUT/stage/c).glob('*/ep*_seed*/result.json'):
                try:
                    _, metrics, _ = analyze_directory(f.parent, seal['q7_center'])
                    rows.append(metrics)
                except Exception:
                    pass
            partial[stage][c] = aggregate(rows)
    attempts = [r for seed in (7, 8, 9) for r in read_jsonl(OUT/f'attempts_seed{seed}.jsonl')]
    evidence = []
    for attempt in attempts:
        directory = Path(attempt.get('directory', ''))
        details = {name: read_json(directory/name) for name in ('classification.json', 'memory_audit.json', 'event_statistics.json')
                   if (directory/name).exists()}
        if details:
            evidence.append({'attempt': attempt, 'evidence': details})
    summary = {'phase': 5, 'revision': 5, 'validity_passed': False, 'state': state,
        'partial_engineering_diagnostics_only': partial, 'attempts': attempts,
        'attempt_diagnostic_evidence': evidence,
        'seal_sha256': sha256(SEAL), 'no_confirmatory_inference': True, 'automatic_remediation': False}
    atomic_json(OUT/'summary.json', summary)
    lines = ['阶段5正式评测有效性失败，已停止新增回合。',
        f'原因：{state.get("error")}', '全部尝试和错误保留，未重跑任何已有success/fail/timeout结局的回合。',
        '未计算确证性主假设结论；现有指标仅供工程诊断。', '按§13.4停止汇报，等待用户决定，不自行补救。', '',
        f'状态：`{OUT}/status.json`；已有回合与全部尝试明细：`{OUT}/summary.json`，superseded/和groups/。']
    body = '\n'.join(lines)+'\n';(OUT/'REPORT.md').write_text(body)
    tracked = ECM/'reports/lowlevel_rescue_D5_B_20261104';tracked.mkdir(exist_ok=True)
    atomic_json(tracked/'summary.json', summary)
    (ECM/'reports/lowlevel_rescue_D5_B_20261104.md').write_text(body)
