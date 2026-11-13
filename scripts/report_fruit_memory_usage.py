"""Exploratory D6 delivery only; all D5 artifacts remain read-only."""
from collections import Counter
from pathlib import Path
import json

from analyze_fruit_failure_modes import analyze_episode, aggregate
from fruit_memory_diag_core import (OUT, G1, LABEL, MODES, TASKS, checked_layout, hidden_metrics,
    hidden_summary, retained_metrics, count_failure_metrics, paired_ci, diagnose_rule, ratio)
from posthoc_fruit_memory_usage import verify_preservation, summarize_retention, behavior_by_fruit
from v_a_common import ECM, atomic_json, read_json, read_jsonl, sha256, timestamp


def fmt(v):
    n, d = v['numerator'], v['denominator']
    return f'{n}/{d} ({n/d:.2%})' if d else '0/0（无可用分母）'


def ci(v):
    return f'{v["difference"]*100:.2f} [{v["ci95"][0]*100:.2f}, {v["ci95"][1]*100:.2f}]'


def collect():
    rows = []
    center = read_json(G1/'expert_profile.json')['q7_center']
    for condition in MODES:
        for task in TASKS:
            for e in range(220, 260):
                for seed in (7, 8):
                    meta = OUT/'audits'/condition/task/f'ep{e}_seed{seed}/summary.json'
                    if not meta.exists():continue
                    evidence = read_json(meta)
                    assert evidence['passed']
                    d = Path(evidence['source'])
                    for name, digest in evidence['source_sha256'].items():assert sha256(d/name) == digest
                    result = read_json(d/'result.json');raw = read_jsonl(d/'raw_frames.jsonl')
                    controls = read_jsonl(d/'control_actions.jsonl');layout = checked_layout(result, raw)
                    m = analyze_episode(d, center)
                    assert not m['engineering_error']
                    row = {'condition': condition, 'task': task, 'episode': e, 'model_seed': seed,
                        'outcome': result['outcome'], 'source': str(d), 'reused_G1': evidence['reused_G1'],
                        'metrics': m, 'audited_decisions': evidence['decisions']}
                    if task == 'FruitOccludedRipe':
                        row.update(hidden=hidden_metrics(layout, raw), retention=retained_metrics(raw, read_jsonl(d/'policy_decisions.jsonl')))
                    else:row['count_failure'] = count_failure_metrics(raw, controls, result['outcome'])
                    rows.append(row)
    return rows


def deliver(state, complete=False):
    post = read_json(OUT/'posthoc/summary.json')
    rows = collect()
    if complete:assert len(rows) == 480
    pilot = read_json(OUT/'pilot_comparison.json') if (OUT/'pilot_comparison.json').exists() else None
    comparisons = {};conditions = {}
    for condition in MODES:
        own = [r for r in rows if r['condition'] == condition]
        oc = [r for r in own if r['task'] == 'FruitOccludedRipe']
        count = [r for r in own if r['task'] == 'FruitCount']
        drops = [d for r in count for d in r['count_failure']['drops']]
        phases = Counter(d['phase_proxy'] for d in drops)
        conditions[condition] = {'all': aggregate([r['metrics'] for r in own]),
            'by_task': {t: aggregate([r['metrics'] for r in own if r['task'] == t]) for t in TASKS},
            'hidden': hidden_summary(oc), 'fruit_behavior': behavior_by_fruit(oc), 'pre_gust_retained': summarize_retention(oc),
            'count_failure': {'failed_episodes': sum(r['outcome'] == 'fail' for r in count), 'drop_events': len(drops),
                'phases': {p: ratio(phases[p], len(drops)) for p in ('grasp', 'pull', 'carry', 'release')},
                'timeout_tails': dict(Counter(r['count_failure']['timeout_tail']['label'] for r in count if r['count_failure']['timeout_tail']))}}
    if complete:
        for condition in MODES[1:]:
            for task in TASKS:
                fields = ['success']+(['omission_hidden_ripe', 'checked_hidden_unripe', 'wrong_fruit', 'wrong_in_basket',
                    'wrong_hidden_unripe', 'wrong_visible_unripe', 'omission_plus_wrong_fruit', 'omission_or_wrong_fruit'] if task == 'FruitOccludedRipe' else [])
                for metric in fields:
                    name = f'{task}/{condition}-K0_normal/{metric}'
                    comparisons[name] = paired_ci(rows, condition, 'K0_normal', metric, task, (7, 8), name)
        verdict = diagnose_rule(rows)
    else:verdict = {'verdict': None, 'reason': state.get('error', 'Inference diagnosis pending/incomplete'), 'label': LABEL}
    verify_preservation()
    summary = {'phase': 6, 'revision': 6, 'label': LABEL, 'complete': complete, 'generated_utc': timestamp(),
        'state': {**state, 'state': 'awaiting_D6_review' if complete else state['state']}, 'posthoc': post, 'pilot_bitwise_check': pilot, 'inference_episodes': len(rows),
        'conditions': conditions, 'paired_layout_bootstrap': comparisons, 'diagnosis_rule_14_5': verdict,
        'memory_audits_passed': len(rows), 'audited_decisions': sum(r['audited_decisions'] for r in rows),
        'preservation': {'D5_artifacts_unchanged': True, 'sealed_files_sha256': sha256(OUT/'preservation/sealed_files_sha256.json'),
            'D5_tree_manifest_sha256': sha256(OUT/'preservation/d5_tree_before.json')},
        'D5_confirmatory_conclusions_unchanged': True, 'paper_modified': False}
    atomic_json(OUT/'summary.json', summary)
    with (OUT/'inference_episodes.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r, ensure_ascii=False)+'\n')
    conclusion = f'判定{verdict["verdict"]}（按§14.5点估计规则）' if complete else '尚未判定A/B/C：推理消融未完成'
    lines = [f'{LABEL}；{conclusion}。',
        f'第一部分D5既有1200条及200条遮挡训练示范已分析；第二部分完成{len(rows)}/480条。',
        ('遮挡成功率下降'+f'{verdict["success_drop_pp"]:.2f}个百分点，遗漏+错果上升{verdict["omission_plus_commission_rise_pp"]:.2f}个百分点。' if complete else '第二部分需完成G1五条逐位核对和dev220–259、种子7/8的三种条件。'),
        'D5中K0/K4/K8每次遮挡决策均保留第0帧；没有事件帧挤掉阵风前帧的证据。',
        ('诊断完成后在D6停下等待审阅；D5确证性结论保持原样。' if complete else 'D5封存产物未改；本报告只展示已完成部分，不用不完整数据硬判C。'), '',
        '## 第一部分：遮挡任务的遗漏、查看与错果', '',
        f'来源：`{OUT}/posthoc/summary.json`、episodes.jsonl和experts.json。只用既有D5回放，布局标签从回放种子重建并核对初始GT，没有读取val示范或产生新val回放。', '',
        '| 范围/条件 | 遗漏hidden ripe | 查看hidden unripe | 查看/有hidden unripe回合 | 错果 | hidden错果 | visible错果 |', '|---|---:|---:|---:|---:|---:|---:|']
    for stage, cohorts in post['conditions'].items():
        for c, v in cohorts.items():
            h = v['hidden']
            lines.append(f'| {stage}/{c} | {fmt(h["omission_hidden_ripe"])} | {fmt(h["checked_hidden_unripe"])} | {fmt(h["checked_hidden_unripe_among_eligible"])} | {fmt(h["wrong_fruit"])} | {fmt(h["wrong_hidden_unripe"])} | {fmt(h["wrong_visible_unripe"])} |')
    lines += ['', '| 范围/模型/果实组 | 果实数 | 接近15cm | 接近后脱果 | 接近未脱果 | 实际叶片抬起 |', '|---|---:|---:|---:|---:|---:|']
    for stage, cohorts in post['conditions'].items():
        for c, v in cohorts.items():
            for group, b in v['fruit_behavior'].items():
                lines.append(f'| {stage}/{c}/{group} | {b["fruits"]} | {fmt(b["approached"])} | {fmt(b["approached_then_detached"])} | {fmt(b["approached_without_detachment"])} | {fmt(b["actual_leaf_push"])} |')
    lines += ['', '接近/未脱果是几何代理，不等于主动查看；已有日志不能排除瞬时触碰或单步抓握。PUSH_START作用于移动叶簇中心，另列的果实中心15cm口径是用户指定的主指标。错果含未成熟脱果或错误入篮，另有仅wrong_in_basket和每颗果实明细。', '',
        '| 布局聚类对比（非预登记） | 差异百分点 [95%区间] |', '|---|---|']
    for name, v in post['layout_cluster_bootstrap_comparisons'].items():lines.append(f'| {name} | {ci(v)} |')
    lines += ['', '## 第一部分：阵风前帧保留', '', '| 范围/条件 | 阶段 | pre_gust_retained/决策 | 第0帧/决策（全部） |', '|---|---|---:|---:|']
    for stage, cohorts in post['conditions'].items():
        for c, v in cohorts.items():
            r = v['pre_gust_retained']
            for phase, value in r['by_phase'].items():lines.append(f'| {stage}/{c} | {phase} | {fmt(value)} | {fmt(r["frame_zero_selected"])} |')
    lines += ['', '这里使用物理步<8；策略相对时钟=物理步−30，未把策略步<8误当阵风前。第1/2/3颗之前按已脱离成熟目标数分期，已摘尽单列。均匀取帧包含第0帧；事件关键帧仍有均匀剩余槽，也保留第0帧。保留证据不等于模型确实利用证据。', '',
        '## 第一部分：专家示范对照', '', '| 专家 | 遗漏 | 几何查看/全部 | 几何查看/有hidden unripe | 明确reach hidden ripe | 首次15cm接近时目标吻合 | 明确unripe目标数 |', '|---|---:|---:|---:|---:|---:|---:|']
    for kind, e in post['expert_comparison'].items():
        h = e['hidden']
        lines.append(f'| {kind} | {fmt(h["omission_hidden_ripe"])} | {fmt(h["checked_hidden_unripe"])} | {fmt(h["checked_hidden_unripe_among_eligible"])} | {fmt(e["explicit_reach_to_hidden_ripe"])} | {fmt(e["first_15cm_entry_target_matches"])} | {e["explicit_unripe_targets"]} |')
    lines += ['', '专家会直接规划被遮挡成熟目标，却也经常路过未成熟邻居的15cm范围。因此该几何查看指标不能单独支持“策略逐个查看而不用记忆”的解释。训练与D5布局不同，专家比较仅作描述。', '',
        '## 第一部分：计数掉果阶段与超时末段', '', '| 范围/条件 | 失败回合 | 掉果事件 | 抓取 | 拉拽 | 搬运 | 释放 | 超时末段标签（回合数） |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for stage, cohorts in post['conditions'].items():
        for c, v in cohorts.items():
            x = v['count_failure'];p = x['drop_phases']
            lines.append(f'| {stage}/{c} | {x["failed_episodes"]}/{x["episodes"]} | {x["dropped_fruit_events"]} | {fmt(p["grasp"])} | {fmt(p["pull"])} | {fmt(p["carry"])} | {fmt(p["release"])} | {x["timeout_tail_labels"]} |')
    lines += ['', '阶段依据脱果、首次8cm分离、张开命令和脱果后TCP移动5cm构成观察性代理，不把落地时刻当成失手原因；分母为掉果果实事件，失败回合另列。超时末段200步的原始运动、夹爪和TCP统计在逐回合文件中。', '',
        '## 第二部分：G1核对、消融实现与审计', '', f'五条核对结果：`{OUT}/pilot_comparison.json`；当前记录：{pilot}。',
        '默认不开启任何消融。启用诊断服务后仍需每条context显式指定recent16或no_pregust；normal无标记。封存源码全部不改。no_pregust排除物理步<20，初次只有11帧，沿用官方右padding；不删历史、不改帧号/位置编码。',
        f'每条审计的源文件摘要和逐决策检查：`{OUT}/audits`。已审计{len(rows)}/480条，共{summary["audited_decisions"]}次决策；复用G1也在新目录重做审计，未向G1写入文件。', '',
        '| 条件/任务 | 成功 | 首颗入篮 | 工程错误 | 执行腕偏离 |', '|---|---:|---:|---:|---:|']
    for condition in MODES:
        for task in TASKS:
            m = conditions[condition]['by_task'][task]
            lines.append(f'| {condition}/{task} | {fmt(m["success"])} | {fmt(m["first_target_in_basket"])} | {fmt(m["engineering_error"])} | {fmt(m["wrist_deviation_executed"])} |')
    lines += ['', '| 条件 | 遗漏hidden ripe | 几何查看hidden unripe | 错果 | 仅错误入篮 | hidden错果 | visible错果 | 遗漏+错果总分/回合 |', '|---|---:|---:|---:|---:|---:|---:|---|']
    for condition in MODES:
        h = conditions[condition]['hidden'];score = h['omission_plus_wrong_fruit']
        lines.append(f'| {condition} | {fmt(h["omission_hidden_ripe"])} | {fmt(h["checked_hidden_unripe"])} | {fmt(h["wrong_fruit"])} | {fmt(h["wrong_in_basket"])} | {fmt(h["wrong_hidden_unripe"])} | {fmt(h["wrong_visible_unripe"])} | {score["score_sum"]}/{score["episodes"]} |')
    lines += ['', '| 相对normal（探索性） | 差异百分点 [95%布局聚类区间] |', '|---|---|']
    for name, v in comparisons.items():lines.append(f'| {name} | {ci(v)} |')
    lines += ['', '| 推理条件 | 记忆中阵风前帧/决策 | 第0帧/决策 |', '|---|---:|---:|']
    for condition in MODES:
        r = conditions[condition]['pre_gust_retained']
        lines.append(f'| {condition} | {fmt(r["all"])} | {fmt(r["frame_zero_selected"])} |')
    lines += ['', '| 推理条件/计数 | 失败回合 | 抓取事件 | 拉拽事件 | 搬运事件 | 释放事件 | 超时末段 |', '|---|---:|---:|---:|---:|---:|---|']
    for condition in MODES:
        r = conditions[condition]['count_failure'];p = r['phases']
        n = conditions[condition]['by_task']['FruitCount']['episodes']
        lines.append(f'| {condition} | {r["failed_episodes"]}/{n} | {fmt(p["grasp"])} | {fmt(p["pull"])} | {fmt(p["carry"])} | {fmt(p["release"])} | {r["timeout_tails"]} |')
    if complete:
        change = comparisons['FruitCount/K0_recent16-K0_normal/success']
        a, b = (conditions[c]['count_failure']['phases'] for c in ('K0_recent16', 'K0_normal'))
        deltas = {p: a[p]['numerator']-b[p]['numerator'] for p in a}
        lines += ['', f'计数recent16相对normal的成功率差为{ci(change)}个百分点；每80回合的掉果果实事件数差（recent−normal）为{deltas}。结合超时末段与掉果阶段作描述，不把记忆消融后的控制退化误认为计数任务恢复了计数需求。历史运动/阶段信息或帧分布改变都是可能解释，本诊断不定位唯一原因。']
    lines += ['', '## §14.5判定与局限', '', f'判定：{verdict}。',
        '合计错误=遗漏回合指示+commission回合指示，同一回合可计2，另列并集及仅错误入篮。按有符号的成功下降/错误上升和预先指定10pp、5pp门槛判定；CI不参与改判。这是探索性诊断，不是新的确证性记忆效果结论。',
        '计数recent16即使下降，也可能因控制阶段信息或输入帧分布改变；计数环境达N即终止的结构缺陷依然存在，不能据此证明在计数。所有任务重设计都留给用户决定。', '',
        '## 保全与下一步', '', f'保全摘要：{summary["preservation"]}。D5所有文件大小/mtime、链接及封存摘要核对不变；新数据只来自dev220–259种子7/8。',
        f'来源：`{OUT}/posthoc`、`{OUT}/episodes`、`{OUT}/audits`、`{OUT}/summary.json`与执行计划。',
        'D6交付后停止等待用户审阅；全部标注探索性、非预登记，不改变D5确证性结论，不写入论文。']
    body = '\n'.join(lines)+'\n';(OUT/'REPORT.md').write_text(body)
    tracked = ECM/'reports/lowlevel_rescue_D6_diag_20261104';tracked.mkdir(exist_ok=True)
    atomic_json(tracked/'summary.json', summary)
    (ECM/'reports/lowlevel_rescue_D6_diag_20261104.md').write_text(body)
    root = read_json(ECM/'reports/lowlevel_rescue_status.json')
    root['links'].update(D6_report='reports/lowlevel_rescue_D6_diag_20261104.md', D6_summary=str(tracked/'summary.json'))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json', root)
