#!/usr/bin/env python3
"""Read-only independent checks and review evidence for revision4 stop A.

Never launches policies, training, protocol sealing or val evaluation.
Runtime sources and the draft's preregistered definitions are preserved.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import re
import subprocess

from analyze_fruit_failure_modes import analyze_episode, aggregate
from audit_fruit_rescue_memory import audit
from fruit_lowlevel_v2_metrics import event_statistics, aggregate_event_statistics
from run_fruit_lowlevel_v2_stage5 import OUT, RUN, K0, G1, DATA, DEV, DRAFT, verify_inputs
from v_a_common import ECM, atomic_json, read_json, read_jsonl, sha256, timestamp


def ratio(v):
    n, d = v['numerator'], v['denominator']
    return f'{n}/{d} ({n/d:.2%})' if d else '0/0（不适用/无观测）'


def main():
    plan = read_json(OUT/'plan.json')
    verify_inputs(plan)
    state = read_json(OUT/'status.json')
    assert state['state'] == 'awaiting_protocol_review_A' and not state.get('error')
    current = read_json(OUT/'summary.json')
    assert current['complete_to_stop_A'] and not current['evaluation_protocol_sealed']
    assert not current['formal_evaluation_started'] and not current['val_layouts_accessed']
    initial = OUT/'summary_before_independent_review.json'
    if not initial.exists():
        assert current['draft_sha256_at_stop_A'] == sha256(DRAFT)
        atomic_json(initial, current)
        (OUT/'report_before_independent_review.md').write_bytes((OUT/'report.md').read_bytes())
        (OUT/'draft_before_independent_review.md').write_bytes(DRAFT.read_bytes())
    base = read_json(initial)
    begin, end = '<!-- STOP_A_RESULTS_BEGIN -->', '<!-- STOP_A_RESULTS_END -->'
    prepared = subprocess.check_output(['git', 'show', '324134f:reports/fruit_lowlevel_v2_protocol_20261103.md'], cwd=ECM, text=True)
    assert prepared.split(begin)[0] == DRAFT.read_text().split(begin)[0], 'Draft definitions changed after preparation'
    assert prepared.split(end)[1] == DRAFT.read_text().split(end)[1]
    training = {}
    norm = sha256(DATA/'assets/robomme/norm_stats.json')
    original = read_json(K0/'arms/k0.json')
    for arm, run in (('K0', K0), ('K4', RUN), ('K8', RUN)):
        spec = read_json(run/f'arms/{arm.lower()}.json')
        restored = deepcopy(spec)
        restored.update(arm=original['arm'], checkpoint_base_dir=original['checkpoint_base_dir'])
        restored['sampler']['events'] = 0
        assert restored == original
        checkpoint = Path(base['checkpoints'][arm])
        steps = sorted(p.name for p in checkpoint.parent.iterdir() if p.is_dir() and p.name.isdigit())
        assert steps == ['11999']
        assert sha256(checkpoint/'assets/robomme/norm_stats.json') == norm
        for name, digest in base['checkpoint_sha256'][arm].items():
            assert sha256(checkpoint/name) == digest
        t1 = base['T1'][arm]
        assert t1['passed']
        row = t1['arms'][arm.lower()]
        loss = [float(x) for x in re.findall(r'Step \d+: .*?\bloss=([^,\s]+)', Path(row['log']).read_text(errors='replace'))]
        assert len(loss) == 240 and all(math.isfinite(x) for x in loss)
        assert loss[0] == row['first_loss'] and loss[-1] == row['last_loss']
        training[arm] = {'verified': True, 'loss_records': len(loss), 'final_checkpoint_only': steps,
            'norm_stats_sha256': norm, 'log_sha256': sha256(row['log']), 'T1_utc': t1['time_utc']}
    k8_second = datetime.strptime(Path(base['T1']['K8']['arms']['k8']['log']).stem.split('_', 1)[1], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    training_status = read_json(RUN/'train_status.json')
    assert training_status['arms'] == {'k8': 'complete'}
    k8_start = datetime.fromisoformat(training_status['started_utc'])
    assert k8_second <= k8_start < k8_second+timedelta(seconds=1)
    assert datetime.fromisoformat(base['validity']['K4']['completed_utc']) < k8_start
    oracle = read_json(ECM/'results/fruit_dart_kf_eval_20261027/protocol.json')['params']['oracle']
    center = read_json(G1/'expert_profile.json')['q7_center']
    records = []
    zero = []
    total_decisions = 0
    for group, conditions, episodes in (('validity', ('K4', 'K8'), range(200, 220)),
                                        ('smoke', ('K0', 'K4', 'K8'), (200,))):
        for condition in conditions:
            rows, events = [], []
            directories = list((OUT/group/condition).glob('*/ep*/result.json'))
            assert len(directories) == 2*len(episodes)
            for task in ('FruitCount', 'FruitOccludedRipe'):
                for episode in episodes:
                    d = OUT/group/condition/task/f'ep{episode}'
                    result = read_json(d/'result.json')
                    assert (result['condition'], result['task'], result['episode_id'], result['model_seed']) == (condition, task, episode, 7)
                    assert not result['error'] and result['client_exit_code'] == 0
                    assert result['joint_guard'] == 'v2' and result['exec_horizon'] == 16
                    assert result['observe_hold_steps'] == 30 and result['total_step_budget'] == 1300
                    manifest = DEV/('count' if task == 'FruitCount' else 'occluded_ripe')/f'ep{episode}_manifest.json'
                    assert result['environment_seed'] == read_json(manifest)['seed']
                    started = read_json(d/'started.json')
                    assert started['manifest_sha256'] == sha256(manifest)
                    assert started['plan_sha256'] == sha256(OUT/'plan.json')
                    command = started['command']
                    context = Path(command[command.index('--context')+1])
                    cfg_file = context.parent/'config.json'
                    cfg = read_json(cfg_file)
                    assert sha256(cfg_file) == started['server_config_sha256']
                    assert cfg['checkpoints'][condition] == base['checkpoints'][condition]
                    assert cfg['model_seed'] == 7 and cfg['oracle'] == oracle
                    evidence = audit(d, oracle)
                    assert evidence['passed'] and evidence == read_json(d/'memory_audit.json')
                    total_decisions += evidence['decisions']
                    event = event_statistics(d, oracle['keyframes'][condition])
                    assert event['event_validity_passed'] and event == read_json(d/'event_statistics.json')
                    events.append(event)
                    raw = read_jsonl(d/'raw_frames.jsonl')
                    controls = read_jsonl(d/'control_actions.jsonl')
                    assert len(raw) == len(controls)+1 and len(controls) == result['executed_steps'] <= 1300
                    assert all(r['frame_id'] == i and r['policy_control_step'] == i-30 for i, r in enumerate(raw))
                    row = analyze_episode(d, center)
                    assert not row['engineering_error']
                    row.update(task=task, episode_id=episode)
                    rows.append(row)
                    if condition != 'K0' and not event['designated_event_slot_fraction']['numerator']:
                        minimum = min(r['gt']['fruit_distance'] for r in raw)
                        assert not event['contact_events'] and minimum > .03
                        zero.append({'group': group, 'condition': condition, 'task': task, 'episode': episode,
                            'minimum_distance_m_all_physical_frames': minimum,
                            'minimum_distance_m_before_last_decision': event['min_distance_before_last_decision'],
                            'physical_detach_events': event['physical_detach_events'],
                            'explanation': '全程未进入距离≤3cm的事件近区；实际脱果不等同于距离代理触发。',
                            'source': str(d/'raw_frames.jsonl')})
                    records.append({'group': group, 'condition': condition, 'task': task, 'episode': episode,
                        'server_session': result['server_session'], 'steps': len(controls),
                        'audited_decisions': evidence['decisions'], 'guard_projection_steps': sum(bool(c.get('guard_projection')) for c in controls),
                        'inputs_sha256': {name: sha256(d/name) for name in ('result.json', 'started.json', 'raw_frames.jsonl',
                            'control_actions.jsonl', 'event_signal.jsonl', 'policy_decisions.jsonl', 'oracle_decisions.jsonl',
                            'memory_audit.json', 'event_statistics.json')}})
            expected = base[group][condition]
            assert aggregate(rows) == expected['performance']['all']
            for task in ('FruitCount', 'FruitOccludedRipe'):
                assert aggregate([r for r in rows if r['task'] == task]) == expected['performance']['by_task'][task]
            assert aggregate_event_statistics(events) == expected['event_statistics']
    assert len(records) == 86
    for task in ('FruitCount', 'FruitOccludedRipe'):
        assert len({r['server_session'] for r in records if r['group'] == 'smoke' and r['task'] == task}) == 1
    review = {'verified_utc': timestamp(), 'source_script_sha256': sha256(Path(__file__)),
        'training': training, 'sequential_training_verified': True, 'unchanged_dataset_and_K0_files': 411,
        'unchanged_runtime_sources': len(plan['source_sha256']), 'original_draft_definitions_unchanged': True,
        'original_runtime_draft_sha256': base['draft_sha256_at_stop_A'],
        'all_validity_and_smoke_metrics_recomputed': True, 'all_memory_and_event_audits_recomputed': True,
        'episodes_verified': len(records), 'decisions_verified': total_decisions,
        'zero_event_episodes_with_specific_explanations': zero, 'records': records,
        'evaluation_sealed': False, 'formal_eval_started': False, 'no_val_results_or_layouts_read_by_this_validator': True}
    atomic_json(OUT/'independent_validation.json', review)
    lines = (OUT/'report_before_independent_review.md').read_text().splitlines()
    lines[1] = 'K0直接复用11999；K4/K8已依次从79999完成12000步，数据、统计、种子42及8卡FSDP均与K0相同。'
    lines += ['', '## 独立复核与事件汇总', '',
        f'来源：`{OUT}/independent_validation.json`；逐项核对训练配方、来源摘要、检查点、全部86条原始轨迹，重新计算{total_decisions}次记忆/事件决策审计及全部统一指标，与自动产物一致。',
        'K4有效性完成时间早于K8训练启动；每模型仅有11999检查点。411份数据/K0保全文件和32份运行来源摘要保持一致。协议在启动前写下的全部定义与准备提交324134f相同，仅补入实测证据。', '',
        '| 条件 | contact | detach代理 | 实际脱果 | 事件槽/有效槽 | 零事件回合 |', '|---|---:|---:|---:|---:|---:|']
    for condition in ('K4', 'K8'):
        e = base['validity'][condition]['event_statistics']['all']
        lines.append(f'| {condition} | {e["contact_events"]} | {e["detach_event_keyframes"]} | {e["physical_detach_events"]} | {ratio(e["designated_event_slot_fraction"])} | {len(e["zero_event_slot_episodes"])}/40 |')
    lines += ['', '零事件回合的具体解释（全物理帧最小距离，不只看决策前缀）：', '',
        '| 条件/任务/布局 | 最小距离cm | 实际脱果数 | 原因 |', '|---|---:|---:|---|']
    for r in zero:
        lines.append(f'| {r["condition"]}/{r["task"]}/{r["episode"]} | {r["minimum_distance_m_all_physical_frames"]*100:.6f} | {r["physical_detach_events"]} | {r["explanation"]} |')
    lines += ['', '这些回合没有已观察到事件却漏选事件槽的证据；不改变3cm阈值或选帧设置。脱果代理与实际脱果可能不同，二者均保留。', '',
        '## 全部§5及§9.5指标', '', '每个条件均列合计与分任务；这里的性能只作记录，未用来调整任何设置。']
    for scope in ('all', 'FruitCount', 'FruitOccludedRipe'):
        get = lambda c: base['validity'][c]['performance']['all'] if scope == 'all' else base['validity'][c]['performance']['by_task'][scope]
        versions = {c: get(c) for c in ('K4', 'K8')}
        lines += ['', f'### {scope}', '', '| 指标 | K4 | K8 |', '|---|---:|---:|']
        for metric in ('success', 'fail', 'timeout', 'engineering_error', 'first_target_in_basket', 'never_detached',
                       'near_miss', 'wrist_over_limit', 'wrist_deviation_executed', 'wrist_deviation_proposed', 'freeze_200',
                       'freeze_200_basket_unsuccessful', 'count_exact'):
            lines.append('| '+metric+' | '+' | '.join(ratio(v[metric]) for v in versions.values())+' |')
        for metric in ('correct_in_basket', 'wrong_in_basket'):
            lines.append('| '+metric+'（果实总数/有数据回合） | '+' | '.join(f'{v[metric]["total_fruits"]}/{v[metric]["episodes_with_data"]}' for v in versions.values())+' |')
        for metric in ('start_idle', 'leave_home_step', 'post_release_resume_steps'):
            lines.append('| '+metric+'中位步 | '+' | '.join(str(v[metric]['median']) for v in versions.values())+' |')
        lines.append('| 从未离家 | '+' | '.join(ratio(v['leave_home_step']['never_left_home']) for v in versions.values())+' |')
        lines.append('| 入篮后未恢复（右删失） | '+' | '.join(ratio(v['post_release_resume_steps']['unresumed_at_end']) for v in versions.values())+' |')
        lines.append('| 任一关节裁剪/全部控制步 | '+' | '.join(ratio(v['guard_clip_rate']['any_joint']) for v in versions.values())+' |')
        for j in range(7):
            lines.append(f'| 关节{j+1}裁剪/全部控制步 | '+' | '.join(ratio(v['guard_clip_rate']['per_joint'][j]) for v in versions.values())+' |')
        for context in ('carrying_fruit', 'closed_empty', 'open_after_basket', 'open_before_first_detach'):
            lines.append('| 偏离起始'+context+' | '+' | '.join(ratio(v['deviation_onset_context'][context]) for v in versions.values())+' |')
    lines += ['', '离家与恢复中位数仅计算观察到的事件，未观察值单列，不填0。count_exact只适用于FruitCount；裁剪分母包括30步保持，未去掉回合中段帧。', '',
        '## 停点A的审阅事项', '',
        '请按HANDOFF §12.5审阅两模型T1、有效性与冒烟，以及协议草案。草案预定义四项单侧Holm检验、10000次布局聚类bootstrap、互斥失败分类和种子7无包络敏感性分析；计数提前停止明确为最后200步的可观察代理，限制已列明。',
        '当前停在A，等待审阅后再封存和正式评测；没有读取val布局、执行正式/敏感性回合或写论文。']
    body = '\n'.join(lines)+'\n'
    (OUT/'report.md').write_text(body)
    (ECM/'reports/lowlevel_rescue_D5_A_20261103.md').write_text(body)
    draft = DRAFT.read_text()
    DRAFT.write_text(draft[:draft.index(begin)+len(begin)]+'\n\n'+body+'\n'+draft[draft.index(end):])
    summary = deepcopy(base)
    summary['independent_validation'] = review
    summary['draft_sha256_at_stop_A'] = sha256(DRAFT)
    summary['independent_review_utc'] = review['verified_utc']
    atomic_json(OUT/'summary.json', summary)
    atomic_json(ECM/'reports/lowlevel_rescue_D5_A_20261103/summary.json', summary)
    root = read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase=5, instruction_revision=4, state='awaiting_protocol_review_A',
        updated_utc=timestamp(), decision='Stop A reached and independently verified; await user review before sealing or val evaluation.',
        next='User review of T1, dev validity, dev200 smoke and unsealed protocol draft.')
    root['links'].update(D5_A_validation=str(OUT/'independent_validation.json'))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json', root)
    print({'verified': True, 'stop': 'A', 'episodes': len(records), 'audited_decisions': total_decisions,
           'zero_event_explanations': len(zero), 'evaluation_sealed': False}, flush=True)


if __name__ == '__main__':
    main()
