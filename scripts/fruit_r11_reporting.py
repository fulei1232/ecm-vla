"""Revision 11 review evidence, including failed attempts and descriptive pixels."""
from collections import Counter
from pathlib import Path

from v_a_common import read_json, timestamp
from fruit_r11_common import OUT, TRACKED, REPORT, DOC, proof, preserve


def metric_groups():
    groups = {}
    for f in sorted((OUT/'replays').glob('*/*/*/ep*/case_metrics.json')):
        purpose, task, level, episode = f.parent.relative_to(OUT/'replays').parts
        result = read_json(f.parent/'result.json') if (f.parent/'result.json').exists() else None
        group = ('R0_expert' if purpose=='audit' and level=='history' else
                 'R0_scripts/'+level if purpose=='audit' else
                 'clean_data' if purpose=='train_clean' else 'DART_data' if purpose=='train_dart' else purpose)
        metrics = read_json(f)
        row = {'source': str(f.parent), 'episode': result['episode'] if result else episode,
               'attempt': result.get('attempt',0) if result else None,
               'success': result['success'] if result else None, 'metrics': metrics}
        groups.setdefault(task+'/'+group, []).append(row)
    output = {}
    for key, rows in groups.items():
        scopes = {}
        for scope in ('all_attempts', 'accepted_only'):
            own = rows if scope=='all_attempts' else [r for r in rows if r['success']]
            episodes = {}
            for row in own:
                body = row['metrics']['basket_body']
                ep = episodes.setdefault(str(row['episode']), {'episode': row['episode'], 'total_body_pixels': 0,
                    'visible_frames': 0, 'maximum_single_camera_pixels': 0, 'peak': None})
                ep['total_body_pixels'] += body['total_pixels_both_cameras']
                ep['visible_frames'] += body['visible_frames_either_camera']
                if body['maximum_single_camera_pixels'] > ep['maximum_single_camera_pixels']:
                    ep['maximum_single_camera_pixels'] = body['maximum_single_camera_pixels']
                    peak = body['peak_screenshot']
                    ep['peak'] = {**peak, 'source': row['source'], 'attempt': row['attempt'],
                        'screenshots': {c: str(Path(row['source'])/p) for c,p in peak['files'].items()}}
            top = sorted((r for r in episodes.values() if r['total_body_pixels']),
                         key=lambda r:(-r['total_body_pixels'], str(r['episode'])))[:5]
            distances = [r['metrics']['gripper_shield']['minimum_distance_m'] for r in own
                         if r['metrics']['gripper_shield']['minimum_distance_m'] is not None]
            scopes[scope] = {'attempts': len(own), 'episodes': len(episodes),
                'failed_attempts': sum(r['success'] is False for r in own),
                'observations': sum(r['metrics']['observations'] for r in own),
                'body_visible_episodes': sum(bool(e['visible_frames']) for e in episodes.values()),
                'body_visible_attempts': sum(bool(r['metrics']['basket_body']['visible_frames_either_camera']) for r in own),
                'body_visible_frames_either_camera': sum(e['visible_frames'] for e in episodes.values()),
                'maximum_single_camera_pixels': max((e['maximum_single_camera_pixels'] for e in episodes.values()), default=0),
                'top5_episodes_by_cumulative_pixels_both_cameras': top,
                'gripper_entered_frames': sum(r['metrics']['gripper_shield']['entered_frames'] for r in own),
                'finger_entered_frames': sum(r['metrics']['gripper_shield']['finger_entered_frames'] for r in own),
                'tcp_entered_frames': sum(r['metrics']['gripper_shield']['tcp_entered_frames'] for r in own),
                'minimum_gripper_distance_m': min(distances) if distances else None,
                'attempt_metrics': own}
        output[key] = scopes
    return {'instruction_revision': 11, 'groups': output,
        'body_visible_frames_definition': 'One control observation if either camera has basket entity pixels; initial and terminal included',
        'maximum_pixels_definition': 'Maximum basket pixels in a single camera frame',
        'top5_definition': 'Episodes ranked by cumulative basket pixels across both cameras and all attempts within scope; show peak frame from both cameras',
        'shield_pixels_excluded': True, 'basket_body_and_gripper_report_only': True,
        'no_shield_distance_is_null': True}


def report(state, support):
    rows = support.collect()
    cohorts = {}
    for level in support.COHORTS:
        own = [r for r in rows if r['level']==level]
        checked = sum(r['guard_checked_steps'] for r in own)
        clipped = sum(r['guard_clipped_steps'] for r in own)
        cohorts[level] = {'episodes': len(own), 'successes': sum(r['success'] for r in own),
            'passed': None if level in support.REPORT_ONLY else support.gate(level,own),
            'report_only': level in support.REPORT_ONLY, 'guard_checked_steps': checked,
            'guard_clipped_steps': clipped, 'guard_clip_rate': clipped/checked if checked else None,
            'failure_modes': dict(Counter(r['outcome'] for r in own if not r['success'])),
            'early_start_episodes': sum(bool(r['early_start_steps']) for r in own),
            'budget_exhausted_exact_N': sum(r['budget_exhausted_exact_N'] for r in own)}
    names = ('revision11_tests.json','approved_reuse.json','ep112_reviewed_failure.json',
        'platform_manifest.json','clock_fit.json','information_audit.json','other_checks.json',
        'R0_gate.json','data_attempt_ledger.json','data_phase_timing.json','data_delivery.json')
    checks = {n: read_json(OUT/n) if (OUT/n).exists() else None for n in names}
    progress = {}
    for purpose in ('train_clean','train_dart'):
        for f in (OUT/'replays'/purpose).glob('*/*/ep*/validity.json'):
            r = read_json(f.parent/'result.json')
            group = progress.setdefault(r['task']+'/'+purpose, {'completed_attempts': 0, 'successful_episodes': 0, 'failure_modes': {}})
            group['completed_attempts'] += 1
            group['successful_episodes'] += bool(r['success'])
            if not r['success']:
                group['failure_modes'][r['outcome']] = group['failure_modes'].get(r['outcome'],0)+1
    descriptive = metric_groups()
    proof('visibility_gripper_audit.json', descriptive)
    preserve()
    summary = {'instruction_revision': 11, 'created_utc': timestamp(), 'state': state,
        'count_R0_replays': len(rows), 'count_R0_target': 350, 'cohorts': cohorts,
        'checks': checks, 'data_progress': progress,
        'visibility_gripper_audit': str(TRACKED/'visibility_gripper_audit.json'),
        'R10_R0_historical_unshielded_only': True, 'old_83_demonstrations_not_mixed': True,
        'old_ep112_failure_preserved_excluded_not_repeated': True,
        'occluded_R0_accepted_not_rerun': True, 'all_previous_artifacts_unchanged': True,
        'training': False, 'val150_199_used': False,
        'cases': [{k:r[k] for k in ('task','level','episode','N','success','outcome','steps','source')} for r in rows]}
    proof('summary.json', summary)
    info, delivery = checks['information_audit.json'], checks['data_delivery.json']
    lines = [f"修订11：{state['state']}；{'已停止' if state['state'].startswith('R11_') or state['state']=='awaiting_data_review' else '准备或执行中，未宣称通过'}。",
        f"罩内计数R0完成{len(rows)}/350；信息门槛：{info['passed'] if info else '待执行'}；遮挡R0已定稿，不重跑。",
        f"新数据成功{sum(g['successful_episodes'] for g in progress.values())}/400；构建与共用归一化：{'完成' if delivery else '未完成'}；不训练。",
        'ep112保留为正常DART robot_drop失败，不进入训练，不再要求单条诊断成功。',
        '修订10的R0只作无遮挡板历史记录；旧83条数据不混用；D5/D6及所有历史结果保持，不使用val、不改论文。','',
        '## 固定规则与执行顺序','', DOC.read_text(),'',
        '## 计数主队列','', '| 条件 | 成功/完成 | 通过 | 裁剪/检查步 |','|---|---:|---|---:|']
    for level, c in cohorts.items():
        lines.append(f"| {level} | {c['successes']}/{c['episodes']} | {'只报告' if c['report_only'] else c['passed']} | {c['guard_clipped_steps']}/{c['guard_checked_steps']} |")
    lines += ['', '## 信息可得性','']
    if info:
        lines += ['| 条件/N段 | 全采摘覆盖 | 比例 |','|---|---:|---:|']
        for condition, bands in info['conditions'].items():
            for band, r in bands.items():
                lines.append(f"| {condition}/{band} | {r['all_broad']}/{r['episodes']} | {r['broad_fraction']} |")
    else:
        lines.append('待350条全部完成后按冻结定义计算。它是专家轨迹的信息上限，不代表模型一定会使用。')
    lines += ['', '## 篮子本体与夹爪（均只报告）','',
        '| 任务/组别（全部尝试） | 尝试 | 本体可见回合/帧 | 单帧最大像素 | 夹爪进入帧 | 最小距离(m) |',
        '|---|---:|---:|---:|---:|---:|']
    for key, scopes in descriptive['groups'].items():
        v = scopes['all_attempts']
        lines.append(f"| {key} | {v['attempts']} | {v['body_visible_episodes']}/{v['body_visible_frames_either_camera']} | {v['maximum_single_camera_pixels']} | {v['gripper_entered_frames']} | {v['minimum_gripper_distance_m']} |")
        for ep in v['top5_episodes_by_cumulative_pixels_both_cameras']:
            peak = ep['peak']
            lines.append(f"ep{ep['episode']}累计{ep['total_body_pixels']}像素，峰值步{peak['step']}："+
                         '、'.join(f'[{c}]({p})' for c,p in peak['screenshots'].items())+'。')
    lines += ['', f"逐尝试、按任务和clean/DART分组、全部尝试及成功子集：[visibility_gripper_audit.json]({TRACKED}/visibility_gripper_audit.json)。",
        '篮内果和备用果仍为零像素硬门槛；夹爪用原URDF可见手指盒和TCP相对14个罩子渲染盒的欧氏距离，无遮罩距离为null。','',
        '## 数据耗时','']
    if delivery:
        cost = delivery['data_generation_timing']
        phases = cost['data_stage_wall_times']
        lines += [f"生成并行总墙钟{phases['raw_generation_wall_seconds']:.3f}秒；含构建、归一化和交付共{phases['total_wall_seconds']:.3f}秒。",'',
            '| 任务/N/范围 | 回合数 | 平均秒 | 标准差 | 范围 |','|---|---:|---:|---:|---|']
        for task, groups in cost['by_task_and_N'].items():
            for scope, bands in groups.items():
                for n, row in bands.items():
                    d = row['process_wall_seconds']
                    lines.append(f"| {task}/{n}/{scope} | {d['n']} | {d['mean']} | {d['std']} | {d['range']} |")
        lines.append('逐尝试耗时之和不等于8卡并行墙钟；生成耗时含规划和HDF5写入，后续VLA评测还需要推理。')
    else:
        lines.append(f"数据尚未完成；已有阶段耗时：{checks['data_phase_timing.json']}。")
    lines += ['', '## 决策与下一步','', state.get('reason','从时钟校准开始，门槛全过才生成400条数据，生成完停下。'),
        f'机器汇总：[summary.json]({TRACKED}/summary.json)。', '']
    content = '\n'.join(lines)
    REPORT.write_text(content)
    (OUT/'REPORT.md').write_text(content)
    return summary
