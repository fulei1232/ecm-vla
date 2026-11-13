"""R9 validity, timing, information gates and review reports; no training."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

from v_a_common import ECM, read_json, read_jsonl, sha256, atomic_json, timestamp
from fruit_r9_common import OUT, TRACKED, REPORT, DOC, R8, GUARD, COHORTS, REPORT_ONLY, preserve, proof
sys.path[:0] = [str(ECM), str(ECM / 'src')]
from memory_management.fruit_rescue_platform import guard_v2
from fruit_picking.task_v2_spec import choose_count
from fruit_picking.task_v2_audit_r8 import fit_clock
from fruit_picking.task_v2_audit_r9 import episode_information, information_summary, estimate_onset


def gate(level, rows):
    if len(rows) != 50:return None
    if level in REPORT_ONLY:return True
    success = sum(r['success'] for r in rows)
    passed = success >= 45 if level == 'history' else success <= 15
    if level == 'history':
        checked = sum(r['guard_checked_steps'] for r in rows)
        passed &= checked > 0 and sum(r['guard_clipped_steps'] for r in rows) / checked < .005
    return bool(passed)


def collect(purpose='audit', level=None):
    rows = []
    for f in sorted((OUT / 'replays' / purpose).glob('*/*/ep*/validity.json')):
        validity = read_json(f);assert validity['passed']
        r = read_json(f.parent / 'result.json')
        if level and r['level'] != level:continue
        r['source'] = str(f.parent);rows.append(r)
    return rows


def audit_case(directory):
    r = read_json(directory / 'result.json');trace = read_jsonl(directory / 'trace.jsonl')
    actions = read_json(directory / 'actions.json');decisions = read_json(directory / 'decisions.json')
    assert len(trace) == len(actions) + 1 == r['steps'] + 1
    assert 30 <= r['steps'] <= r['budget']
    assert r['task_revision'] == 9 and r['distance_signal'] == 'any_hanging'
    if r['task']=='count':
        assert r['fruit_actors']==r['M']*(r['N']+1) and r['reserve_actors']==r['M']*r['N']
    assert not any(v[k] for v in r['max_visibility'].values() for k in ('basket', 'basket_fruit', 'inactive_reserve'))
    guard = read_json(GUARD)
    for i, a in enumerate(actions):
        assert a['step'] == i + 1
        expected, details = guard_v2(a['proposed'], a['measured_arm_qpos'], guard)
        assert np.array_equal(expected, a['guarded_clean'])
        actual = expected.copy();actual[:7] += a['noise_offset']
        assert np.array_equal(actual, a['executed'])
        assert details['guard_clipped'] == a['guard_clipped']
        if r['purpose'] != 'train_dart':assert not any(a['noise_offset'])
        if i < 30:assert a['phase'] == 'observe' and a['executed'][7] == 1
        if a['phase'] == 'wait_for_green':assert a['executed'] == actions[0]['executed']
    assert r['guard_checked_steps'] == len(actions) - 30
    assert r['guard_clipped_steps'] == sum(a['guard_clipped'] for a in actions[30:])
    if r['purpose'] == 'early_start_probe':
        assert r['outcome'] == 'early_start' and not r['success'] and r['early_start_steps']
        assert actions[30]['phase'] == 'early_start_probe' and 30 < r['steps'] < r['green_step']
    elif r['task'] == 'count':assert not r['early_start_steps']
    any_distances = [f['signals']['any_hanging'] for f in trace]
    lamps = [f['visible']['lamp_green'] for f in trace]
    fits = read_json(OUT / 'clock_fit.json') if r['level'].startswith('clock') else None
    if fits:assert r['clock_fit_sha256'] == sha256(OUT / 'clock_fit.json')
    for decision in decisions:
        p = decision['provided'];t = decision['step'];view = trace[t]['visible'];level = r['level']
        keys = {'tree','ground','lamp_green','N','visible_basket_count','frame0','release_events','t','green_step'}
        if level == 'clock_onset_k0':keys.add('lamp_bracket')
        assert set(p) == keys
        assert p['ground'] == view['ground'] and p['lamp_green'] == view['lamp_green'] and p['visible_basket_count'] == 0
        if not (r['task'] == 'occluded_ripe' and level == 'history'):assert p['tree'] == view['tree']
        assert p['frame0'] == (None if level == 'current' else trace[0]['visible'])
        assert p['release_events'] == (None if level != 'history' else [e for e in r['own_release_events'] if e['step'] <= t])
        assert p['t'] == (t if level.startswith('clock') else None)
        if level == 'clock_onset_k0':
            onset, bracket = estimate_onset(any_distances, lamps, t)
            assert p['green_step'] == onset and p['lamp_bracket'] == bracket
        else:assert p['green_step'] == (r['green_step'] if level == 'clock_onset' else None)
        if r['task'] == 'count':
            clock = fits['absolute' if level == 'clock' else 'relative'] if fits else None
            actual_level = 'nonstop' if level == 'clock_onset_k0' and p['green_step'] is None else level
            expected = choose_count(actual_level, p['N'], p['tree'], p['frame0']['tree'] if p['frame0'] else None,
                len(p['release_events']) if p['release_events'] is not None else None,
                p['visible_basket_count'], len(p['ground']), p['t'], clock, p['green_step'])
            assert expected == decision['chosen_slot']
            assert all(0 <= f['slot'] < r['M'] and f['id'] == f['slot'] for f in p['tree'])
    assert all(c['passed'] and c['mask_iou'] >= .98 and c['mean_abs_color_diff_255'] <= 2 for c in r['appearance_checks'])
    if r['success'] and r['task'] == 'count':
        assert r['outcome'] == 'finish_signal' and r['steps'] < 2800
        assert r['final_robot_in_basket'] == r['N'] and r['appearance_final_hold_available']
    if r['training_data']:
        import h5py
        assert read_json(OUT / 'R0_gate.json')['passed'] and r['data_frames'] == r['steps'] + 1
        assert sha256(directory / r['h5']) == r['h5_sha256'] and sha256(directory / r['gt']) == r['gt_sha256']
        with h5py.File(directory / r['h5']) as f:
            ep = f[f"episode_{r['episode']}"]
            assert int(ep['setup/n_timesteps'][()]) == r['data_frames']
            assert [bool(ep[f'timestep_{i}/info/is_video_demo'][()]) for i in range(r['data_frames'])] == [True]*30+[False]*(r['data_frames']-30)
    return {'passed': True, 'decisions': len(decisions),
        'source_sha256': {name: sha256(directory / name) for name in ('result.json','trace.jsonl','actions.json','decisions.json')},
        'image_sha256': {name: sha256(directory / name) for item in r['snapshots'].values() for name in item['files'].values()}}


def cost_comparison(row):
    purpose = row['purpose']
    if purpose not in ('budget_expert', 'budget_nonstop'):return None
    level = 'history' if purpose == 'budget_expert' else 'nonstop'
    d = R8 / 'replays' / purpose / 'count' / level / f"ep{row['episode']}"
    before = read_json(d / 'result.json');started = read_json(d / 'started.json')
    seconds = (datetime.fromisoformat(before['created_utc']) - datetime.fromisoformat(started['utc'])).total_seconds()
    return {'episode': row['episode'], 'purpose': purpose, 'R8_N': before['N'], 'R9_N': row['N'],
        'R8_wall_seconds': seconds, 'R9_wall_seconds': row['wall_seconds'], 'ratio': row['wall_seconds'] / seconds,
        'passed': row['wall_seconds'] <= 2 * seconds,
        'R8_fruit_actors': len(before['slot_of']), 'R9_fruit_actors': row['fruit_actors'],
        'R9_scene_entities': row['scene_actors'] + row['scene_articulations'],
        'scope': 'Matched geometry/role, D600; process launch through result, includes initialization and recording; no duration normalization.'}


def budget_check():
    experts = collect('budget_expert');nonstop = collect('budget_nonstop')
    assert all(r['N'] == 10 and r['green_step'] == 630 and r['budget'] == 2800 for r in experts + nonstop)
    finished = [r['steps'] for r in experts if r['success']]
    over = [r['steps'] for r in nonstop if r['outcome'] == 'overpick' and r['final_robot_picked'] == 11]
    latest = max(finished, default=None);last_over = max(over, default=None)
    passed = len(experts) == len(nonstop) == len(finished) == len(over) == 10 and 2800-latest >= 150 and last_over < 2800
    result = {'passed': bool(passed), 'budget': 2800, 'D': 600, 'N': 10,
        'expert_complete': len(finished), 'expert_attempts': len(experts), 'max_finish_step_including_hold': latest,
        'minimum_remaining_steps': 2800-latest if latest is not None else None,
        'nonstop_overpick': len(over), 'nonstop_attempts': len(nonstop), 'max_Nplus1_detach_step': last_over,
        'proposed_budget_if_failed': max(2800, (latest+150) if latest is not None else 2800,
                                       (last_over+1) if last_over is not None else 2800) if not passed else None,
        'cost_comparisons': [cost_comparison(r) for r in experts+nonstop]}
    proof('budget_check.json', result);return result


def fit_calibration():
    rows = collect('calibration');assert len(rows) == 10
    cases = [{'N': r['N'], 'green_step': r['green_step'], 'basket_entry_steps': [e['step'] for e in r['basket_entries']],
              'decision_points': r['decision_points'], 'reference_success': r['success']} for r in rows]
    value = {'source_scope': 'dev250–259, new N2–10 distribution; frozen before audit cohorts',
        'source_sha256': {str(Path(r['source'])/'result.json'): sha256(Path(r['source'])/'result.json') for r in rows},
        'absolute': fit_clock(cases, False), 'relative': fit_clock(cases, True), 'created_utc': timestamp()}
    assert not (OUT/'clock_fit.json').exists(), 'No clock refitting'
    proof('clock_fit.json', value);return value


def information(rows, output='information_audit.json', gate_required=True):
    values = [episode_information(r, read_jsonl(Path(r['source'])/'trace.jsonl'),
                                  read_json(Path(r['source'])/'decisions.json')) for r in rows]
    value = information_summary(values, gate=gate_required);proof(output, value);return value


def descriptive(values):
    return {'n': len(values), 'mean': float(np.mean(values)) if values else None,
            'std': float(np.std(values)) if values else None, 'range': [min(values),max(values)] if values else None}


def report(state):
    rows = collect();cohorts = {}
    for level in COHORTS:
        own = [r for r in rows if r['level'] == level];checks = [c for r in own for c in r['appearance_checks']]
        checked = sum(r['guard_checked_steps'] for r in own);clipped = sum(r['guard_clipped_steps'] for r in own)
        cohorts[level] = {'episodes': len(own), 'successes': sum(r['success'] for r in own),
            'passed': None if level in REPORT_ONLY else gate(level, own), 'report_only': level in REPORT_ONLY,
            'guard_checked_steps': checked, 'guard_clipped_steps': clipped, 'guard_clip_rate': clipped/checked if checked else None,
            'failure_modes': dict(Counter(r['outcome'] for r in own if not r['success'])),
            'appearance_checks': len(checks), 'minimum_mask_iou': min([c['mask_iou'] for c in checks], default=None),
            'maximum_mean_color_diff_255': max([c['mean_abs_color_diff_255'] for c in checks], default=None),
            'maximum_pixel_diff_255': max([c['max_abs_color_diff_255'] for c in checks], default=None),
            'replenishment_delays': [e['delay_steps'] for r in own for e in r['replenish_events']],
            'early_start': sum(bool(r['early_start_steps']) for r in own),
            'budget_exact_N': sum(r['budget_exhausted_exact_N'] for r in own),
            'max_visibility': {cam: {k: max([r['max_visibility'][cam][k] for r in own], default=0)
                for k in ('basket','basket_fruit','inactive_reserve','lamp')} for cam in ('front_camera','hand_camera')}}
    experts = [r for r in rows if r['level'] == 'history'];timing = {}
    for N in ('all', *range(2,11)):
        own = experts if N == 'all' else [r for r in experts if r['N'] == N]
        timing[str(N)] = {'episodes': len(own), 'first_basket_after_green': descriptive([r['basket_entries'][0]['step']-r['green_step'] for r in own if r['basket_entries']]),
            'inter_basket_intervals': descriptive([b['step']-a['step'] for r in own for a,b in zip(r['basket_entries'],r['basket_entries'][1:])])}
    named = {name: read_json(OUT/name) if (OUT/name).exists() else None for name in
        ('cpu_preflight.json','R8_information_definition_crosscheck.json','memory_tensor_checks.json','budget_check.json','early_start_check.json',
         'information_audit.json','R0_gate.json','data_delivery.json')}
    costs = [read_json(path) for path in sorted((OUT/'replays').glob('budget_*/*/*/ep*/cost_comparison.json'))]
    preserve()
    summary = {'revision':9, 'phase':'R0/data_only', 'created_utc':timestamp(), 'state':state,
        'count_R0_replays':len(rows), 'count_R0_target':350, 'cohorts':cohorts,
        'expert_timing_by_N':timing, 'checks':named, 'accepted_occluded_R0_not_rerun':True,
        'simulation_cost_comparisons':costs,
        'distance_signal':'any_hanging', 'arms':['K0','K8','KA'], 'temporal_index':'order',
        'D5_D6_R7_R8_artifacts_unchanged':True,
        'preservation_scope':'Frozen result trees, reports, checkpoints and seal documents; authorized live source changes have 15401b6 snapshots and default-behavior checks.',
        'training':False, 'val150_199_used':False,
        'cases':[{k:r[k] for k in ('task','level','episode','N','success','outcome','steps','source')} for r in rows]}
    proof('summary.json',summary)
    terminal = state['state'].startswith('R9_')
    lines = [f"修订9状态：{state['state']}；{'已停止等待审阅' if terminal or state['state']=='awaiting_data_review' else '仍在准备或执行，未宣称通过'}。",
        f"计数R0完成{len(rows)}/350条；遮挡R0沿用已审阅结果，不重跑。",
        '记忆条件为K0/K8/KA，16帧及模型结构不变；新流程统一any_hanging及顺序时间编码。',
        f"原因：{state.get('reason','按固定队列继续；全部R0门槛通过才生成数据')}。",
        'D5/D6/旧R0产物保持，不训练、不写论文、不使用val150–199。','',
        '## 开跑前固定的规则','',DOC.read_text() if DOC.exists() else '规则草案尚在准备。','',
        '## 计数捷径审计','', '| 条件 | 成功 | 门槛 | 判定 | 裁剪步/检查步 | 失败 |',
        '|---|---:|---|---|---:|---|']
    for level,c in cohorts.items():
        threshold = '只报告' if level in REPORT_ONLY else '≥45/50' if level=='history' else '≤15/50'
        lines.append(f"| {level} | {c['successes']}/{c['episodes']} | {threshold} | {c['passed']} | {c['guard_clipped_steps']}/{c['guard_checked_steps']} | {c['failure_modes']} |")
    def link(name):return f'[{name}]({TRACKED}/{name})' if (TRACKED/name).exists() else name+'：待执行'
    lines += ['', '## 检查与信息可得性','',f'完整机器记录：{link("summary.json")}。',
        f'默认兼容性与遮挡第0帧只读审计：{link("cpu_preflight.json")}。',
        f'信息定义与已审阅R8数字复核：{link("R8_information_definition_crosscheck.json")}。',
        f'张量默认兼容性、顺序编码、训练/推理一致性：{link("memory_tensor_checks.json")}。',
        f'预算与实体/耗时：{link("budget_check.json")}。',
        f'信息可得性：{link("information_audit.json")}。它是专家轨迹的信息上限，不代表模型一定会使用。','',
        '| 压力角色/布局 | R8总耗时(s) | R9总耗时(s) | 倍数(≤2) | R9果实/全部scene实体 |',
        '|---|---:|---:|---:|---:|']
    for c in costs:lines.append(f"| {c['purpose']}/ep{c['episode']} | {c['R8_wall_seconds']:.3f} | {c['R9_wall_seconds']:.3f} | {c['ratio']:.3f} | {c['R9_fruit_actors']}/{c['R9_scene_entities']} |")
    lines += ['',
        '## 按N分组的专家耗时','', '| N | 回合 | 首颗入篮(灯亮后) | 相邻入篮间隔 |', '|---|---:|---|---|']
    for N,t in timing.items():lines.append(f"| {N} | {t['episodes']} | {t['first_basket_after_green']} | {t['inter_basket_intervals']} |")
    lines += ['', '## 决策与后续','',
        '任一门槛或检查失败：保持失败与未完成队列，立即停止，不调参、不换审计种子、不改定义。',
        '全部R0通过：自动生成双任务干净0–99与DART100–199，构建一次共用数据和归一化统计；数据完成即停，不训练。',
        f'原始回放与图像：{OUT}/replays。新数据交付记录：{OUT}/data_delivery.json。','']
    text='\n'.join(lines)
    REPORT.write_text(text);(OUT/'REPORT.md').write_text(text)
    return summary
