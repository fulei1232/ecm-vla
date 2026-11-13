"""R0-only preservation, information audits and reporting; no training pipeline imports."""
from collections import Counter
from pathlib import Path
import sys
import numpy as np

from v_a_common import ECM, atomic_json, read_json, sha256, timestamp
from posthoc_fruit_memory_usage import tree_state
sys.path.insert(0, str(ECM))
from fruit_picking.task_v2_spec import choose_count

OUT = ECM/'results/fruit_task_v2_R0_20261105'
REPORT = ECM/'reports/fruit_task_v2_design_20261105.md'
PLAN_DOC = ECM/'reports/fruit_task_v2_R0_plan_20261105.md'
REGISTRY = ECM/'reports/split_registry_fruit_task_v2_20261105.json'
TASKS = ('count', 'occluded_ripe')
LEVELS = ('current', 'initial_current', 'history')


def cohort_gate(task, level, episodes, successes, checked, clipped):
    if episodes != 50:return None
    low = task == 'count' and level != 'history' or task == 'occluded_ripe' and level == 'current'
    passed = successes <= 15 if low else successes >= 45
    if level == 'history':passed &= checked > 0 and clipped/checked < .005
    return bool(passed)


def preserve():
    for name in ('fruit_lowlevel_v2_eval_20261104', 'fruit_memory_usage_diag_20261104'):
        assert tree_state(ECM/'results'/name) == read_json(OUT/'preservation'/f'{name}_tree.json'), name
    for p, h in read_json(OUT/'preservation/files_sha256.json').items():
        assert sha256(p) == h, p


def audit_information(directory):
    result = read_json(directory/'result.json')
    rows = read_json(directory/'decisions.json')
    level, task = result['level'], result['task']
    for r in rows:
        v = r['provided']
        assert set(v) == {'current_tree', 'N', 'frame0_tree', 'delivery_events', 'visible_basket_count'}
        if level == 'current':assert v['frame0_tree'] is None and v['delivery_events'] is None
        if level == 'initial_current':assert v['delivery_events'] is None
        if task == 'occluded_ripe' and r['physical_step'] >= 20 and level != 'history':
            assert all(f['ripe_visible'] is None for f in v['current_tree'])
        if task == 'count':
            expect = choose_count(level, v['N'], v['current_tree'], v['frame0_tree'],
                len(v['delivery_events']) if level == 'history' else None, v['visible_basket_count'])
            assert expect == r['chosen_fruit']
    sys.path.insert(0,str(ECM/'src'))
    from memory_management.fruit_rescue_platform import guard_v2
    limits=read_json(ECM/'results/fruit_k0v2_gate_20261102/guard_v2.json')
    actions=read_json(directory/'actions.json')
    assert len(actions)==result['steps'] and 30<=len(actions)<=1300
    for i,a in enumerate(actions):
        assert a['step']==i+1
        expected,details=guard_v2(a['proposed'],a['measured_arm_qpos'],limits)
        assert np.array_equal(expected,np.asarray(a['executed']))
        assert details['guard_clipped']==a['guard_clipped']
        if i<30:assert a['phase']=='observe' and a['executed'][7]==1
    assert result['guard_checked_steps']==len(actions)-30
    assert result['guard_clipped_steps']==sum(a['guard_clipped'] for a in actions[30:])
    return {'passed': True, 'decisions': len(rows), 'level': level,
        'selection_inputs_exclude_seed_clock_hidden_wind_and_global_basket_count': True,
        'result_sha256': sha256(directory/'result.json'), 'decisions_sha256': sha256(directory/'decisions.json'),
        'actions_sha256':sha256(directory/'actions.json'),'trace_sha256':sha256(directory/'trace.jsonl'),
        'screenshot_sha256':{name:sha256(directory/name) for shot in result['snapshots'].values() for name in shot['files'].values()}}


def report(state):
    rows = []
    for f in sorted((OUT/'replays').glob('*/*/ep*/result.json')):
        if not (f.parent/'validity.json').exists():continue
        proof=read_json(f.parent/'validity.json')
        assert proof['passed'] and proof['result_sha256']==sha256(f)
        r = read_json(f);r['source'] = str(f.parent);rows.append(r)
    cohorts = {}
    for task in TASKS:
        cohorts[task] = {}
        for level in LEVELS:
            own = [r for r in rows if r['task'] == task and r['level'] == level]
            n, success = len(own), sum(r['success'] for r in own)
            low = task == 'count' and level != 'history' or task == 'occluded_ripe' and level == 'current'
            checked = sum(r['guard_checked_steps'] for r in own)
            clipped = sum(r['guard_clipped_steps'] for r in own)
            rate = clipped/checked if checked else None
            passed = cohort_gate(task,level,n,success,checked,clipped)
            cohorts[task][level] = {'episodes': n, 'success': success, 'success_rate': success/n if n else None,
                'threshold': '<=15/50' if low else '>=45/50', 'passed': passed,
                'guard_clipped_steps': clipped, 'guard_checked_steps': checked, 'guard_clip_rate': rate,
                'failure_modes': dict(Counter(r.get('failure_mode',r['outcome']) for r in own if not r['success']))}
    preserve()
    screen=read_json(OUT/'logical_screen.json') if (OUT/'logical_screen.json').exists() else None
    cameras={task:{camera:{'episodes':sum(r['task']==task for r in rows),
        'episodes_with_basket_visible':sum(bool(r.get('basket_seen',{}).get(camera,[])) for r in rows if r['task']==task),
        'control_frames_with_basket_visible':sum(len(r.get('basket_seen',{}).get(camera,[])) for r in rows if r['task']==task)}
        for camera in ('front_camera','hand_camera')} for task in TASKS}
    summary = {'revision': 7, 'stage': 'R0', 'state': state, 'episodes_completed': len(rows), 'logical_screen':screen,
        'camera_visibility':cameras,
        'cohorts': cohorts, 'engineering_errors': state.get('engineering_errors', []),
        'D5_D6_unchanged': True, 'train_data_generated': False, 'models_trained': False,
        'paper_modified': False, 'registry': str(REGISTRY), 'generated_utc': timestamp(), 'records': rows}
    atomic_json(OUT/'summary.json', summary)
    tracked = ECM/'reports/fruit_task_v2_R0_20261105';tracked.mkdir(exist_ok=True)
    atomic_json(tracked/'summary.json', summary)
    stop = state.get('state') in ('R0_audit_failed', 'R0_engineering_error')
    complete = state.get('state') == 'awaiting_R0_review'
    heading = ('R0审计未通过，按§15.6停止，等待用户审阅。' if stop else
               'R0审计通过，按§15.4停止，等待用户审阅。' if complete else 'R0候选设计已准备；物理审计尚未完成，不宣称通过。')
    lines = [heading, 'D6原报告保留C；用户裁决按B处理，作为新任务设计依据。D5/D6封存产物不改。',
        f'已完成{len(rows)}/300条计划内dev审计；没有生成训练数据、训练模型或使用val150–199。',
        f'停止原因：{state.get("reason", "待完成三类策略、相机截图及专家/裁剪率审计")}。',
        '不写进论文；R0审阅后才能进入数据生成与新预登记流程。', '',
        '## 任务语义与实现说明', '', PLAN_DOC.read_text(), '',
        '## 实测结果（未完成项明确留空）', '',
        '| 任务 | 信息限制 | 成功 | 门槛 | 通过 | 包络裁剪步/检查步 | 失败方式 |', '|---|---|---:|---|---|---:|---|']
    for task, levels in cohorts.items():
        for level, r in levels.items():
            lines.append(f'| {task} | {level} | {r["success"]}/{r["episodes"]} | {r["threshold"]} | {r["passed"]} | {r["guard_clipped_steps"]}/{r["guard_checked_steps"]} | {r["failure_modes"]} |')
    lines += ['', '只有满50条才判断队列门槛；专家的裁剪率以去掉开头30步后的执行步为分母，须严格<0.5%。出现失败即停，不追加调参、换seed或重试。']
    if screen:
        lines += ['', '## CPU逻辑排查（不是物理R0审计）', '',
            f'固定50个dev布局，在假设抓放完美且能数清当前树果的理想模型下：猜风落1颗成功{screen["guess1_success"]}/50，猜2颗成功{screen["guess2_success"]}/50，仅看剩1颗停止成功{screen["current_remaining1_success"]}/50。',
            '这些数字不能当作真实专家成功率或R0通过/失败的物理证据，但明确提示二元风落歧义仍可被猜中。本次物理审计在开始前固定更强的猜2基线；不把实际风落数交给它。',
            f'逐布局逻辑记录：`{OUT}/logical_screen.json`。']
    lines += ['',
        '## 相机截图与篮子可见性', '', '以下只展示实际物理回放的原始截图；缺失阶段不补画或伪造。每例JSON还保存逐控制步的双相机篮子分割像素计数。']
    for task in TASKS:
        lines += ['',f'{task}全部已审计回放的双相机篮子可见性：{cameras[task]}。']
        own = sorted([r for r in rows if r['task'] == task and r['level'] == 'history'], key=lambda r:r['episode'])
        if not own:
            lines += ['', f'{task}：尚无专家回放，截图与篮子可见性未验证。'];continue
        full=[r for r in own if all(s in r['snapshots'] for s in ('opening','post_gust','before_grasp','release'))]
        r = full[0] if full else own[0];d = Path(r['source'])
        lines += ['', f'示例：{task}/dev{r["episode"]}，结局{r["outcome"]}。按episode升序首个四阶段截图齐全的专家尝试选取，不按成功与否；没有齐全尝试时展示最早尝试的缺图状态。']
        for camera in ('front_camera','hand_camera'):
            seen = r['basket_seen'][camera]
            lines += [f'- {camera}：篮子出现于{len(seen)}个控制帧；首次记录{seen[:1]}。']
        for stage in ('opening','post_gust','before_grasp','release'):
            if stage not in r['snapshots']:
                lines += [f'{stage}：该尝试未到达，缺图。'];continue
            shot = r['snapshots'][stage]
            lines += ['', f'{stage}，物理步{shot["step"]}：']
            for camera, name in shot['files'].items():
                lines += [f'![{task} {stage} {camera}]({d/name})']
    lines += ['', '## 保全与交付', '',
        f'完整机器摘要：`{tracked}/summary.json`。原始回放、信息输入审计及日志：`{OUT}`。',
        'D5/D6结果目录大小、mtime、链接及保护文件sha256与执行前记录一致。划分仅登记；R0不生成train/val示范。',
        f'执行状态：`{OUT}/status.json`。当前{state["state"]}；后续须由用户审阅决定。']
    body = '\n'.join(lines)+'\n';REPORT.write_text(body);(OUT/'REPORT.md').write_text(body)
    return summary
