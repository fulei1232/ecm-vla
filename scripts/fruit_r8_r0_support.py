"""R8 independent input/guard audits, event diagnostics and review-only delivery."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

from v_a_common import ECM,read_json,read_jsonl,atomic_json,sha256,timestamp
from posthoc_fruit_memory_usage import tree_state
sys.path[:0]=[str(ECM),str(ECM/'src')]
from fruit_picking.task_v2_spec import choose_count
from fruit_picking.task_v2_audit_r8 import fit_clock
from memory_management.event_keyframes import detect_events
from memory_management.fruit_rescue_platform import guard_v2

OUT=ECM/'results/fruit_task_v2_R0_r8_20261105'
TRACKED=ECM/'reports/fruit_task_v2_R0_r8_20261105'
REPORT=ECM/'reports/fruit_task_v2_design_r8_20261105.md'
DOC=ECM/'reports/fruit_task_v2_R0_r8_plan_20261105.md'
REGISTRY=ECM/'reports/split_registry_fruit_task_v2_r8_20261105.json'
TASKS=('count','occluded_ripe')
COHORTS=[('count','history'),('occluded_ripe','history'),('occluded_ripe','initial_current'),
    ('occluded_ripe','current'),('count','current'),('count','initial_current'),('count','initial_ground'),
    ('count','clock_onset'),('count','clock')]


def preserve():
    for name in ('fruit_lowlevel_v2_eval_20261104','fruit_memory_usage_diag_20261104','fruit_task_v2_R0_20261105'):
        assert tree_state(ECM/'results'/name)==read_json(OUT/'preservation'/f'{name}_tree.json'),name
    for p,h in read_json(OUT/'preservation/files_sha256.json').items():assert sha256(p)==h,p


def gate(task,level,rows):
    if len(rows)!=50:return None
    if level=='clock_onset':return True
    n=sum(r['success'] for r in rows)
    high=level=='history' or task=='occluded_ripe' and level=='initial_current'
    passed=n>=45 if high else n<=15
    if level=='history':
        checked=sum(r['guard_checked_steps'] for r in rows);clipped=sum(r['guard_clipped_steps'] for r in rows)
        passed &= checked>0 and clipped/checked<.005
    return bool(passed)


def audit_case(d):
    r=read_json(d/'result.json');trace=read_jsonl(d/'trace.jsonl');decisions=read_json(d/'decisions.json');actions=read_json(d/'actions.json')
    assert len(trace)==len(actions)+1==r['steps']+1
    assert all(t['step']==i for i,t in enumerate(trace))
    assert 30<=r['steps']<=r['budget']
    assert not any(v[k] for v in r['max_visibility'].values() for k in ('basket','basket_fruit','inactive_reserve'))
    guard=read_json(ECM/'results/fruit_k0v2_gate_20261102/guard_v2.json')
    for i,a in enumerate(actions):
        assert a['step']==i+1
        expected,details=guard_v2(a['proposed'],a['measured_arm_qpos'],guard)
        assert np.array_equal(expected,np.asarray(a['executed'])) and details['guard_clipped']==a['guard_clipped']
        if i<30:assert a['phase']=='observe' and a['executed'][7]==1
        if a['phase']=='wait_for_green':assert a['executed']==actions[0]['executed']
    assert r['guard_checked_steps']==len(actions)-30
    assert r['guard_clipped_steps']==sum(a['guard_clipped'] for a in actions[30:])
    if r['task']=='count':assert not r['early_start_steps']
    fits=read_json(OUT/'clock_fit.json') if r['level'] in ('clock','clock_onset') else None
    if fits:assert r['clock_fit_sha256']==sha256(OUT/'clock_fit.json')
    for decision in decisions:
        p=decision['provided'];t=decision['step'];view=trace[t]['visible'];level=r['level']
        assert set(p)=={'tree','ground','lamp_green','N','visible_basket_count','frame0','release_events','t','green_step'}
        assert p['ground']==view['ground'] and p['lamp_green']==view['lamp_green']
        assert p['visible_basket_count']==0
        if not (r['task']=='occluded_ripe' and level=='history'):assert p['tree']==view['tree']
        assert p['frame0']==(None if level=='current' else trace[0]['visible'])
        if level!='history':assert p['release_events'] is None
        else:assert p['release_events']==[e for e in r['own_release_events'] if e['step']<=t]
        if level not in ('clock','clock_onset'):assert p['t'] is None
        else:assert p['t']==t
        if level!='clock_onset':assert p['green_step'] is None
        else:assert p['green_step']==r['green_step']
        if r['task']=='count':
            clock=fits['relative' if level=='clock_onset' else 'absolute'] if fits else None
            expect=choose_count(level,p['N'],p['tree'],p['frame0']['tree'] if p['frame0'] else None,
                len(p['release_events']) if p['release_events'] is not None else None,p['visible_basket_count'],
                len(p['ground']),p['t'],clock,p['green_step'])
            assert expect==decision['chosen_slot']
            assert all(0<=f['slot']<r['M'] and f['id']==f['slot'] for f in p['tree'])
        elif level!='history' and t>=20:assert all(f['ripe_visible'] is None for f in p['tree'])
    assert all(c['passed'] and c['mask_iou']>=.98 and c['mean_abs_color_diff_255']<=2 for c in r['appearance_checks'])
    if r['task']=='count' and r['outcome']=='finish_signal':assert r['appearance_final_hold_available']
    if r['success'] and r['task']=='count':assert r['outcome']=='finish_signal' and r['steps']<1800 and r['final_robot_in_basket']==r['N']
    assert r['outcome']!='wind_basket_anomaly'
    return {'passed':True,'decisions':len(decisions),'actions':len(actions),'source_sha256':
        {n:sha256(d/n) for n in ('result.json','trace.jsonl','actions.json','decisions.json')},
        'image_sha256':{n:sha256(d/n) for s in r['snapshots'].values() for n in s['files'].values()}}


def collect(purpose='audit',task=None,level=None):
    rows=[]
    for f in sorted((OUT/'replays'/purpose).glob('*/*/ep*/validity.json')):
        proof=read_json(f)
        assert proof['passed']
        for n,h in proof['source_sha256'].items():assert sha256(f.parent/n)==h
        r=read_json(f.parent/'result.json')
        if task and r['task']!=task or level and r['level']!=level:continue
        r['source']=str(f.parent);rows.append(r)
    return rows


def fit_calibration():
    rows=collect('calibration','count','history');assert len(rows)==10
    cases=[{'N':r['N'],'green_step':r['green_step'],'basket_entry_steps':[e['step'] for e in r['basket_entries']],
        'decision_points':r['decision_points'],'reference_success':r['success']} for r in rows]
    data={'source_scope':'dev250–259 only; before audit cohorts','source_sha256':{str(Path(r['source'])/'result.json'):sha256(Path(r['source'])/'result.json') for r in rows},
        'absolute':fit_clock(cases,False),'relative':fit_clock(cases,True),'created_utc':timestamp()}
    if (OUT/'clock_fit.json').exists():
        old=read_json(OUT/'clock_fit.json');data['created_utc']=old['created_utc'];assert old==data
    else:atomic_json(OUT/'clock_fit.json',data)
    return data


def budget_check():
    experts=collect('budget_expert','count','history');nonstop=collect('budget_nonstop','count','nonstop')
    assert all(r['N']==5 and r['green_step']==630 and r['budget']==1800 for r in experts+nonstop)
    finish=[r['steps'] for r in experts if r['success']]
    over=[r['steps'] for r in nonstop if r['outcome']=='overpick' and r['final_robot_picked']==6]
    max_finish=max(finish) if finish else None;max_over=max(over) if over else None
    passed=(len(experts)==len(nonstop)==10 and len(finish)==len(over)==10
        and 1800-max_finish>=150 and max_over<1800)
    first=[r['basket_entries'][0]['step']-r['green_step'] for r in experts if r['basket_entries']]
    gaps=[b['step']-a['step'] for r in experts for a,b in zip(r['basket_entries'],r['basket_entries'][1:])]
    proposal=max(1800,(max_finish+150) if max_finish else 1800,
        630+max(first)+5*max(gaps)+40+150 if first and gaps else 1800)
    proof={'passed':bool(passed),'budget':1800,'D':600,'N':5,'dev_episodes':list(range(250,260)),
        'expert_complete':len(finish),'expert_attempts':len(experts),'max_finish_step_including_hold':max_finish,
        'minimum_remaining_steps':1800-max_finish if max_finish else None,
        'nonstop_overpick':len(over),'nonstop_attempts':len(nonstop),'max_Nplus1_detach_step':max_over,
        'proposed_budget_if_timing_failed':int(proposal) if not passed else None,
        'unresolved_motion_failures':[(r['episode'],r['outcome']) for r in experts+nonstop if r['outcome'] not in ('finish_signal','overpick')],
        'note':'Fixed worst-parameter diagnostic overrides only; geometry seeds unchanged; no automatic budget update.'}
    atomic_json(OUT/'budget_check.json',proof);return proof


def event_diagnostics(rows):
    data={}
    for task in TASKS:
        own=[r for r in rows if r['task']==task and r['level']=='history']
        entry={'episodes':len(own),'versions':{}}
        for version in ('mature','any_hanging'):
            total=Counter();records=[]
            for r in own:
                trace=read_jsonl(Path(r['source'])/'trace.jsonl');d=np.array([f['signals'][version] for f in trace])
                events=detect_events(d,.03,.005)
                triggers={x for e in events for x in (e['start'],e['release']) if x is not None}
                keys={x for e in events for x in (e['contact'],e['release']) if x is not None}
                lamp=[r['green_step']] if r['green_step'] is not None else []
                replenish=[e['step'] for e in r['replenish_events']]
                record={'episode':r['episode'],'events':len(events),'event_intervals':events,
                    'lamp_transition_near_edges':sum(t in triggers for t in lamp),
                    'lamp_transition_keyframes':sum(t in keys for t in lamp),
                    'replenishment_near_edges':sum(t in triggers for t in replenish),
                    'replenishment_keyframes':sum(t in keys for t in replenish)}
                for k in ('events','lamp_transition_near_edges','lamp_transition_keyframes','replenishment_near_edges','replenishment_keyframes'):total[k]+=record[k]
                records.append(record)
            entry['versions'][version]={'totals':dict(total),'records':records}
        data[task]=entry
    return {'source':'frozen event_keyframes.detect_events(radius=.03,tol=.005)',
        'decision':'No signal version adopted; both candidates await R0 review.',
        'maturity_sensitive_uses_GT_labels':True,'any_hanging_accepts_no_maturity_argument':True,
        'both_use_GT_geometry_and_hanging_state':True,'attribution':'coincidences at exact physical event step, not proof of causation',
        'tasks':data}


def descriptive(values):
    return {'n':len(values),'mean':float(np.mean(values)) if values else None,
        'std':float(np.std(values)) if values else None,'range':[min(values),max(values)] if values else None}


def timing(rows):
    count=[r for r in rows if r['task']=='count' and r['level']=='history']
    result={}
    for N in ('all',2,3,4,5):
        own=count if N=='all' else [r for r in count if r['N']==N]
        first=[r['basket_entries'][0]['step']-r['green_step'] for r in own if r['basket_entries']]
        gaps=[b['step']-a['step'] for r in own for a,b in zip(r['basket_entries'],r['basket_entries'][1:])]
        result[str(N)]={'episodes':len(own),'first_basket_after_green':descriptive(first),'inter_basket_intervals':descriptive(gaps)}
        result[str(N)]['red_light_wait_steps']=descriptive([r['green_step']-30 for r in own])
        result[str(N)]['red_wait_fraction_after_prefix']=descriptive([(r['green_step']-30)/(r['steps']-30) for r in own if r['steps']>30])
    return result


def report(state):
    rows=collect();diagnostics=sum(len(collect(p)) for p in ('calibration','budget_expert','budget_nonstop'))
    cohorts={}
    for task,level in COHORTS:
        own=[r for r in rows if r['task']==task and r['level']==level]
        checks=[c for r in own for c in r['appearance_checks']]
        total=sum(r['guard_checked_steps'] for r in own);clipped=sum(r['guard_clipped_steps'] for r in own)
        cohorts[f'{task}/{level}']={'episodes':len(own),'successes':sum(r['success'] for r in own),
            'passed':gate(task,level,own) if level!='clock_onset' else None,'report_only':level=='clock_onset',
            'guard_checked_steps':total,'guard_clipped_steps':clipped,'guard_clip_rate':clipped/total if total else None,
            'failure_modes':dict(Counter(r['outcome'] for r in own if not r['success'])),
            'appearance_checks':len(checks),'minimum_mask_iou':min([c['mask_iou'] for c in checks],default=None),
            'maximum_mean_color_diff_255':max([c['mean_abs_color_diff_255'] for c in checks],default=None),
            'maximum_pixel_color_diff_255':max([c['max_abs_color_diff_255'] for c in checks],default=None),
            'last_hold_image_comparisons':sum(r['appearance_final_hold_available'] for r in own),
            'replenishment_delay_count':sum(e['delay_steps']>0 for r in own for e in r['replenish_events']),
            'replenishment_delay_steps':[e['delay_steps'] for r in own for e in r['replenish_events']],
            'positive_replenishment_delay_steps':descriptive([e['delay_steps'] for r in own for e in r['replenish_events'] if e['delay_steps']>0]),
            'early_start_episodes':sum(bool(r['early_start_steps']) for r in own),
            'budget_exhausted_exact_N':sum(r['budget_exhausted_exact_N'] for r in own),
            'max_visibility':{camera:{k:max([r['max_visibility'][camera][k] for r in own],default=0)
                for k in ('basket','basket_fruit','inactive_reserve','lamp')}
                for camera in ('front_camera','hand_camera')}}
    events=event_diagnostics(rows)
    budget=read_json(OUT/'budget_check.json') if (OUT/'budget_check.json').exists() else None
    clock=read_json(OUT/'clock_fit.json') if (OUT/'clock_fit.json').exists() else None
    geometry=read_json(OUT/'geometry_unit_checks.json') if (OUT/'geometry_unit_checks.json').exists() else None
    native=read_json(OUT/'ep245_native_replay.json') if (OUT/'ep245_native_replay.json').exists() else None
    d5=read_json(OUT/'D5_readonly_basket_audit.json')
    preserve()
    summary={'revision':8,'stage':'R0','state':state,'created_utc':timestamp(),
        'formal_replays':len(rows),'formal_target':450,'calibration_and_stress_replays':diagnostics,
        'cohorts':cohorts,'budget_check':budget,'clock_fit':clock,'geometry_checks':geometry,
        'ep245_native_replay_passed':bool(native and native['passed']),
        'D5_readonly_scan':{k:d5[k] for k in ('scanned_episodes','affected_episodes','flagged_fruits')},
        'event_signal_comparison':events,'expert_timing_by_N':timing(rows),
        'D5_D6_R7_unchanged':True,'no_training_data_or_models':True,'no_val_rollouts':True,
        'extra_early_start_clock_queue':False,'early_start_clock_risk_recorded':True,
        'cases':[{k:r[k] for k in ('task','level','episode','N','M','success','outcome','steps','source')} for r in rows]}
    TRACKED.mkdir(exist_ok=True)
    for d in (OUT,TRACKED):atomic_json(d/'summary.json',summary)
    stop=state['state'] in ('R8_check_failed','R8_budget_failed','R8_audit_failed','R8_engineering_error')
    complete=state['state']=='awaiting_R0_review'
    heading='R0修订8已通过并停止等待审阅。' if complete else 'R0修订8触发停止条件，未通过；等待用户审阅。' if stop else 'R0修订8候选准备/审计中，未宣称通过。'
    lines=[heading,f'全新dev门槛回放完成{len(rows)}/450条，独立校准/压力回放完成{diagnostics}/30条；不复用旧候选结果。',
        f'几何单元检查已完成，ep245物理重放通过={bool(native and native["passed"])}；D5只读扫描1200条，末帧异常{d5["affected_episodes"]}回合。',
        f'当前状态={state["state"]}，原因={state.get("reason","等待物理检查和队列完成")}。',
        'D5/D6和旧R0产物不改；不生成训练数据、不训练、不写论文；R0后等待用户审阅。','',
        '## 开跑前固定的任务定义与信息边界','',DOC.read_text(),'',
        '## 入篮几何与回归检查','',
        f'底板世界z为[0.150,0.156]m；篮壁/篮口z为[0.150,0.270]m。v2严格使用0<rel_z<0.12，两任务都生效；v1原判定保持不变。单元检查：{geometry}。',
        f'原ep245的89条执行命令物理重放：{"已通过" if native else "尚未执行"}；完整记录`{OUT}/ep245_native_replay.json`。',
        f'D5仅扫描已保存末帧，异常果实{d5["flagged_fruits"]}颗、回合{d5["affected_episodes"]}/1200；没有重跑、重判或修改D5。逐例：`{OUT}/D5_readonly_basket_audit.json`。','',
        '## 预算实测（10个固定布局，D=600、N=5）','',f'{budget if budget else "尚未完成，不能宣称1800步可行"}。',
        '预算不满足时仅提出候选值并停止，不自动调整；规划/物理失败导致无法核算时明确标为未验证，不将延长预算说成能修复规划。','',
        '## 捷径与专家审计','',
        '| 任务/信息条件 | 成功 | 门槛 | 通过 | 裁剪步/检查步 | 典型失败 |','|---|---:|---|---|---:|---|']
    for name,c in cohorts.items():
        task,level=name.split('/');high=level=='history' or task=='occluded_ripe' and level=='initial_current'
        rule='只报告' if level=='clock_onset' else '≥45/50' if high else '≤15/50'
        lines.append(f'| {name} | {c["successes"]}/{c["episodes"]} | {rule} | {c["passed"]} | {c["guard_clipped_steps"]}/{c["guard_checked_steps"]} | {c["failure_modes"]} |')
    ia,ig=(cohorts[f'count/{l}'] for l in ('initial_current','initial_ground'))
    lines+=['',f'计数(ii)取ii-a和ii-g较高者；当前两者完成数{ia["episodes"]}/50、{ig["episodes"]}/50。未满两队列时不伪造较高成功率。',
        '专家裁剪率分母去掉前30步，包含红灯等待和完成保持；要求严格<0.5%。时钟iv-o无门槛。','',
        '## 补果外观与延迟','',
        '| 条件 | 比较次数 | 最低IoU | 最大平均色差/255原始值 | 最大单像素色差原始值 | 完成保持末帧比较数 | 延迟次数/步数统计 |','|---|---:|---:|---:|---:|---:|---|']
    for name,c in cohorts.items():
        if name.startswith('count/'):
            lines.append(f'| {name} | {c["appearance_checks"]} | {c["minimum_mask_iou"]} | {c["maximum_mean_color_diff_255"]} | {c["maximum_pixel_color_diff_255"]} | {c["last_hold_image_comparisons"]} | {c["replenishment_delay_count"]}/{c["positive_replenishment_delay_steps"]} |')
    lines+=['','平均色差门槛是原始像素值≤2，即≤2/255；最大差值只报告。无完成保持段的失败回合记N/A，另检查每次补果后的相同ROI，不将N/A记为通过。','',
        '## 专家时间统计（dev200–249）','',
        '| N | 回合数 | 首次入篮：均值/标准差/范围 | 相邻间隔：均值/标准差/范围 |','|---|---:|---|---|']
    for N,v in summary['expert_timing_by_N'].items():lines.append(f'| {N} | {v["episodes"]} | {v["first_basket_after_green"]} | {v["inter_basket_intervals"]} |')
    lines+=['','## 时钟拟合与预先固定的停止规则','',f'校准结果：{clock if clock else "尚未拟合"}。',
        '拟合及偏置选择仅用dev250–259，之后冻结。iv不给灯亮时刻；iv-o记住首次观测到的灯亮时刻、使用同一校准批次拟合的相对时间模型，只作解释。','',
        '## 事件信号特权信息检查','',
        '原信号读取成熟度真值；遮挡中性化后，K4/K8的选帧仍可能间接泄露成熟度。计数全为成熟果，两版本应一致。any_hanging不接收成熟度参数，但仍使用几何和挂树状态真值，不能宣称完全无特权信息。',
        '指示灯只改颜色，不进入距离函数；补果改变候选几何，但生成时TCP至少距原位15cm，远于3cm近场半径。下表报告实测同一步的近场边沿和关键帧重合，不把重合当作因果证明。',
        '| 任务/版本 | 专家回合 | 近场事件 | 灯变化边沿/关键帧 | 补果边沿/关键帧 |','|---|---:|---:|---|---|']
    for task,data in events['tasks'].items():
        for version,v in data['versions'].items():
            t=v['totals'];lines.append(f'| {task}/{version} | {data["episodes"]} | {t.get("events",0)} | {t.get("lamp_transition_near_edges",0)}/{t.get("lamp_transition_keyframes",0)} | {t.get("replenishment_near_edges",0)}/{t.get("replenishment_keyframes",0)} |')
    lines+=['','两个版本目前均为诊断候选，采用哪一个留给用户R0审阅，之后三个模型统一使用；没有修改K0/K4/K8定义。','',
        '## 相机截图和可见性','']
    for task in TASKS:
        pictures=[r for r in rows if r['task']==task and r['level']=='history']
        if not pictures:pictures=[r for p in ('budget_expert','calibration') for r in collect(p,task,'history')]
        required=('before_green','after_green','after_wind','before_grasp','release','after_replenishment','finish_hold') if task=='count' else ('opening','post_gust','before_grasp','release')
        if not pictures:lines.append(f'{task}：尚无新回放截图。');continue
        picture=sorted(pictures,key=lambda r:(-sum(s in r['snapshots'] for s in required),r['episode']))[0]
        lines += [f'{task}图例来自{picture["purpose"]}/dev{picture["episode"]}，N={picture["N"]}；按阶段完整度、episode序选择，不按成功结果筛图。',
            f'该例双相机各物体最大像素数：{picture["max_visibility"]}。全部逐步像素在trace.jsonl，所有篮子/篮内果实像素须为0。']
        for stage in required:
            if stage not in picture['snapshots']:lines.append(f'{stage}：未到达，缺图。');continue
            shot=picture['snapshots'][stage];lines += ['',f'{stage}，物理步{shot["step"]}：']
            for camera,name in shot['files'].items():lines.append(f'![{task} {stage} {camera}]({Path(picture["source"])/name})')
    lines+=['','## 全部门槛回放的像素最大值','',
        '| 条件 | 前置相机：篮子/篮内果/停放备用/灯 | 腕部相机：篮子/篮内果/停放备用/灯 |','|---|---|---|']
    for name,c in cohorts.items():lines.append(f'| {name} | {c["max_visibility"]["front_camera"]} | {c["max_visibility"]["hand_camera"]} |')
    lines+=['','| 条件 | early_start回合 | 预算耗尽但恰好N颗入篮 |','|---|---:|---:|']
    for name,c in cohorts.items():lines.append(f'| {name} | {c["early_start_episodes"]} | {c["budget_exhausted_exact_N"]} |')
    lines+=['','## 停止与产物','',
        f'完整机器摘要：`{TRACKED}/summary.json`。源数据、动作、有限信息输入及逐例有效性：`{OUT}/replays`。',
        f'划分：`{REGISTRY}`；val150–199仅保留，不生成、不用；val0–149排除。',
        '用户明确选择不增加提前开工时钟队列。early_start不罚会留下提前开工再按时钟停的风险，本次如实记录，既定门槛不据此改变。',
        '所有门槛/几何/外观/可见性/预算检查失败即停，不调参、不换种子、不自行修改任务定义；不生成训练数据、不训练、不写论文。']
    body='\n'.join(lines)+'\n';REPORT.write_text(body);(OUT/'REPORT.md').write_text(body)
    return summary
