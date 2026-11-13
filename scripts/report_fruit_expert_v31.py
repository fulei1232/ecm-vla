#!/usr/bin/env python3
"""Revision3 audit, failure coverage and automatic expert selection for stage3."""
import argparse
from collections import Counter
import json
from pathlib import Path

import h5py
import numpy as np

from fruit_expert_audit_stats import summarize_demonstrations,load_episode,fraction
from report_fruit_expert_v3 import failure_breakdown
from report_fruit_expert_v2 import illustration
from v_a_common import ECM,atomic_json,read_json,sha256,timestamp


def augment(stats):
    wrist=[];commands=[];excess={p:[] for p in ('reach','carry')}
    for row in stats['records']:
        _,q,a,_,gt=load_episode(Path(row['manifest']));commands.append(a[:,6])
        phases=[r['phase'] for r in gt];starts=[0]+[i for i in range(1,len(q)) if phases[i]!=phases[i-1]]
        for start,end in zip(starts,starts[1:]+[len(q)]):
            phase=phases[start]
            if phase=='approach': wrist.append(q[end-1,4:7])
            if phase in excess:
                values=q[start:end,0]
                excess[phase].append(max(0.,float(np.abs(np.diff(values)).sum()-abs(values[-1]-values[0]))))
    values=np.asarray(wrist).reshape(-1,3);center=np.median(values,axis=0) if len(values) else np.full(3,np.nan)
    stats['approach_end_wrist']={f'q{j+5}':{'median':float(center[j]) if len(values) else None,
        'within_median_plusminus_1rad':fraction(np.sum(np.abs(values[:,j]-center[j])<=1),len(values))} for j in range(3)}
    a=np.concatenate(commands) if commands else np.empty(0)
    stats['q7_commands_outside_pi']=fraction(np.sum(np.abs(a)>np.pi),len(a))
    stats['q7_command_range']=[float(a.min()),float(a.max())] if len(a) else None
    stats['q1_extra_travel']={p:{'segments':len(rows),'p90':float(np.quantile(rows,.9)) if rows else None,'values':rows}
                              for p,rows in excess.items()}
    return stats


def position_cell(position):
    x,y,z=position
    return ('left' if y>.08 else 'right' if y<-.08 else 'middle')+'/'+('upper' if z>=.54 else 'lower')


def coverage(root):
    counts={t:{f'{side}/{height}':{'target_attempts':0,'failed_attempts':0} for side in ('left','middle','right')
               for height in ('upper','lower')} for t in ('count','occluded_ripe')}
    cases=[];methods=Counter()
    for path in sorted(root.glob('*/attempts/ep*_seed*.json')):
        attempt=read_json(path);task=attempt['task'];directory=path.parent.parent
        h5=directory/attempt['h5' if attempt['ok'] else 'failed_h5']
        with h5py.File(h5) as f:
            raw=f[f"episode_{attempt['episode']}/setup/layout_json"][()]
            layout=json.loads(raw.decode() if isinstance(raw,bytes) else raw)
        latest=None;reach_index=0
        for trace in attempt['planning_audit']:
            if trace.get('phase') in ('reach','carry'):
                invoked=trace.get('method')=='rrt' or any(a.get('method')=='rrt' for a in trace.get('attempts',[]))
                methods['segments']+=1;methods['rrt_used']+=int(invoked)
                methods['method_'+str(trace.get('method'))]+=1
            if trace.get('snapshot'):
                with np.load(trace['snapshot'],allow_pickle=False) as snap: meta=json.loads(str(snap['metadata']))
                latest=meta
                if trace.get('phase')=='reach':
                    target=meta['labels']['target'];counts[task][position_cell(layout['fruits'][target]['pos'])]['target_attempts']+=1
                    reach_index+=1
            elif trace.get('phase')=='reach':
                target=trace.get('target',layout['targets'][reach_index]);reach_index+=1
                counts[task][position_cell(layout['fruits'][target]['pos'])]['target_attempts']+=1
        if not attempt['ok']:
            context=attempt.get('failure_context') or {'phase':latest['phase'],'target':latest['labels']['target']}
            target=context['target'];position=layout['fruits'][target]['pos'];cell=position_cell(position)
            counts[task][cell]['failed_attempts']+=1
            cases.append({'attempt':str(path),'task':task,'seed':attempt['seed'],'phase':context['phase'],
                          'target':target,'initial_position':position,'cell':cell,'reason':attempt['reason']})
    return {'by_task':counts,'failures':cases,'methods':dict(methods),
        'definition':'Target attempt = one reach planning call, including rejected attempts; failure belongs to the target causing episode failure. left:y>.08, right:y<-.08, middle:closed interval; upper:z>=.54, lower:z<.54. Not all unvisited fruits in the layout.'}


def choose_expert(gates,new_stats,old_stats):
    technical=[key for key in gates if key!='attempt_success']
    if not all(gates[key] for key in technical):
        return 'canonical','v3.1 has a non-success core regression; use the accepted v3 baseline, without another expert iteration.'
    if gates['attempt_success']: return 'canonical_v31','All revision3 core gates passed.'
    def score(stats):
        rows=stats['by_task'].values();success=sum(r['success'] for r in rows);attempts=sum(r['success']+r['failed'] for r in rows)
        return success/attempts if attempts else 0.
    if score(new_stats)>=score(old_stats): return 'canonical_v31','Only trial-success gate failed; pooled successful attempts/all attempts is higher or tied (tie favors v3.1).'
    return 'canonical','Only trial-success gate failed; accepted v3 has higher pooled trial success.'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    old_cache=out/'v3_extended_stats.json'
    old=read_json(old_cache) if old_cache.exists() else augment(read_json(ECM/'results/fruit_expert_v3_audit_20261102/canonical_stats.json'))
    atomic_json(old_cache,old)
    new=augment(summarize_demonstrations(out/'demos/rm75b_eg2/train'));atomic_json(out/'v31_stats.json',new)
    planning=read_json(out/'planning/summary.json')
    location={'v3':coverage(ECM/'results/fruit_expert_v3_audit_20261102/demos_settle2/rm75b_eg2/train'),
              'v31':coverage(out/'demos/rm75b_eg2/train')};atomic_json(out/'failure_coverage.json',location)
    failures=failure_breakdown(new);complete=new['episodes']==200
    gates={'planning_determinism':complete and planning['determinism']['passed'],
        'target_continuity':complete and planning['target_continuity']['passed'],
        'method_consistency':complete and planning['method_consistency']['passed'],
        'wrist_consistency':complete and all(new['approach_end_wrist'][q]['within_median_plusminus_1rad']['fraction'] is not None and
            new['approach_end_wrist'][q]['within_median_plusminus_1rad']['fraction']>=.95 for q in ('q5','q6','q7')),
        'q7_range':complete and new['q7_commands_outside_pi']['numerator']==new['q7_states_outside_pi']['numerator']==0,
        'attempt_success':complete and all(row['attempt_success_rate']['fraction'] is not None and
            row['attempt_success_rate']['fraction']>=.9 and
            failures[task]['categories'].get('planning_failure',{}).get('fraction',0)<=.05 for task,row in new['by_task'].items())}
    mode,reason=choose_expert(gates,new,old)
    selection={'generated_utc':timestamp(),'revision':3,'ik_mode':mode,
        'solver_version':'canonical_v3.1_20261102' if mode=='canonical_v31' else 'canonical_v3_20261102',
        'reason':reason,'continue_phase3_without_confirmation':True,'core_gates':gates,
        'success_comparison_rule':'pooled successful attempts/all attempts; per-task values always reported; ties favor v3.1',
        'baseline_v3_reassessment':{'target_continuity':100,'method_consistency':97,'q1_swing_gate':'void by §11.2'},
        'source_audit':str(out/'audit.json')}
    videos=[]
    for task in ('count','occluded_ripe'):
        for episode in (0,1):
            path=out/f'demos/rm75b_eg2/train/{task}/ep{episode}_manifest.json'
            if path.exists(): videos.append(illustration(path,out/'videos'))
    audit={'phase':2,'revision':3,'generated_utc':timestamp(),'core_gates':gates,'all_core_gates_passed':all(gates.values()),
        'selected_expert':selection,'v3':old,'v31':new,'planning':planning,'failure_coverage':location,
        'failure_categories':failures,'v3_collision_diagnosis':read_json(out/'v3_failure_diagnosis/summary.json'),
        'v3_failures_replanned':read_json(out/'v3_failure_replanning/summary.json'),'videos':videos,
        'D1_revision1_metrics':read_json(ECM/'results/fruit_expert_v2_audit_20261102/d1_new_metrics/summary.json')}
    atomic_json(out/'audit.json',audit);atomic_json(out/'summary.json',audit);atomic_json(out/'selection.json',selection)
    ratio=lambda row:f"{row['numerator']}/{row['denominator']}" if row['denominator'] else '无样本'
    lines=[f"阶段2修订3完成：选择{selection['solver_version']}，按§11.5直接进入阶段3。",
        f"v3.1核心项：{gates}。",f"v3.1接受示范{new['episodes']}/200；24次v3失败中重规划成功{audit['v3_failures_replanned']['successes']}/24。",
        '第1关节摆幅门槛已作废，改报告多余行程；连续性与方法一致率分别判定。',
        'v2/v3代码和产物保留；不再迭代专家；下一步重新生成干净/DART并训练K0。','',
        '## 做了什么','',
        '先完成全部24次v3失败碰撞诊断，再新增canonical_v31分支：路径中段0.018m，目标及TCP距目标≤0.05m用0.008m；目标果实沿用v3排除语义，不外扩。tuck的q2–q7固定为[-1,-.0061,2,-.181,.0534,-1.3933]，reach的q1取目标，carry取当前；顺序直线→tuck→high_ready→RRT。仅因目标在0.008m碰撞才更换IK，路径失败不会更换目标。carry仍直接rest，settle2、observe30、close10、open8及screw均不变。', '',
        'RRT使用同一原始点云：端点可行时以0.018m搜索；端点需0.008m时，原生绑定无法提供随状态变化的碰撞半径，按§11.4.4用0.008m搜索后仅对中段外扩验证。全部方法共用分区验证器，RRT输出密集插值复核；每次调用保留半径和首个碰撞记录。', '',
        '## 与v3对照','', '| 项目 | v3 | v3.1 | 目标 |','|---|---:|---:|---|',
        f"| ★规划确定性 | 100/100 | {sum(r['passed'] for r in planning['determinism']['cases'])}/100 | 100% |",
        f"| ★目标连续性 | 100/100 | {planning['target_continuity']['passed_instances']}/100 | ≥98% |",
        f"| ★方法一致率 | 97/100 | {planning['method_consistency']['passed_instances']}/100 | ≥95% |"]
    for task in ('count','occluded_ripe'):
        lines.append(f"| ★逐尝试成功 {task} | {ratio(old['by_task'][task]['attempt_success_rate'])} | {ratio(new['by_task'][task]['attempt_success_rate'])} | ≥90% |")
        lines.append(f"| ★规划失败 {task} | {ratio(failure_breakdown(old)[task]['categories'].get('planning_failure',fraction(0,old['by_task'][task]['success']+old['by_task'][task]['failed'])))} | {ratio(failures[task]['categories'].get('planning_failure',fraction(0,failures[task]['attempts'])))} | ≤5% |")
    for q in ('q5','q6','q7'):
        lines.append(f"| ★抓取{q}一致性 | {ratio(old['approach_end_wrist'][q]['within_median_plusminus_1rad'])} | {ratio(new['approach_end_wrist'][q]['within_median_plusminus_1rad'])} | ≥95% |")
    for key in ('q7_states_outside_pi','q7_commands_outside_pi'):
        lines.append(f"| ★{key} | {ratio(old[key])} | {ratio(new[key])} | 0 |")
    for phase in ('reach','carry'):
        lines.append(f"| {phase} q1多余行程P90 rad | {old['q1_extra_travel'][phase]['p90']} | {new['q1_extra_travel'][phase]['p90']} | 只报告 |")
    for name,stats in (('v3',old),('v31',new)):
        lines.append(f"| {name}回合长度中位/P10/P90 | {np.quantile(stats['episode_lengths'],[.5,.1,.9]).tolist() if stats['episode_lengths'] else None} | — | 只报告 |")
    lines+=['',f"全部数值源：v3_extended_stats.json、v31_stats.json、planning/summary.json。选择理由：{reason}。",'',
        '## 3×2失败位置分布','',location['v31']['definition'],'',
        '| 任务/格子 | v3目标尝试 | v3失败 | v3.1目标尝试 | v3.1失败 |','|---|---:|---:|---:|---:|']
    for task in ('count','occluded_ripe'):
        for cell in location['v3']['by_task'][task]:
            x=location['v3']['by_task'][task][cell];y=location['v31']['by_task'][task][cell]
            lines.append(f"| {task}/{cell} | {x['target_attempts']} | {x['failed_attempts']} | {y['target_attempts']} | {y['failed_attempts']} |")
    lines+=['','每格失败归于导致整回合失败的目标果实；未访问果实不虚增尝试数。失败原因、种子与初始xyz均见failure_coverage.json。换种子使两版成功布局不完全配对。', '',
        '## 运动与静止监督','',f"方法/RRT计数（所有尝试）：{ {name:value['methods'] for name,value in location.items()} }。",'',
        '| 阶段 | v3命中/监督样本 | v3.1命中/监督样本 |','|---|---:|---:|']
    for phase in ('observe','grasp','release','settle','done','other'):
        lines.append(f"| {phase} | {ratio(old['static_delayed_supervision']['by_phase'][phase])} | {ratio(new['static_delayed_supervision']['by_phase'][phase])} |")
    lines+=['', '此处审计示范的observe仍保留原监督标记以便对照；阶段3新生成的全部400条示范才将前30帧设为历史，不删除任何帧。', '',
        '## 碰撞诊断与24次失败重规划','',
        f"完整诊断：[report.md](v3_failure_diagnosis/report.md)，目标两余量碰撞数{audit['v3_collision_diagnosis']['targets_colliding_by_radius']}。原生RRT入口未发现把点云半径自动改回0.008m的代码。",'',
        f"v3.1从保存失败状态单次重规划成功{audit['v3_failures_replanned']['successes']}/24，逐条结果在v3_failure_replanning/summary.json。这不是24个完整物理回合的成功率。",'',
        '## 示范抽帧','']
    for video in videos: lines+=['',f"![示范]({Path(video['frames']).relative_to(out)})",f"源：{video['manifest']}；视频：{video['video']}。"]
    lines+=['','## 决策与下一步','',
        f"按§11.5选择{selection['solver_version']}，不停下等待确认，自动执行§11.6阶段3。仅两版均因代码错误等无法使用时才停止；不再迭代专家。",'',
        '阶段3数据、归一化统计、8卡布局是相对旧K0训练的三项允许差别。observe前30帧只作历史，评测A0+30步固定保持（计入1300预算），包络v2仅作不阻塞对照。']
    text='\n'.join(lines)+'\n';(out/'report.md').write_text(text)
    tracked=ECM/'reports/lowlevel_rescue_D2_v31_20261102';tracked.mkdir(exist_ok=True)
    atomic_json(tracked/'summary.json',audit)
    (ECM/'reports/lowlevel_rescue_D2_v31_20261102.md').write_text(text.replace('](videos/','](../results/'+out.name+'/videos/').replace(
        '](v3_failure_diagnosis/','](../results/'+out.name+'/v3_failure_diagnosis/'))
    root=read_json(ECM/'reports/lowlevel_rescue_status.json');root.update(phase=2,instruction_revision=3,state='phase3_expert_selected',
        selected_expert=selection,updated_utc=timestamp())
    root['links'].update(D2_v31_report=str(out/'report.md'),D2_v31_audit=str(out/'audit.json'),D2_v31_selection=str(out/'selection.json'))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)
    print(selection,flush=True)


if __name__=='__main__': main()
