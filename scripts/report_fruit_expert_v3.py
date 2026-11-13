#!/usr/bin/env python3
"""Revision-2 D2 delivery: exactly five core gates and three-way comparisons."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from fruit_expert_audit_stats import fraction,load_episode,summarize_demonstrations
from report_fruit_expert_v2 import illustration
from v_a_common import ECM,atomic_json,read_json,sha256,timestamp

V2=ECM/'results/fruit_expert_v2_audit_20261102'


def add_wrist_statistics(stats):
    wrist=[];commands=[];all_rows=[]
    for record in stats['records']:
        path=Path(record['manifest']);assert sha256(path)==record['manifest_sha256']
        man,q,a,supervised,gt=load_episode(path)
        for key in ('h5','gt'): assert sha256(path.parent/man[key])==man[key+'_sha256']
        commands.append(a[:,6])
        for i,r in enumerate(gt):
            if r['phase']=='approach' and (i==len(gt)-1 or gt[i+1]['phase']!='approach'):
                wrist.append(q[i,4:7])
        all_rows.append(man)
    values=np.asarray(wrist,dtype=np.float64).reshape(-1,3)
    medians=np.median(values,axis=0) if len(values) else np.full(3,np.nan)
    stats['approach_end_wrist']={f'q{j+5}':{'median':float(medians[j]) if len(values) else None,
        'within_median_plusminus_1rad':fraction(np.sum(np.abs(values[:,j]-medians[j])<=1),len(values)),
        'values':values[:,j].tolist()} for j in range(3)}
    positive=values[values[:,1]>0,2]
    stats['approach_end_q7_q6_positive_iqr']=np.quantile(positive,[.25,.75]).tolist() if len(positive) else None
    a=np.concatenate(commands) if commands else np.empty(0)
    stats['q7_commands_outside_pi']=fraction(np.sum(np.abs(a)>np.pi),len(a))
    stats['q7_command_range']=[float(a.min()),float(a.max())] if len(a) else None
    stats['reach_carry_methods_accepted']=dict(Counter(t['method'] for man in all_rows for t in man.get('planning_audit',[])
        if t.get('phase') in ('reach','carry') and t.get('status')=='Success'))
    return stats


def failure_breakdown(stats):
    counts={task:Counter() for task in ('count','occluded_ripe')}
    for record in stats['failed_attempt_records']:
        task=Path(record['manifest']).parent.name
        for failure in record['failures']:
            reason=failure['reason']
            label=('detach_failure' if 'did not detach' in reason else 'basket_miss' if 'missed the basket' in reason else
                   'planning_failure' if 'no plan to' in reason or 'q7 command outside' in reason else 'other')
            counts[task][label]+=1
    return {task:{'attempts':stats['by_task'][task]['success']+stats['by_task'][task]['failed'],
                  'categories':{key:fraction(value,stats['by_task'][task]['success']+stats['by_task'][task]['failed'])
                                for key,value in counter.items()}} for task,counter in counts.items()}


def basket_miss_exceeds_three_percent(stats):
    rows=failure_breakdown(stats)
    total=sum(row['attempts'] for row in rows.values())
    failures=sum(row['categories'].get('basket_miss',{}).get('numerator',0) for row in rows.values())
    return bool((total and failures/total>.03) or any(
        row['categories'].get('basket_miss',{}).get('fraction',0)>.03 for row in rows.values()))


def core_gates(stats,planning):
    complete=stats['episodes']==200
    return {'planning_determinism':bool(complete and planning['determinism']['passed']),
        'target_continuity':bool(complete and planning['continuity']['passed']),
        'grasp_wrist_consistency':bool(complete and all(
            stats['approach_end_wrist'][q]['within_median_plusminus_1rad']['fraction'] is not None and
            stats['approach_end_wrist'][q]['within_median_plusminus_1rad']['fraction']>=.95 for q in ('q5','q6','q7'))),
        'reach_carry_q1_swing':bool(complete and all(stats['motion'][p]['q1_swing_p90'] is not None and
            stats['motion'][p]['q1_swing_p90']<1. for p in ('reach','carry'))),
        'q7_command_and_state_range':bool(complete and stats['q7_commands_outside_pi']['denominator']>0 and
            stats['q7_commands_outside_pi']['numerator']==0 and stats['q7_states_outside_pi']['numerator']==0)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();selection=read_json(out/'selection.json')
    versions={}
    for name,file in (('legacy','legacy_stats.json'),('nearest_v2','new_stats.json')):
        cache=out/f'{name}_extended_stats.json'
        versions[name]=read_json(cache) if cache.exists() else add_wrist_statistics(read_json(V2/file))
        for record in versions[name]['records']: assert sha256(Path(record['manifest']))==record['manifest_sha256']
        atomic_json(cache,versions[name])
    canonical=add_wrist_statistics(summarize_demonstrations(Path(selection['data_root'])))
    versions['canonical_v3']=canonical;atomic_json(out/'canonical_stats.json',canonical)
    planning=read_json(out/'planning/summary.json');physics=read_json(out/'physics/summary.json')
    gates=core_gates(canonical,planning);all_passed=all(gates.values())
    failures={name:failure_breakdown(stats) for name,stats in versions.items()}
    methods=canonical['reach_carry_methods_accepted'];n_methods=sum(methods.values())
    noncore={'rrt_rate':fraction(methods.get('rrt',0),n_methods),
        'rrt_rate_at_most_2percent':bool(n_methods and methods.get('rrt',0)/n_methods<=.02),
        'by_task':{t:{'attempt_success_at_least_90percent':canonical['by_task'][t]['attempt_success_rate']['fraction'] is not None and
            canonical['by_task'][t]['attempt_success_rate']['fraction']>=.9,
            'each_failure_category_at_most_3percent':all(row['fraction']<=.03 for row in failures['canonical_v3'][t]['categories'].values())}
            for t in ('count','occluded_ripe')}}
    # Include rejected attempts, without silently substituting their rates for accepted-demo gates.
    all_attempts=[read_json(path) for path in Path(selection['data_root']).glob('*/attempts/ep*_seed*.json')]
    all_q7={key:sum(row.get('q7_all_attempt_steps',{}).get(key,0) for row in all_attempts)
            for key in ('samples','state_outside_pi','command_outside_pi')}
    all_attempt_methods=Counter(trace.get('method') for row in all_attempts for trace in row.get('planning_audit',[])
        if trace.get('phase') in ('reach','carry'))
    noncore['all_attempt_reach_carry_methods']=dict(all_attempt_methods)
    noncore['all_attempt_q7']=all_q7
    noncore['rrt_rate_accepted_demos']=noncore['rrt_rate']
    noncore['rrt_rate']=fraction(all_attempt_methods.get('rrt',0),sum(all_attempt_methods.values()))
    noncore['rrt_rate_at_most_2percent']=bool(noncore['rrt_rate']['denominator'] and noncore['rrt_rate']['fraction']<=.02)
    videos=[]
    for task in ('count','occluded_ripe'):
        for e in (0,1):
            path=Path(selection['data_root'])/task/f'ep{e}_manifest.json'
            if path.exists(): videos.append(illustration(path,out/'videos'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));lengths=[n for stats in versions.values() for n in stats['episode_lengths']]
    bins=np.histogram_bin_edges(lengths,bins=22)
    for name,stats in versions.items(): ax.hist(stats['episode_lengths'],bins=bins,alpha=.4,label=name)
    ax.set(xlabel='Episode length (control steps)',ylabel='Episodes');ax.legend();fig.tight_layout()
    fig.savefig(out/'episode_lengths.png',dpi=160);plt.close(fig)
    length_summary={name:{'median':float(np.median(stats['episode_lengths'])) if stats['episode_lengths'] else None,
        'p10_p90':np.quantile(stats['episode_lengths'],[.1,.9]).tolist() if stats['episode_lengths'] else None} for name,stats in versions.items()}
    audit={'phase':2,'revision':2,'generated_utc':timestamp(),'solver_version':'canonical_v3_20261102',
        'platform':{'condition':'A0','joint_guard':'off','exec_horizon':16},'core_gates':gates,
        'all_core_gates_passed':all_passed,'audit_scale_complete':canonical['episodes']==200,
        'noncore':noncore,'comparison':versions,'failure_categories':failures,'planning_audit':planning,
        'physics_report_only':physics,'selection':selection,'episode_length_comparison':length_summary,'videos':videos,
        'D1_revision1_metrics':read_json(V2/'d1_new_metrics/summary.json'),
        'rollout_metrics_note':'Phase2 is expert-only. Section5 policy rollout metrics are not newly measured; original D1 five-condition metrics and revision1 additions are attached unchanged.',
        'source_files':{'plan':str(out/'plan.json'),'stats':str(out/'canonical_stats.json'),
                        'planning':str(out/'planning/summary.json'),'physics':str(out/'physics/summary.json')}}
    atomic_json(out/'audit.json',audit);atomic_json(out/'summary.json',audit)
    def ratio(row): return f"{row['numerator']}/{row['denominator']} ({row['fraction']:.2%})" if row['fraction'] is not None else '无样本'
    lines=[f"阶段2专家v3：{'五个核心项全部达标，自动进入阶段3' if all_passed else '核心审计不达标，停止并等待审阅'}。",
        f"完成 {canonical['episodes']}/200 条 train0–99 双任务示范；平台为 A0，v2 代码快照和所有原产物保留。",
        f"核心项结果：{gates}。",
        f"采用 settle={selection['settle_steps']}；规划确定性与固定动作物理回放分开审计，物理分叉仅报告。",
        "决策依据：§10.4–10.5；非核心项不阻塞，核心项失败不继续自迭代。",'',
        '## 做了什么','',
        '新增独立 canonical 分支。carry 直接取 rest 手臂关节角；reach 先用 rest 初值执行原生 CLIK，失败才用确定性随机初值后备，等权按距 rest 排序，q7 映射到 rest 附近的等价值。目标在检查路径之前确定，不因路径碰撞改换 IK 分支。路径点云半径为0.018m（原0.008m+0.01m），依次直线、经固定rest、经固定high_ready，最后才RRT。固定high_ready数值与全部路径使用记录见代码和规划trace。', '',
        'approach/pull仍用原plan_screw和0.008m点云，保持observe30、close10、open8、done5。canonical settle默认2；如任一任务或合计落篮失败占尝试>3%，只按指令恢复12并在独立目录重生成，短settle的完整结果保留。下一颗reach规划完成后、执行轨迹前检查前一颗；最后一颗在done后检查。规划本身不推进仿真时钟，因此该延迟并不凭空增加物理等待时间。', '',
        '## 三方对照','', '| 项目 | 旧专家 | v2 nearest | v3 canonical | 目标 |','|---|---:|---:|---:|---|']
    for t in ('count','occluded_ripe'):
        lines.append('| 逐尝试成功率 '+t+' | '+' | '.join(ratio(s['by_task'][t]['attempt_success_rate']) for s in versions.values())+' | ≥90%，非核心 |')
    for q in ('q5','q6','q7'):
        lines.append('| ★ 抓取末'+q+'在自身中位数±1rad | '+' | '.join(ratio(s['approach_end_wrist'][q]['within_median_plusminus_1rad']) for s in versions.values())+' | 各≥95% |')
    for phase in ('reach','carry'):
        lines.append('| ★ '+phase+'的q1摆幅P90 rad | '+' | '.join(str(s['motion'][phase]['q1_swing_p90']) for s in versions.values())+' | <1.0 |')
    for key,title in (('q7_state_range','实测范围'),('q7_command_range','指令范围')):
        lines.append('| ★ q7'+title+' rad | '+' | '.join(str(s[key]) for s in versions.values())+' | 全在[−π,π] |')
    for key,title in (('q7_states_outside_pi','实测越界'),('q7_commands_outside_pi','指令越界')):
        lines.append('| q7'+title+' | '+' | '.join(ratio(s[key]) for s in versions.values())+' | 0 |')
    lines += [f"| ★ 同一保存状态的规划重复 | 未按新口径测 | 未按新口径测（D2物理+重规划7/10通过） | {sum(r['passed'] for r in planning['determinism']['cases'])}/100 | reach50+carry50，各3次完全一致 |",
        f"| ★ 目标连续性 | 未按新口径测 | 未按新口径测 | {planning['continuity']['passed_instances']}/100 | ≥98%，目标最大差<.05rad且方法不变 |",
        f"| RRT比例 | 未完整记录 | 见来源中的reach/carry统计 | {ratio(noncore['rrt_rate'])} | ≤2%，非核心 |",'',
        f"所有分布来自各自成功示范，失败换种子导致布局不完全配对。旧/v2来源为 {V2}/legacy_stats.json、new_stats.json 及其中 manifest；新增手腕三关节和指令范围均从原 HDF5 补算，未覆盖旧文件。v3来源 canonical_stats.json。逐尝试各类失败占比：`audit.json/failure_categories`。完整失败动作、GT及保存规划输入在选中数据根目录 attempts、hdf5_files、gt、planning_states。",'',
        f"非核心项：{noncore}。RRT主比例的分母包含失败尝试中所有reach/carry规划调用，另列被接受示范的比例。失败原因三方对照：{failures}。",'',
        f"settle选择：{selection}。该记录给出settle2逐任务失败类别和实际分母；触发时只恢复为12一次，两套数据均保留。",'',
        '## 规划确定性与连续性','',
        '按 episode 顺序交错两个任务抽取成功示范中的 reach/carry 实例，保留输入状态、果实目标、点云、附着体、速度/加速度限值和源文件哈希。每个确定性实例从同一保存输入重复3次，比对全部关节路径、时间、速度、加速度和规划trace，要求逐位一致。连续性以原始规划为基准，五次分别在果实中心施加确定的±2mm坐标扰动、测量关节施加±1e−3rad，重新计算接近方向和预抓取位姿；以六次目标的逐关节最大两两差和方法是否一致判定。完整每例记录在planning目录。原生RRT无公共设种子API，实际调用一律保留，不假定它确定。', '',
        '## D2三个布局的物理首次分叉','',
        '| 原D2布局 | 重复对 | 首次动作不同（0起始索引/阶段/幅度rad） | 首次已记录状态不同（已执行步数/字段/幅度） |',
        '|---|---|---|---|']
    for case in physics['original_d2_replanning_runs']:
        for pair in case['pairs']:
            state=pair['first_state_difference_exact']
            detail='已记录字段全程一致' if state is None else f"{state['after_control_steps']} / {state['differences']}"
            lines.append(f"| ep{case['episode']} | {pair['pair']} | {pair['first_action_difference_timestep']} / {pair.get('first_action_phase')} / {pair['first_action_difference_rad']} | {detail} |")
    lines += ['',
        '原D2重复中的首次状态差和首次动作差见physics/original_d2_divergence.json。对occluded_ripe的ep0/1/3，各固定采用D2 repeat0的同一动作数组，在三个独立进程重放，全程不调用规划器。记录机器人qpos/qvel/TCP、果实位姿/速度/附着/入篮、叶片位姿/push。首次逐位不同与首次>1e−6字段单位的不同分别报告；状态索引k表示已执行k步，避免把执行前记录误作第k个动作的结果。', '',
        '| 布局 | 重放对 | 首次逐位分叉：已执行步数/对象/幅度 | 首次>1e−6 |','|---|---|---|---|']
    for case in physics['fixed_action_replay']:
        if not case['complete']:
            lines.append(f"| ep{case['episode']} | 缺失 | {case['missing']} | 未完成 |")
        else:
            for pair in case['pairs']:
                def describe_first(row):
                    return '全程完全一致' if row is None else f"{row['after_control_steps']} / "+str(row['differences'])
                lines.append(f"| ep{case['episode']} | {pair['pair']} | {describe_first(pair['first_exact'])} | {describe_first(pair['first_above_1e_6'])} |")
    lines += ['', '原D2没有保存关节速度、果实朝向/速度和叶片位姿，已记录字段一致不能证明完整物理状态一致。固定动作若仍发生分叉，才是物理回放微差的直接证据；与原D2规划切换之间的因果关系另需结合首次动作分叉步数解释。没有观测到的叶片或果实分叉不作推断。', '',
        '## 静止监督与回合长度','', '| 阶段 | 旧专家 | v2 | v3 |','|---|---:|---:|---:|']
    for phase in ('observe','grasp','release','settle','done','other'):
        lines.append('| '+phase+' | '+' | '.join(ratio(s['static_delayed_supervision']['by_phase'][phase]) for s in versions.values())+' |')
    lines += ['', '静止监督定义沿用修订1：相对当前实测状态，前8动作最大关节位移<.01rad、后8≥.05rad，尾部重复末动作，仅统计原监督锚点。observe仍在D2原始记录中保留监督标记；仅阶段3构建时将前30帧设为历史。settle缩短不删帧，done仍保留监督。', '',
        f"长度中位数/P10/P90：{length_summary}。保持段实际长度见canonical_stats.json/hold_segment_lengths。",'',
        '![三方回合长度](episode_lengths.png)','', '## 4段示范抽帧','']
    for video in videos: lines.extend([f"视频 `{video['video']}`；源 `{video['manifest']}`。",'',f"![示范]({Path(video['frames']).relative_to(out)})",''])
    lines += ['## 决策与下一步','',
        '五个核心项全部通过，按§10.5自动进入阶段3，执行§9.4：observe只作历史、前30步固定保持计入1300步预算；settle在专家端处理，不实施中途非监督帧。' if all_passed else
        '存在核心项失败，按§10.1/10.5停下等用户审阅，不再自行迭代v3，不生成阶段3数据或训练。', '',
        '本阶段无新策略闭环指标；§5完整旧D1指标及§9.5补算已原样附入summary.json/D1_revision1_metrics，不能把专家结果当成K0改善。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    tracked=ECM/'reports/lowlevel_rescue_D2_v3_20261102';tracked.mkdir(exist_ok=True)
    atomic_json(tracked/'summary.json',audit)
    prefix='../results/'+out.name+'/'
    text=(out/'report.md').read_text().replace('](episode_lengths.png)',']('+prefix+'episode_lengths.png)').replace('](videos/',']('+prefix+'videos/')
    (ECM/'reports/lowlevel_rescue_D2_v3_20261102.md').write_text(text)
    print({'all_core_gates_passed':all_passed,'core_gates':gates},flush=True)


if __name__=='__main__': main()
