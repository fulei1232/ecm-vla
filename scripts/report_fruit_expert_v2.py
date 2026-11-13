#!/usr/bin/env python3
"""D2 audit/report, including revision-1 static supervision, builder and D1 metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import h5py
import numpy as np

from fruit_expert_audit_stats import summarize_demonstrations
from v_a_common import ECM, atomic_json, read_json, sha256, timestamp


def failure_categories(stats):
    counts=Counter()
    for record in stats['failed_attempt_records']:
        for failure in record['failures']:
            reason=failure['reason']
            category=('no_plan' if 'no plan to' in reason else 'detach_failure' if 'did not detach' in reason else
                      'basket_miss' if 'missed the basket' in reason else 'other')
            counts[category]+=1
    attempts=sum(row['success']+row['failed'] for row in stats['by_task'].values())
    return {'counts':dict(counts),'all_attempts':attempts,'no_plan_rate':{
        'numerator':counts['no_plan'],'denominator':attempts,'fraction':counts['no_plan']/attempts if attempts else None}}


def trajectory(result_path):
    row=read_json(result_path)
    if not row.get('ok') or row.get('error'): return row,None,None
    with h5py.File(result_path.parent/row['h5'],'r') as f:
        ep=f[f"episode_{row['episode']}"]
        ids=sorted(int(k.split('_')[1]) for k in ep if k.startswith('timestep_'))
        q=np.stack([ep[f'timestep_{i}/obs/joint_state'][()] for i in ids])
        a=np.stack([ep[f'timestep_{i}/action/joint_action'][()] for i in ids])
    return row,q,a


def determinism(out):
    cases=[]
    for task in ('count','occluded_ripe'):
        for episode in range(5):
            paths=[out/'determinism'/task/f'ep{episode}'/f'repeat{j}'/'result.json' for j in range(3)]
            row={'task':task,'episode':episode,'results':[str(p) for p in paths]}
            if not all(p.exists() for p in paths):
                row.update(passed=False,reason='missing fixed-seed repetitions',max_difference_rad=None)
            else:
                runs=[trajectory(p) for p in paths]
                if any(q is None for _,q,_ in runs):
                    row.update(passed=False,reason='one or more fixed-seed expert failures',max_difference_rad=None,
                               failures=[r for r,q,a in runs if q is None])
                elif any(q.shape!=runs[0][1].shape or a.shape!=runs[0][2].shape for _,q,a in runs):
                    row.update(passed=False,reason='trajectory length mismatch',max_difference_rad=None,
                               lengths=[len(q) for _,q,_ in runs])
                else:
                    delta=max(max(float(np.max(np.abs(q-runs[0][1]))),float(np.max(np.abs(a[:,:7]-runs[0][2][:,:7]))))
                              for _,q,a in runs[1:])
                    row.update(passed=delta<1e-6,max_difference_rad=delta,
                               gripper_actions_identical=all(np.array_equal(a[:,7],runs[0][2][:,7]) for _,q,a in runs[1:]))
            cases.append(row)
    values=[r['max_difference_rad'] for r in cases if r['max_difference_rad'] is not None]
    return {'layouts':10,'repetitions_per_layout':3,'cases':cases,
            'max_difference_rad':max(values) if values else None,'all_passed':all(r['passed'] for r in cases)}


def illustration(manifest,out):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    from report_fruit_exec_guard import frames
    man=read_json(manifest)
    movie=out/f"{man['task']}_ep{man['episode']}.mp4"
    image=out/f"{man['task']}_ep{man['episode']}.png"
    out.mkdir(parents=True,exist_ok=True)
    with h5py.File(manifest.parent/man['h5'],'r') as f, imageio.get_writer(movie,fps=20) as writer:
        ep=f[f"episode_{man['episode']}"]
        n=int(ep['setup/n_timesteps'][()])
        for i in range(n):
            ts=ep[f'timestep_{i}']
            rgb=np.concatenate([ts['obs/front_rgb'][()],ts['obs/wrist_rgb'][()]],axis=1)
            canvas=Image.new('RGB',(512,286),'white');canvas.paste(Image.fromarray(rgb),(0,30))
            ImageDraw.Draw(canvas).text((5,5),f"{man['task']} ep{man['episode']} step {i}",fill='black')
            writer.append_data(np.asarray(canvas))
    frames(movie,image)
    return {'manifest':str(manifest),'video':str(movie),'frames':str(image)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    old=read_json(out/'legacy_stats.json') if (out/'legacy_stats.json').exists() else summarize_demonstrations(ECM/'data/fruit_picking/rm75b_eg2/train')
    for record in old['records']: assert sha256(Path(record['manifest']))==record['manifest_sha256']
    atomic_json(out/'legacy_stats.json',old)
    new=summarize_demonstrations(out/'demos/rm75b_eg2/train');atomic_json(out/'new_stats.json',new)
    old_seeds={(r['task'],r['episode']):r['seed'] for r in old['records']}
    changed_seeds=[{'task':r['task'],'episode':r['episode'],'old_seed':old_seeds.get((r['task'],r['episode'])),
                    'new_seed':r['seed']} for r in new['records'] if old_seeds.get((r['task'],r['episode']))!=r['seed']]
    repeat=determinism(out);atomic_json(out/'determinism.json',repeat)
    builder=read_json(out/'builder_assessment.json')
    d1=read_json(out/'d1_new_metrics/summary.json')
    assert d1['old_metrics_unchanged']
    gates={}
    for task,row in new['by_task'].items():
        rate=row['attempt_success_rate']
        gates[f'success_{task}']=bool(rate['denominator'] and rate['fraction']>=.95)
    gates['all_200_demos_present']=new['episodes']==200
    gates['determinism']=repeat['all_passed']
    for phase,row in new['motion'].items():
        gates[f'{phase}_q1_p90_below_1']=row['q1_swing_p90'] is not None and row['q1_swing_p90']<1.
        gates[f'{phase}_q7_swing_over_pi_zero']=row['q7_swing_over_pi']['denominator']>0 and row['q7_swing_over_pi']['numerator']==0
    grasp=new['approach_end_q7']['deviation_over_1rad']
    gates['grasp_q7_consistent']=grasp['denominator']>0 and grasp['numerator']==0
    wrist=new['q7_states_outside_pi']
    gates['all_q7_states_within_pi']=wrist['denominator']>0 and wrist['numerator']==0
    expected={'observe':30,'grasp':10,'release':8,'settle':12,'done':5}
    gates['original_hold_lengths_unchanged']=all(new['hold_segment_lengths'][phase] and
        all(n==length for n in new['hold_segment_lengths'][phase]) for phase,length in expected.items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    lengths=old['episode_lengths']+new['episode_lengths']
    bins=np.histogram_bin_edges(lengths,bins=20)
    ax.hist(old['episode_lengths'],bins=bins,alpha=.5,label='legacy')
    ax.hist(new['episode_lengths'],bins=bins,alpha=.5,label='nearest v2')
    ax.set(xlabel='Episode length (control steps)',ylabel='Episodes');ax.legend();fig.tight_layout()
    fig.savefig(out/'episode_lengths.png',dpi=160);plt.close(fig)
    length_summary={name:{'episodes':len(rows),'median':float(np.median(rows)) if rows else None,
                          'p10_p90':np.quantile(rows,[.1,.9]).tolist() if rows else None}
                    for name,rows in [('legacy',old['episode_lengths']),('nearest',new['episode_lengths'])]}
    videos=[]
    for task in ('count','occluded_ripe'):
        manifests=sorted((out/'demos/rm75b_eg2/train'/task).glob('ep*_manifest.json'),key=lambda p:int(p.name.split('_')[0][2:]))[:2]
        videos.extend(illustration(path,out/'videos') for path in manifests)
    audit={'phase':2,'revision':1,'generated_utc':timestamp(),'platform':{'condition':'A0','joint_guard':'off','exec_horizon':16},
           'guard_v1_deprecated':True,'gates':gates,'all_gates_passed':all(gates.values()),
           'expert_success_definition':'successful attempts/all attempts, including rejected layout retries; accepted-episode and first-attempt rates also reported',
           'legacy':old,'nearest':new,'determinism':repeat,'episode_length_comparison':length_summary,
           'expert_failure_categories':{'legacy':failure_categories(old),'nearest':failure_categories(new)},
           'layout_seed_changes_due_to_retry':changed_seeds,
           'length_gate_note':'Original task gives no numeric episode-length margin; histogram and median/p10-p90 require qualitative review before phase3.',
           'builder_assessment':builder,'d1_recomputed_with_revision1_metrics':d1,'videos':videos,
           'planning_note':'mplib 0.1.1 has no public native RRT seed setter; IK uses deterministic local sampling, straight paths have priority; actual determinism is audited, never assumed',
           'sources':{'plan':str(out/'plan.json'),'new_stats':str(out/'new_stats.json'),'legacy_stats':str(out/'legacy_stats.json'),
                      'determinism':str(out/'determinism.json'),'builder_assessment':str(out/'builder_assessment.json'),
                      'd1_new_metrics':str(out/'d1_new_metrics/summary.json')}}
    atomic_json(out/'audit.json',audit);atomic_json(out/'summary.json',audit)
    def ratio(x): return f"{x['numerator']}/{x['denominator']}"
    lines=[f"阶段 2：{'数值审计通过，待完整复核' if audit['all_gates_passed'] else '审计不达标，停下等用户确认'}。",
           f"已确认：新专家生成 {new['episodes']}/200 条成功示范，10 个布局×3 次确定性审计结果见下表。",
           "已确认：平台为 A0（不开包络、执行16步）；v1作废；专家保持 observe/grasp/release/settle/done 步数不变。",
           "已完成 §9.3 静止监督、动作目标相对状态分位数及构建器评估，§9.5 已补算 D1 五个条件。",
           "决策：任何审计项不通过不得进入阶段3；未改训练构建器，未训练，未使用新门槛集或封存验证集。",'',
           '## 做了什么','',
           '新增 opt-in ik_mode=nearest，默认仍为 legacy。调用本机 mplib.IK，以局部确定性采样代替原生随机配置提供器；将 q7 选为距当前最近且在限位内的2π等价值，拒绝 |q7|>π，再按加权关节距离排序。权重为 [64,24,4,2,1,1,1]，基座/肩部较高；按顺序每次只向一个目标规划，优先碰撞检查通过的关节直线并使用原TOPP时间参数化，失败才退回RRT。每段记录 straight/rrt/screw。approach/pull 的 plan_screw 和所有保持步数不变。', '',
           '所有新示范位于本结果目录 demos，固定布局重复位于 determinism；原数据、原D1结果、封存划分不覆盖。随机源与限制见 audit.json/planning_note。基础关节权重通过开发中的CPU运动学探针确定，未使用开发门槛集或验证集。', '',
           '## 审计数值与目标对照','', '| 审计项 | 旧专家 | 新专家 | 目标/判定 |','|---|---:|---:|---|']
    for task in ('count','occluded_ripe'):
        lines.append(f"| 专家成功率 {task}（含重试失败） | {ratio(old['by_task'][task]['attempt_success_rate'])} | {ratio(new['by_task'][task]['attempt_success_rate'])} | ≥95%；{gates['success_'+task]} |")
        lines.append(f"| 已接受 episode {task} | {ratio(old['by_task'][task]['accepted_episode_rate'])} | {ratio(new['by_task'][task]['accepted_episode_rate'])} | 不替代逐尝试成功率 |")
    lines.append(f"| 10布局×3次轨迹最大差异 rad | 未审计 | {repeat['max_difference_rad']} | <1e-6；{repeat['all_passed']} |")
    for phase in ('reach','carry'):
        lines.append(f"| {phase} q1摆幅P90 rad | {old['motion'][phase]['q1_swing_p90']} | {new['motion'][phase]['q1_swing_p90']} | <1.0；{gates[phase+'_q1_p90_below_1']} |")
        lines.append(f"| {phase} q7摆幅>π | {ratio(old['motion'][phase]['q7_swing_over_pi'])} | {ratio(new['motion'][phase]['q7_swing_over_pi'])} | 0 |")
    lines.append(f"| approach末q7偏离中位数>1rad | {ratio(old['approach_end_q7']['deviation_over_1rad'])} | {ratio(grasp)} | 0 |")
    lines.append(f"| q7状态超出[−π,π] | {ratio(old['q7_states_outside_pi'])} | {ratio(wrist)} | 0 |")
    lines += ['',f"新 q7 中心（approach末中位数）：{new['approach_end_q7']['median']} rad。所有数值来自 legacy_stats.json/new_stats.json/determinism.json；逐示范及失败尝试来源在 audit.json 中列出。回合长度比较：{json.dumps(length_summary,ensure_ascii=False)}。",'',
              '![回合长度直方图](episode_lengths.png)','',
              f"因专家失败后仍按旧规则换种子，成功布局相对旧示范有 {len(changed_seeds)}/{len(new['records'])} 条种子不同；列表见 audit.json。长度及轨迹分布为各自成功示范的描述对照，不假定布局完全配对。",'',
              f"规划方法计数：{json.dumps(new['planning_methods'])}；保持段审计：{gates['original_hold_lengths_unchanged']}。成功率按每个实际尝试计，失败后换种子规则保留16次上限；原 manifest 的失败尝试也纳入旧专家分母。",'',
              f"失败原因分类及 no-plan 规划失败比例（含分母）见 audit.json/expert_failure_categories：{json.dumps(audit['expert_failure_categories'],ensure_ascii=False)}。",'',
              '## 静止监督样本（修订1 §9.3）','',
              '动作块取前16步，尾部按官方方式重复末动作；以该样本的实测 joint_state 为位移参照，前8步手臂最大位移<0.01rad且后8步≥0.05rad。按原始 is_video_demo=False 的监督锚点统计。官方构建器实际保存20步动作块，本项按任务书显式只诊断前16步，未改构建器。', '',
              '| 阶段 | 旧示范 命中/监督样本 | 新示范 命中/监督样本 |','|---|---:|---:|']
    for phase in ('observe','grasp','release','settle','done','other'):
        lines.append(f"| {phase} | {ratio(old['static_delayed_supervision']['by_phase'][phase])} | {ratio(new['static_delayed_supervision']['by_phase'][phase])} |")
    lines.append(f"| 合计 | {ratio(old['static_delayed_supervision']['all'])} | {ratio(new['static_delayed_supervision']['all'])} |")
    lines += ['', '## 安全包络v2所需统计（仅提供统计，未实施包络）','',
              '来源 new_stats.json：状态分位为[0,0.5%,1%,50%,99%,99.5%,100%]，完整各关节数值在 audit.json；下表列0.5%/99.5%以及目标相对实测状态偏差。', '',
              '| 关节 | 状态P0.5 | 状态P99.5 | 目标−状态绝对差 P99 | 目标−状态绝对差 P99.9 |','|---|---:|---:|---:|---:|']
    if new['state_samples']:
        quant=new['joint_state_quantiles']['values'];offset=new['action_minus_state_abs_quantiles']['values']
        for j in range(7): lines.append(f"| {j+1} | {quant[1][j]:.9f} | {quant[5][j]:.9f} | {offset[0][j]:.9f} | {offset[1][j]:.9f} |")
    lines += ['', '## 构建器能力评估（先不实施）','',
              f"原生不支持中途非监督帧。官方 build_robomme_dataset.py:{builder['source_assertion_line']} 强制 is_video_demo == (step_idx < exec_start_idx)；ECM 计划器也按完整执行后缀计算样本数。使用真实官方函数/断言和合成HDF5验证：30帧历史前缀通过，中途40–44帧标历史会触发断言。文件哈希和案例见 builder_assessment.json。",'',
              '阶段3按 §9.4 只将开头observe30帧作历史，settle/done暂保持监督。若后续获准扩展，需要独立supervision mask、保留每帧特征和原step_idx、重排连续PKL样本编号并同步改计划/验收/加载检查；此处只是方案，没有实施，也没有删除帧。', '',
              '## D1 两个补充指标（§9.5）','',
              '来源 d1_new_metrics/<条件>/{summary.json,episodes.jsonl}；原 D1 所有旧指标和结果不变。离家为TCP距第0帧位置严格>5cm。每次入篮以各果实 in_basket 的 false→true 转换检测，同一步多个果实入篮记一次事件并列全部果实编号；每个事件独立计时，未恢复的事件右删失单列，未用0填补。中位数仅在观察到离家/恢复的样本中计算，未观察者计数必须一起看。', '',
              '| 条件/任务 | leave_home中位步 | 未离开/回合 | 入篮后恢复中位步 | 结束未恢复/入篮事件 |','|---|---:|---:|---:|---:|']
    for c,summary in d1['conditions'].items():
        for label,row in [('合计',summary['all']),*summary['by_task'].items()]:
            lines.append(f"| {c}/{label} | {row['leave_home_step']['median']} | {ratio(row['leave_home_step']['never_left_home'])} | {row['post_release_resume_steps']['median']} | {ratio(row['post_release_resume_steps']['unresumed_at_end'])} |")
    lines += ['', '## 新专家视频抽帧','']
    for video in videos:
        lines += [f"来源 `{video['manifest']}`；视频 `{video['video']}`。",'',f"![专家示范]({Path(video['frames']).relative_to(out)})",'']
    lines += ['## 已确认、推断、决策与下一步','',
              f"已确认的审计布尔结果：{json.dumps(gates,ensure_ascii=False)}。所有失败尝试及重复失败均保留于 audit.json 和各日志；缺失布局不以成功布局替代。",'',
              '静止监督与闭环等待机制的因果关系仍是推断；此阶段只给出新旧统计，不提前把阶段3的数据改动效果当作已证实。RRT确定性也以重复轨迹实测为准。', '',
              ('数值审计全部通过；完成回合长度/视频与来源的独立复核后，按任务书进入阶段3并先汇报D2。' if audit['all_gates_passed'] else '存在审计不达标项，按 §4.2 停下等用户确认，不进入阶段3。')]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print({'all_gates_passed':audit['all_gates_passed'],'gates':gates},flush=True)


if __name__=='__main__': main()
