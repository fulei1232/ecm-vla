#!/usr/bin/env python3
"""Independently validate completed G1 logs and deliver the full D3 report."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

from analyze_fruit_failure_modes import analyze_episode,aggregate
from v_a_common import ECM,atomic_json,read_json,read_jsonl,sha256,timestamp

OUT=ECM/'results/fruit_k0v2_gate_20261102'
RUN=ECM/'runs/fruit_k0v2_lowlevel_rescue_20261102'


def ratio(row):
    n,d=row['numerator'],row['denominator']
    return f'{n}/{d} ({n/d:.2%})' if d else '不适用/无观测 (0/0)'


def main():
    summary=read_json(OUT/'summary.json');state=read_json(OUT/'status.json')
    assert state['state']=='awaiting_G1_confirmation' and summary['phase']==3
    plan=read_json(OUT/'G1_continuation_plan.json');assert read_json(OUT/'G1_started.json')==plan
    assert plan['episodes']==list(range(220,260)) and plan['model_seed']==7 and plan['total_budget']==1300
    for path,digest in plan['source_sha256'].items(): assert sha256(Path(path))==digest,path
    sealed={RUN/'protocol.json':plan['training_protocol_sha256'],
        Path(plan['checkpoint'])/'params/manifest.ocdbt':plan['checkpoint_manifest_sha256'],
        OUT/'guard_choice.json':plan['guard_choice_sha256'],OUT/'guard_v2.json':plan['guard_stats_sha256'],
        OUT/'expert_profile.json':plan['expert_profile_sha256'],
        ECM/'reports/split_registry_20261102.json':plan['registry_sha256'],
        ECM/'results/fruit_dart_kf_eval_20261027/protocol.json':plan['serving_protocol_sha256']}
    for path,digest in sealed.items(): assert sha256(path)==digest,path
    from run_finetune_kf import verify
    verify(RUN);t1=read_json(RUN/'gate_t1_k0.json');assert t1['passed']
    expert=read_json(OUT/'expert_profile.json');center=expert['q7_center']
    rows=[];starts=[];end_ns=[];records=[];audit_decisions=0;action_steps=0;max_steps=0
    actual=list((OUT/'G1').glob('*/ep*/result.json'))
    assert len(actual)==80
    for task in ('FruitCount','FruitOccludedRipe'):
        for e in range(220,260):
            directory=OUT/f'G1/{task}/ep{e}';result=read_json(directory/'result.json')
            assert result['task']==task and result['episode_id']==e and result['model_seed']==7
            assert result['joint_guard']=='v2' and result['observe_hold_steps']==30 and result['total_step_budget']==1300
            started=read_json(directory/'started.json')
            assert started['plan_sha256']==sha256(OUT/'G1_continuation_plan.json')
            starts.append(started['utc'])
            row=analyze_episode(directory,center);row.update(task=task,episode_id=e,model_seed=None)
            rows.append(row)
            evidence=read_json(directory/'memory_audit.json');audit_decisions+=evidence['decisions']
            controls=read_jsonl(directory/'control_actions.jsonl');raw=read_jsonl(directory/'raw_frames.jsonl')
            assert len(controls)==result['executed_steps']<=1300
            assert len(raw)==len(controls)+1
            assert all(r['control_step']==i and r['frame_id']==i for i,r in enumerate(raw))
            assert all(r['is_demo']==(i<30) and r['exec_start_idx']==30 for i,r in enumerate(raw))
            action_steps+=len(controls);max_steps=max(max_steps,len(controls));end_ns.append(raw[-1]['timestamp_ns'])
            records.append({'task':task,'episode':e,'outcome':result['outcome'],'steps':len(controls),
                'memory_audit_passed':evidence['passed'],'client_error':result.get('error',''),
                'inputs_sha256':{name:sha256(directory/name) for name in ('result.json','started.json',
                    'control_actions.jsonl','raw_frames.jsonl','event_signal.jsonl','memory_audit.json')}})
    recalculated=aggregate(rows);assert recalculated==summary['G1']['all']
    by_task={task:aggregate([row for row in rows if row['task']==task]) for task in ('FruitCount','FruitOccludedRipe')}
    assert by_task==summary['G1']['by_task']
    gates={'overall_success':recalculated['success']['fraction']>=.25,
        'each_task_success':all(row['success']['fraction']>=.15 for row in by_task.values()),
        'first_target_in_basket':recalculated['first_target_in_basket']['fraction']>=.5,
        'wrist_deviation':recalculated['wrist_deviation_executed']['fraction']<=.05,
        'engineering_errors':recalculated['engineering_error']['numerator']==0}
    assert gates==summary['G1_gates'] and all(gates.values())==summary['G1_passed']
    choice=read_json(OUT/'guard_choice.json')
    assert min(starts)>choice['fixed_before_G1_utc']
    assert not list((OUT/'superseded').glob('*/G1/*/ep*/result.json'))
    norm=Path(plan['checkpoint'])/'assets/robomme/norm_stats.json'
    dataset=ECM/'data/finetune_kf/fruit_dart_v2_rm75b_train'
    assert sha256(norm)==sha256(dataset/'assets/robomme/norm_stats.json')
    checkpoint_steps=sorted(int(p.name) for p in (RUN/'ckpt/mme_vla_suite/k0').iterdir() if p.is_dir() and p.name.isdigit())
    assert checkpoint_steps==[11999]
    validation={'validated_utc':timestamp(),'expected_unique_layouts':80,'recorded_results':len(actual),
        'model_seed':7,'first_episode_started_utc':min(starts),
        'last_physical_frame_utc':datetime.fromtimestamp(max(end_ns)/1e9,timezone.utc).isoformat(),
        'guard_frozen_before_all_layouts':True,'all_frozen_source_and_artifact_hashes_verified':True,
        'T1_passed':True,'final_checkpoint_only':checkpoint_steps,'checkpoint_norm_stats_match_dataset':True,
        'total_control_steps':action_steps,'max_episode_control_steps':max_steps,'all_physical_prefix_clocks_verified':True,
        'memory_audits_passed':sum(r['memory_audit_passed'] for r in records),'audited_decisions':audit_decisions,
        'client_engineering_errors':sum(bool(r['client_error']) for r in records),
        'independent_metrics_match':True,'G1_gates_recomputed':gates,'records':records,
        'no_completed_layout_reruns_observed':True}
    atomic_json(OUT/'independent_validation.json',validation)
    if not (OUT/'summary_before_independent_review.json').exists():
        atomic_json(OUT/'summary_before_independent_review.json',summary)
        (OUT/'report_before_independent_review.md').write_text((OUT/'report.md').read_text())
    summary['independent_validation']=validation
    summary['D1_A0_revision1_metrics']=read_json(ECM/'results/fruit_expert_v2_audit_20261102/d1_new_metrics/summary.json')['conditions']['A0']
    summary['postprocessing_fix']=read_json(OUT/'physical_prefix_fix/validation.json')
    atomic_json(OUT/'summary.json',summary)
    old=summary['D1_A0_revision1_metrics'];dev=summary['dev'];g1=summary['G1'];passed=summary['G1_passed']
    lines=[f"阶段3完成：G1{'通过' if passed else '未通过'}，按§4.3停止并等待用户确认。",
        f"成功{ratio(g1['all']['success'])}；分任务count={ratio(by_task['FruitCount']['success'])}、occluded_ripe={ratio(by_task['FruitOccludedRipe']['success'])}。",
        f"首颗入篮{ratio(g1['all']['first_target_in_basket'])}，执行腕偏离{ratio(g1['all']['wrist_deviation_executed'])}，工程错误{ratio(g1['all']['engineering_error'])}。",
        f"v3.1专家、400条新示范、K0训练12000步及T1已完成；80条G1记忆审计通过数{validation['memory_audits_passed']}/80。",
        ('决策：全部G1门槛达标，确认后进入阶段5；尚未启动阶段5。' if passed else
         '决策：G1存在未达标项，确认后按阶段4开展DAgger；尚未启动阶段4。'),'',
        '## 做了什么与来源','',
        '按修订3选择通过全部核心项的canonical_v3.1_20261102，重新生成双任务干净train0–99和DART100–199各200条。'
        'OU噪声σ=.04rad、τ=10步，reach/pull/carry全幅、approach半幅，其余关闭，标签保持干净专家目标。'
        '共164639帧保留全部时间结构，observe前12000帧只作历史，152639监督样本。release/done继续监督，settle2。'
        '原构建器处理并重算统计，K0沿用79999初始化和旧12000步配方，仅改变数据、统计和8卡FSDP布局。', '',
        f'训练来源：`{RUN}/plan.json`、`protocol.json`、`gate_t1_k0.json`；构建来源：`{dataset}/meta/build.json`。'
        '240个记录损失区间齐全且有限，最后记录的区间均值0.0029；最终11999检查点携带的统计与数据集一致。', '',
        f'G1来源：`{OUT}/analysis/G1/summary.json`及相邻episodes.jsonl/csv；逐回合在G1/<任务>/ep<ID>。'
        f'开发集来源：analysis/dev_A0、analysis/dev_v2；旧A0来源：`{ECM}/results/fruit_expert_v2_audit_20261102/d1_new_metrics/summary.json`。', '',
        '## G1门槛','', '| 条件 | 实际 | 要求 | 判定 |','|---|---|---|---|',
        f"| 合计成功 | {ratio(recalculated['success'])} | ≥25% | {gates['overall_success']} |",
        f"| 每任务成功 | count {ratio(by_task['FruitCount']['success'])}；occluded {ratio(by_task['FruitOccludedRipe']['success'])} | 各≥15% | {gates['each_task_success']} |",
        f"| 首颗目标入篮 | {ratio(recalculated['first_target_in_basket'])} | ≥50% | {gates['first_target_in_basket']} |",
        f"| 执行腕偏离 | {ratio(recalculated['wrist_deviation_executed'])} | ≤5% | {gates['wrist_deviation']} |",
        f"| 工程错误 | {ratio(recalculated['engineering_error'])} | 0 | {gates['engineering_errors']} |", '',
        '## 全部统一指标及阶段1对照','',
        '旧A0与新开发集各40条（每任务20），G1为新冻结布局80条（每任务40），不是配对样本。'
        '旧A0的q7中心为旧专家−1.56，新条件为新训练示范中位数−1.5764960646629333；各自遵循专家版本定义。'
        '数据、统计、GPU布局以及平台前缀/包络均有获准变化，以下对照不把改善归因于单个因素，也不比较不同记忆的优劣。']
    for scope in ('all','FruitCount','FruitOccludedRipe'):
        source=lambda s:s['all'] if scope=='all' else s['by_task'][scope]
        versions={'D1 A0':source(old),'新dev off':source(dev['A0']),'新dev v2':source(dev['v2']),'G1 v2':source(g1)}
        lines+=['',f"### {'合计' if scope=='all' else scope}",'', '| 指标 | D1 A0 | 新dev off | 新dev v2 | G1 v2 |','|---|---:|---:|---:|---:|']
        for metric in ('success','fail','timeout','engineering_error','first_target_in_basket','never_detached','near_miss',
            'wrist_over_limit','wrist_deviation_executed','wrist_deviation_proposed','freeze_200','freeze_200_basket_unsuccessful','count_exact'):
            lines.append('| '+metric+' | '+' | '.join(ratio(v[metric]) for v in versions.values())+' |')
        for metric in ('correct_in_basket','wrong_in_basket'):
            lines.append('| '+metric+'（果实总数/有数据回合） | '+' | '.join(
                f"{v[metric]['total_fruits']}/{v[metric]['episodes_with_data']}" for v in versions.values())+' |')
        for metric in ('start_idle','leave_home_step','post_release_resume_steps'):
            lines.append('| '+metric+' 中位步 | '+' | '.join(str(v[metric]['median']) for v in versions.values())+' |')
        lines.append('| never_left_home | '+' | '.join(ratio(v['leave_home_step']['never_left_home']) for v in versions.values())+' |')
        lines.append('| 入篮后未恢复（右删失） | '+' | '.join(ratio(v['post_release_resume_steps']['unresumed_at_end']) for v in versions.values())+' |')
        lines.append('| 任一关节裁剪/控制步 | '+' | '.join(ratio(v['guard_clip_rate']['any_joint']) for v in versions.values())+' |')
        for j in range(7): lines.append(f'| 关节{j+1}裁剪/控制步 | '+' | '.join(ratio(v['guard_clip_rate']['per_joint'][j]) for v in versions.values())+' |')
        for context in ('carrying_fruit','closed_empty','open_after_basket','open_before_first_detach'):
            lines.append('| 偏离起始 '+context+' | '+' | '.join(ratio(v['deviation_onset_context'][context]) for v in versions.values())+' |')
    lines+=['','离家/入篮后恢复的中位数仅用已观察样本；从未离家和未恢复分别列分母，不把未观察值填0。'
        '冻结子集为已放下至少一颗且未成功的回合。count_exact对FruitOccludedRipe不适用。原始逐回合事件和来源均在分析JSONL。', '',
        '## 开发集参考、包络选择和分析修复','',
        '开发集离家中位数36步（≤60参考通过），原A0为138步；放下未成功子集冻结新dev off/v2均0/19，'
        f"原A0为{ratio(old['all']['freeze_200_basket_unsuccessful'])}。"
        '包络验收专家裁剪626/164639=0.3802%（<0.5%），投影55步；dev首颗均29/40，腕偏离off5/40、v2为0/40，故按预定规则采用v2。'
        'dev总任务成功off16/40、v2为15/40，差异完整报告；成功率不是额外的包络采用条件。'
        'guard_choice.json在所有G1启动前冻结，其余量、统计、checkpoint manifest和源码摘要均通过独立校验。', '',
        '最初分析器把只作历史的物理保持帧按旧视频示范规则删除，造成时间对齐报错。修复仅处理显式声明30步物理前缀的日志，'
        '严格核对物理/策略时钟、历史标记和固定保持；旧视频示范规则保持原样。52项测试通过，原400份开发集输入日志摘要完全不变，'
        '训练、控制和物理回放均未改动。错误分析保留在analysis_before_physical_prefix_fix_20261103；证据在physical_prefix_fix/validation.json。', '',
        '## 3×2新训练数据失败位置分布','', '| 数据/任务/格子 | 目标尝试 | 失败 |','|---|---:|---:|']
    for data,coverage in summary['training_data_failure_coverage'].items():
        for task,cells in coverage['by_task'].items():
            for cell,c in cells.items(): lines.append(f"| {data}/{task}/{cell} | {c['target_attempts']} | {c['failed_attempts']} |")
    lines+=['','分母为一次reach目标规划（包括失败）；失败归于导致整回合失败的目标初始位置。'
        'left:y>.08、right:y<−.08、middle闭区间；upper:z≥.54、lower:z<.54。未访问果实不虚增分母，逐例种子/xyz见data_failure_coverage.json。', '',
        '## 独立核验、决策与下一步','',
        f"80个唯一布局、两个任务各40、种子7。{validation['memory_audits_passed']}/80记忆审计通过，共{audit_decisions}次决策；"
        f"最大执行{max_steps}步（含30步保持），全部{action_steps}控制步已核对物理时钟和连续帧。"
        '重新从原始日志计算的全部指标和门槛与自动报告一致；训练封存来源、检查点、统计、包络及冻结划分摘要均未变。'
        '所有逐回合文件摘要和首次启动时间在independent_validation.json。没有发现完成布局被重复评测，也未使用val100–149。', '',
        ('G1全部达标；按§4.3停止等待用户确认，确认后进入阶段5，当前未启动阶段5。' if passed else
         'G1未全部达标；按§4.3停止等待用户确认，确认后进入阶段4，当前未启动阶段4。')]
    text='\n'.join(lines)+'\n';(OUT/'report.md').write_text(text)
    tracked=ECM/'reports/lowlevel_rescue_D3_20261102';atomic_json(tracked/'summary.json',summary)
    (ECM/'reports/lowlevel_rescue_D3_20261102.md').write_text(text)
    root=read_json(ECM/'reports/lowlevel_rescue_status.json');root.update(phase=3,instruction_revision=3,state='awaiting_G1_confirmation',
        G1_passed=passed,G1_gates=gates,updated_utc=timestamp(),decision='Stop at §4.3 G1 review; phase5 if passed, phase4 if failed, after user confirmation.',
        next='Await user confirmation to enter phase5.' if passed else 'Await user confirmation to enter phase4.',
        stage3_platform={'base_condition':'A0','joint_guard':'v2','exec_horizon':16,
            'observe_hold_steps':30,'total_step_budget':1300,'guard_choice_fixed_before_G1':True})
    root['links'].update(D3_versioned_report='reports/lowlevel_rescue_D3_20261102.md',D3_versioned_summary='reports/lowlevel_rescue_D3_20261102/summary.json',
        D3_validation=str(OUT/'independent_validation.json'))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)
    print({'validated':True,'G1_passed':passed,'G1_gates':gates,'audited_decisions':audit_decisions},flush=True)


if __name__=='__main__': main()
