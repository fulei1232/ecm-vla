"""Stage3 recipes, expert-derived guard statistics and reporting helpers."""
from copy import deepcopy
from pathlib import Path
import sys

import h5py
import numpy as np

from v_a_common import ECM,POLICY,CHECKPOINT,atomic_json,read_json,sha256,timestamp


def make_training_spec(run,dataset):
    old=read_json(ECM/'runs/fruit_dart_kf_20261027/arms/k0.json')
    new=deepcopy(old);new.update(dataset=str(dataset),assets_dir=str(dataset/'assets'),
        checkpoint_base_dir=str(run/'ckpt'),gpus=list(range(8)))
    new['train']['fsdp_devices']=8
    # Remove only the explicitly allowed three differences and output location.
    restored=deepcopy(new)
    for key in ('dataset','assets_dir','checkpoint_base_dir','gpus'): restored[key]=old[key]
    restored['train']['fsdp_devices']=old['train']['fsdp_devices']
    assert restored==old
    return new


def prepare_training(run,dataset,selection,protocol_doc):
    from run_finetune_kf import CODE
    if (run/'protocol.json').exists(): return
    build=read_json(dataset/'meta/build.json');assert build['passed'] and build['episodes']==400
    episodes=read_json(dataset/'meta/episodes.json');assert len(episodes)==400 and all(r['exec_start_idx']==30 for r in episodes)
    spec=make_training_spec(run,dataset);(run/'arms').mkdir(parents=True,exist_ok=True);atomic_json(run/'arms/k0.json',spec)
    sources=CODE+[ECM/'scripts/fruit_rescue_stage3_support.py',ECM/'scripts/run_fruit_rescue_stage3.py']
    protocol={'name':'fruit_lowlevel_rescue_k0v2','study':'fruit_lowlevel_rescue','formal':True,'created_utc':timestamp(),
        'protocol_doc':str(protocol_doc),'protocol_doc_sha256':sha256(protocol_doc),'dataset':str(dataset),
        'dataset_build_sha256':sha256(dataset/'meta/build.json'),'dataset_episodes_sha256':sha256(dataset/'meta/episodes.json'),
        'dataset_copy_check':None,'assets_dir':str(dataset/'assets'),
        'assets_norm_stats_sha256':sha256(dataset/'assets/robomme/norm_stats.json'),
        'init_checkpoint':str(CHECKPOINT),'init_params_manifest_sha256':sha256(CHECKPOINT/'params/manifest.ocdbt'),
        'arms':{'k0':{'events':0,'spec_sha256':sha256(run/'arms/k0.json')}},
        'code_identity':[{'path':str(path),'sha256':sha256(path)} for path in sources],
        'selected_expert':selection,'allowed_changes':['data','normalization statistics','GPU layout (FSDP8)'],
        'restart_policy':'an incomplete arm may be superseded and restarted once; all previous logs/checkpoints retained'}
    atomic_json(run/'protocol.json',protocol)
    atomic_json(run/'plan.json',{'baseline_spec':str(ECM/'runs/fruit_dart_kf_20261027/arms/k0.json'),
        'allowed_differences':protocol['allowed_changes'],'output_only_difference':'checkpoint_base_dir',
        'selected_expert':selection,'recipe':spec,'norm_stats_recomputed':True,'observe_history_prefix_steps':30})


def expert_profile(clean,dart,out):
    sys.path.insert(0,str(ECM/'src'))
    from memory_management.fruit_rescue_platform import guard_v2
    states=[];actions=[];wrist=[];records=[]
    for root,first,stop in ((clean,0,100),(dart,100,200)):
        for task in ('count','occluded_ripe'):
            for e in range(first,stop):
                manifest=root/task/f'ep{e}_manifest.json';man=read_json(manifest)
                assert man['history_prefix_steps']==30
                with h5py.File(root/task/man['h5']) as f:
                    ep=f[f'episode_{e}'];n=int(ep['setup/n_timesteps'][()])
                    q=np.asarray([ep[f'timestep_{i}/obs/joint_state'][()] for i in range(n)],dtype=np.float64)
                    a=np.asarray([ep[f'timestep_{i}/action/joint_action'][()] for i in range(n)],dtype=np.float64)
                    flags=[bool(ep[f'timestep_{i}/info/is_video_demo'][()]) for i in range(n)]
                assert flags==[True]*30+[False]*(n-30)
                gt=read_jsonl(root/task/man['gt'])
                for i,row in enumerate(gt):
                    if row['phase']=='approach' and (i==n-1 or gt[i+1]['phase']!='approach'): wrist.append(q[i,6])
                states.append(q);actions.append(a)
                records.append({'manifest':str(manifest),'sha256':sha256(manifest),'frames':n})
    q=np.concatenate(states);a=np.concatenate(actions)
    guard={'version':'v2','lower':(np.quantile(q,.005,axis=0)-.1).tolist(),
           'upper':(np.quantile(q,.995,axis=0)+.1).tolist(),
           'v_max':(1.5*np.quantile(np.abs(a[:,:7]-q),.999,axis=0)).tolist(),
           'definition':'state [.5%,99.5%]+/-.1rad; 1.5*P99.9 abs(clean action label - measured state); arm only'}
    clipped=0;projections=0
    for proposed,measured in zip(a,q):
        _,event=guard_v2(proposed,measured,guard);clipped+=event['guard_clipped'];projections+=event['guard_projection']
    profile={'generated_utc':timestamp(),'q7_center':float(np.median(wrist)),'expert_records':records,
        'states':len(q),'guard':guard,'expert_clip_validation':{'clipped_steps':clipped,'all_steps':len(q),
            'fraction':clipped/len(q),'projection_steps':projections,'passed':clipped/len(q)<.005}}
    atomic_json(out,profile);return profile


def read_jsonl(path):
    import json
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def report_stage3(out,run,selection,dev,guard_choice,g1):
    minimum=lambda value,threshold:value is not None and value>=threshold
    maximum=lambda value,threshold:value is not None and value<=threshold
    gate={'overall_success':minimum(g1['all']['success']['fraction'],.25),
        'each_task_success':all(minimum(row['success']['fraction'],.15) for row in g1['by_task'].values()),
        'first_target_in_basket':minimum(g1['all']['first_target_in_basket']['fraction'],.5),
        'wrist_deviation':maximum(g1['all']['wrist_deviation_executed']['fraction'],.05),
        'engineering_errors':g1['all']['engineering_error']['numerator']==0}
    summary={'phase':3,'revision':3,'generated_utc':timestamp(),'selected_expert':selection,
        'training_plan':read_json(run/'plan.json'),'T1':read_json(run/'gate_t1_k0.json'),
        'dev':dev,'guard_decision':guard_choice,'G1':g1,'G1_gates':gate,'G1_passed':all(gate.values()),
        'training_data_failure_coverage':read_json(out/'data_failure_coverage.json'),
        'D1_A0_comparison':read_json(ECM/'reports/lowlevel_rescue_D1_20261102/summary.json')['conditions']['A0'],
        'requires_confirmation_after_G1':True,'next_if_confirmed':'phase5' if all(gate.values()) else 'phase4'}
    atomic_json(out/'summary.json',summary)
    ratio=lambda row:f"{row['numerator']}/{row['denominator']}"
    lines=[f"阶段3完成：G1{'通过' if summary['G1_passed'] else '未通过'}，按§4.3汇报后等待确认。",
        f"专家为{selection['solver_version']}；400条干净+DART全部重新生成，observe30帧只作历史。",
        f"K0沿用旧训练配方12000步，T1通过；只改变数据、归一化统计和8卡布局。",
        f"G1整任务成功{ratio(g1['all']['success'])}，首颗入篮{ratio(g1['all']['first_target_in_basket'])}，腕偏离{ratio(g1['all']['wrist_deviation_executed'])}。",
        '平台已在G1前冻结：A0+30步固定保持，保持计入1300步总预算。','',
        '## 做了什么','',
        '按§11.5选择专家，按§11.6重新生成干净train0–99与DART100–199各双任务；OU噪声σ=.04rad、τ=10步，reach/pull/carry全幅、approach半幅、其他阶段关闭。标签仍为干净专家目标。所有帧保留，前30帧历史，不对release/done去监督。训练构建使用原官方处理器，重新计算归一化统计。', '',
        f"来源：{run}/plan.json、protocol.json、gate_t1_k0.json；训练与日志保留在{run}。",'',
        '## G1指标','', '| 范围 | 成功 | 首颗入篮 | 腕偏离 | 工程错误 |','|---|---:|---:|---:|---:|']
    for name,row in [('合计',g1['all']),*g1['by_task'].items()]:
        lines.append(f"| {name} | {ratio(row['success'])} | {ratio(row['first_target_in_basket'])} | {ratio(row['wrist_deviation_executed'])} | {ratio(row['engineering_error'])} |")
    lines+=['',f"门槛判定：{gate}。全部§5及§9.5指标、分任务分母、偏离起始上下文及逐关节clip见summary.json/G1和analysis/G1。",'',
        '## 开发集、包络v2与参考检查','',f"包络决定：{guard_choice}。",'',
        'dev200–219的全部指标与不超过60步离家参考、放下后冻结对照均在summary.json/dev及D1_A0_comparison。包络采用仅在专家裁剪率<.5%、首颗入篮不降低且腕偏离更低时；否则保持off，非阻塞。G1之前固定该选择，G1只使用dev220–259、模型种子7一次，不用val。', '',
        '## 选定专家新数据的3×2失败位置分布','',
        '| 数据/任务/格子 | 目标尝试数 | 失败数 |','|---|---:|---:|']
    for source,coverage in summary['training_data_failure_coverage'].items():
        for task,cells in coverage['by_task'].items():
            for cell,counts in cells.items():
                lines.append(f"| {source}/{task}/{cell} | {counts['target_attempts']} | {counts['failed_attempts']} |")
    lines+=['','分母为实际reach规划目标尝试，失败归于导致整回合失败的目标；位置是布局中的初始xyz，具体边界和逐条失败源见summary.json/training_data_failure_coverage。', '',
        '## 决策与下一步','',
        'G1结果已交付，按§4.3等待用户确认；通过后拟进入阶段5，未通过拟进入阶段4。本流程没有自动启动后续阶段。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    tracked=ECM/'reports/lowlevel_rescue_D3_20261102';tracked.mkdir(exist_ok=True);atomic_json(tracked/'summary.json',summary)
    (ECM/'reports/lowlevel_rescue_D3_20261102.md').write_text((out/'report.md').read_text())
    root=read_json(ECM/'reports/lowlevel_rescue_status.json');root.update(phase=3,instruction_revision=3,state='awaiting_G1_confirmation',
        G1_passed=summary['G1_passed'],updated_utc=timestamp())
    root['links'].update(D3_report=str(out/'report.md'),D3_summary=str(out/'summary.json'),D3_training=str(run))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)
