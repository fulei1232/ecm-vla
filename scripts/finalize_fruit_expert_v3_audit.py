#!/usr/bin/env python3
"""Independent artifact review and Chinese D2-v3 handoff; no expert iteration."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from fruit_expert_audit_stats import load_episode
from v_a_common import ECM,atomic_json,read_json,sha256,timestamp


def main():
    out=ECM/'results/fruit_expert_v3_audit_20261102'
    state=read_json(out/'status.json');audit=read_json(out/'audit.json');plan=read_json(out/'plan.json')
    assert state['state']=='awaiting_phase2_v3_review' and not audit['all_core_gates_passed']
    assert audit['audit_scale_complete'] and len(audit['videos'])==4
    for path,digest in plan['source_sha256'].items(): assert sha256(Path(path))==digest,path
    preservation=read_json(out/'v2_preservation.json')
    for category in ('source_snapshots','v2_artifact_sha256'):
        for path,digest in preservation[category].items(): assert sha256(Path(path))==digest,path
    assert sha256(ECM/'reports/split_registry_20261102.json')==plan['registry_sha256']
    assert sha256(ECM/'reports/HANDOFF_lowlevel_rescue_20261102.md')==plan['handoff_sha256']
    assert all(row['exit_code']==0 for row in state['attempts'])
    stats=audit['comparison']['canonical_v3'];assert stats['episodes']==200
    swings={phase:[] for phase in ('reach','carry')};wrist=[]
    frames=0;state_bad=0;command_bad=0;files=0;reach_goals=[]
    q7_states=[];q7_commands=[];manifests=set()
    for record in stats['records']:
        path=Path(record['manifest']);assert sha256(path)==record['manifest_sha256']
        man,q,a,supervised,gt=load_episode(path);manifests.add((man['task'],man['episode']))
        assert man['solver_version']=='canonical_v3_20261102' and man['settle_steps']==2
        assert len(q)==man['steps'] and gt[-1]['success'] and supervised.all()
        for key in ('h5','gt'): assert sha256(path.parent/man[key])==man[key+'_sha256'];files+=1
        frames+=len(q);state_bad+=int(np.sum(np.abs(q[:,6])>np.pi));command_bad+=int(np.sum(np.abs(a[:,6])>np.pi))
        q7_states.append(q[:,6]);q7_commands.append(a[:,6])
        phases=[r['phase'] for r in gt]
        starts=[0]+[i for i in range(1,len(q)) if phases[i]!=phases[i-1]]
        for start,stop in zip(starts,starts[1:]+[len(q)]):
            phase=phases[start]
            if phase in swings: swings[phase].append(float(np.ptp(q[start:stop,0])))
            if phase=='approach': wrist.append(q[stop-1,4:7])
        for trace in man['planning_audit']:
            if trace.get('phase') not in ('reach','carry') or trace.get('status')!='Success': continue
            with np.load(trace['snapshot'],allow_pickle=False) as snapshot:
                reference=snapshot['reference'][:7]
                if trace['phase']=='carry': np.testing.assert_array_equal(trace['goal_arm_qpos'],reference)
                else: reach_goals.append(abs(float(trace['goal_arm_qpos'][0]-reference[0])))
    assert manifests=={(task,e) for task in ('count','occluded_ripe') for e in range(100)}
    assert frames==stats['state_samples']==81730
    assert state_bad==stats['q7_states_outside_pi']['numerator']==0
    assert command_bad==stats['q7_commands_outside_pi']['numerator']==0
    q1={phase:float(np.quantile(values,.9)) for phase,values in swings.items()}
    for phase in q1: assert q1[phase]==stats['motion'][phase]['q1_swing_p90']
    w=np.asarray(wrist);center=np.median(w,axis=0)
    wrist_counts={f'q{j+5}':int(np.sum(np.abs(w[:,j]-center[j])<=1)) for j in range(3)}
    for q,n in wrist_counts.items(): assert n==stats['approach_end_wrist'][q]['within_median_plusminus_1rad']['numerator']
    root=Path(audit['selection']['data_root']);attempts=[read_json(path) for path in root.glob('*/attempts/ep*_seed*.json')]
    assert len(attempts)==224 and sum(row['ok'] for row in attempts)==200
    failed_files=0;failed_frames=0;failed_q7_state_bad=0;failed_q7_command_bad=0
    for row in attempts:
        if row['ok']: continue
        directory=root/row['task']
        for key in ('failed_h5','failed_gt'):
            assert sha256(directory/row[key])==row[key+'_sha256'];failed_files+=1
        with h5py.File(directory/row['failed_h5']) as f:
            episode=f"episode_{row['episode']}"
            ep=f[episode];n=int(ep['setup/n_timesteps'][()]);assert n==row['steps']
            q=np.asarray([ep[f'timestep_{i}/obs/joint_state'][6] for i in range(n)])
            a=np.asarray([ep[f'timestep_{i}/action/joint_action'][6] for i in range(n)])
        failed_frames+=n;failed_q7_state_bad+=int(np.sum(np.abs(q)>np.pi));failed_q7_command_bad+=int(np.sum(np.abs(a)>np.pi))
    assert frames+failed_frames==audit['noncore']['all_attempt_q7']['samples']==85640
    assert failed_q7_state_bad==failed_q7_command_bad==0
    planning=read_json(out/'planning/summary.json')
    deterministic=planning['determinism']['cases']
    assert len(deterministic)==100 and Counter(row['phase'] for row in deterministic)=={'reach':50,'carry':50}
    for row in deterministic:
        assert sha256(Path(row['snapshot']))==row['snapshot_sha256']
        assert sha256(Path(row['manifest']))==row['manifest_sha256']
        assert len(row['hashes'])==3 and len(set(row['hashes']))==1 and row['hashes'][0]==row['recorded_result_sha256']
        assert all(trace==row['traces'][0] for trace in row['traces']) and row['passed']
    continuity=planning['continuity']['cases'];assert len(continuity)==100
    continuity_failures=[];target_ranges=[];passed=0
    for row in continuity:
        assert sha256(Path(row['snapshot']))==row['snapshot_sha256']
        assert sha256(Path(row['manifest']))==row['manifest_sha256']
        assert len(row['traces'])==6 and len(row['perturbation_offsets_m'])==5
        assert np.all(np.abs(np.asarray(row['perturbation_offsets_m']))==.002)
        assert row['joint_perturbation_abs_rad']==.001
        goals=np.asarray(row['goals']);assert goals.shape==(6,7) and np.isfinite(goals).all()
        delta=float(np.ptp(goals,axis=0).max());target_ranges.append(delta)
        ok=delta<.05 and len(set(row['methods']))==1 and all(t['status']=='Success' for t in row['traces'])
        assert ok==row['passed'];passed+=int(ok)
        if not ok:
            continuity_failures.append({'task':row['task'],'episode':row['episode'],'timestep':row['timestep'],
                'goal_max_pairwise_difference_rad':delta,'methods':row['methods'],'waypoints':row['waypoints'],
                'statuses':[t['status'] for t in row['traces']],'source_snapshot':row['snapshot']})
    assert passed==planning['continuity']['passed_instances']==97
    physics=read_json(out/'physics/summary.json');physics_checks=[]
    for case in physics['fixed_action_replay']:
        assert case['complete']
        e=case['episode'];paths=[out/f'physics/ep{e}/repeat{j}/trace.json' for j in range(3)]
        replays=[read_json(path) for path in paths]
        assert all(replay['states']==replays[0]['states'] for replay in replays)
        assert all(replay['final_success'] for replay in replays)
        assert all(len(replay['states'])==replay['steps']+1 for replay in replays)
        source=Path(replays[0]['source']);original=read_json(source)
        assert sha256(Path(replays[0]['source_h5']))==replays[0]['source_sha256']
        with h5py.File(replays[0]['source_h5']) as f:
            ep=f[f'episode_{e}'];n=replays[0]['steps']
            actions=np.asarray([ep[f'timestep_{i}/action/joint_action'][()] for i in range(n)])
            robot=np.asarray([np.r_[ep[f'timestep_{i}/obs/joint_state'][()],ep[f'timestep_{i}/obs/gripper_state'][()]] for i in range(n)])
        assert all(hashlib.sha256(actions.tobytes()).hexdigest()==replay['action_sha256'] for replay in replays)
        replay_robot=np.asarray([state['robot_qpos'] for state in replays[0]['states'][:-1]])
        robot_diff=np.max(np.abs(robot-replay_robot),axis=1)
        gt=[json.loads(line) for line in (source.parent/original['gt']).read_text().splitlines()]
        source_fields_match=all(
            state['robot_tcp_pose']==row['tcp_pos']+row['tcp_quat'] and
            [pose[:3] for pose in state['fruit_pose']]==[fruit['pos'] for fruit in row['fruits']] and
            state['leaf_push']==row['occluder_push'] for state,row in zip(replays[0]['states'],gt))
        physics_checks.append({'episode':e,'three_replays_exactly_identical':True,'actions':n,
            'trace_sha256':{str(path):sha256(path) for path in paths},
            'source_robot_state_max_difference':float(robot_diff.max()),
            'source_tcp_fruit_positions_leaf_push_exact_match':source_fields_match})
    videos=[]
    for video in audit['videos']:
        for key in ('video','frames'): assert Path(video[key]).stat().st_size>0
        videos.append({**video,'video_sha256':sha256(Path(video['video'])),'frames_sha256':sha256(Path(video['frames']))})
    rrt_comparison={}
    for name,version in audit['comparison'].items():
        traces=[]
        for record in version['records']:
            man=read_json(Path(record['manifest']))
            traces.extend(t for t in man.get('planning_audit',[]) if t.get('phase') in ('reach','carry'))
        used=lambda trace: trace.get('method')=='rrt' or any(t.get('method')=='rrt' for t in trace.get('attempts',[]))
        accepted={'rrt_segments':sum(used(trace) for trace in traces),'all_segments':len(traces)}
        for record in version['failed_attempt_records']:
            for failure in record['failures']:
                traces.extend(t for t in failure.get('planning_audit',[]) if t.get('phase') in ('reach','carry'))
        rrt_comparison[name]={'accepted':accepted,
            'all_attempts':{'rrt_segments':sum(used(trace) for trace in traces),'all_segments':len(traces)},
            'definition':'A segment counts if any RRT attempt was made, even if it failed before another branch succeeded. No trace in legacy.'}
    validation={'validated_utc':timestamp(),'new_manifest_count':200,'new_payload_hashes_verified':files,
        'failed_attempt_payload_hashes_verified':failed_files,'all_attempts':len(attempts),'all_process_exits_zero':True,
        'runtime_sources_verified':len(plan['source_sha256']),'v2_metadata_artifacts_unchanged':len(preservation['v2_artifact_sha256']),
        'v2_source_snapshots_verified':len(preservation['source_snapshots']),'frozen_registry_and_handoff_unchanged':True,
        'independent_motion':{'reach_carry_q1_swing_p90':q1,'reach_goal_q1_distance_to_rest_p90':float(np.quantile(reach_goals,.9))},
        'wrist_within_1rad_counts':wrist_counts,'wrist_samples':len(w),'q7_accepted_states_and_commands_outside_pi':[state_bad,command_bad],
        'q7_failed_attempt_states_and_commands_outside_pi':[failed_q7_state_bad,failed_q7_command_bad],
        'accepted_frames':frames,'failed_attempt_frames':failed_frames,'determinism_100_cases_verified':True,
        'continuity_passed':passed,'goal_only_under_005rad_instances':sum(value<.05 for value in target_ranges),
        'max_goal_only_difference_rad':max(target_ranges),'continuity_failures':continuity_failures,
        'perturbation_scope':'Each XYZ coordinate +/-0.002m (Euclidean magnitude sqrt(3)*2mm); each measured arm coordinate +/-0.001rad.',
        'physics_checks':physics_checks,'four_videos_and_frames_verified':videos,
        'rrt_usage_comparison':rrt_comparison,
        'D1_metrics_unchanged':audit['D1_revision1_metrics']==read_json(ECM/'results/fruit_expert_v2_audit_20261102/d1_new_metrics/summary.json')}
    atomic_json(out/'independent_validation.json',validation)
    for name in ('audit','summary'):
        target=out/f'{name}_before_independent_review.json'
        if not target.exists(): atomic_json(target,read_json(out/f'{name}.json'))
    if not (out/'report_before_independent_review.md').exists():
        (out/'report_before_independent_review.md').write_text((out/'report.md').read_text())
    audit['independent_validation']=validation
    atomic_json(out/'audit.json',audit);atomic_json(out/'summary.json',audit)
    lines=(out/'report_before_independent_review.md').read_text().splitlines()
    lines[:5]=[
        '阶段2专家v3完成：五个核心项通过3项、失败2项，按§10.5停下等待审阅。',
        '失败：连续性97/100（要求≥98/100）；reach/carry第1关节摆幅P90=2.045175/2.048776rad（要求各<1）。',
        '通过：规划确定性100/100；抓取q5/q6/q7一致性各540/540；全部记录q7指令和实测均在[−π,π]。',
        '完成200条示范/224次尝试；保留settle=2；D2三个布局固定动作各回放3次，所有已记录物理状态逐位一致。',
        '决策：不进入阶段3，不再自行迭代v3；A0平台、legacy/nearest及v2产物保留，等待指挥者决定。']
    text='\n'.join(lines)+'\n\n## 独立复核与失败项定位\n\n'
    def rrt_ratio(version,scope):
        values=rrt_comparison[version][scope];n,d=values['rrt_segments'],values['all_segments']
        return f'{n}/{d} ({n/d:.2%})' if d else '未记录'
    table_lines=text.splitlines()
    for i,line in enumerate(table_lines):
        if line.startswith('| RRT比例 |'):
            table_lines[i]='| RRT使用比例（含失败尝试） | '+' | '.join(rrt_ratio(name,'all_attempts') for name in audit['comparison'])+' | ≤2%，非核心 |'
            table_lines.insert(i+1,'| 被接受示范中使用过RRT | '+' | '.join(rrt_ratio(name,'accepted') for name in audit['comparison'])+' | 描述对照 |')
            break
    for i,line in enumerate(table_lines):
        if line.startswith('| other |'):
            values=[version['static_delayed_supervision']['all'] for version in audit['comparison'].values()]
            table_lines.insert(i+1,'| 合计 | '+' | '.join(f"{v['numerator']}/{v['denominator']} ({v['fraction']:.2%})" for v in values)+' |')
            break
    text='\n'.join(table_lines)+'\n'
    text+=('RRT使用对照按段统计“是否调用过RRT”，包含RRT失败后改用其他分支的段；'
           'v2全部尝试为62/1136，被接受示范为55/1094；v3分别为26/1144、3/1080。'
           '这与只统计最终成功路径为RRT的v2计数47/1094不同，口径与来源见independent_validation.json/rrt_usage_comparison。\n\n')
    text+=(f'独立校验200个manifest、{files}个成功内容文件哈希、{failed_files}个失败内容文件哈希、17个运行源文件、'
           '236个v2元数据文件及7份v2源文件快照；冻结划分、交接文档、D1补算结果均未变。全部子进程退出码为0。'
           '逐示范重新读取81730个成功控制步，失败另3910步，全部q7记录无越界；手腕及q1摆幅独立重算一致。'
           '4段视频及抽帧、长度直方图已检查。完整证据见independent_validation.json。\n\n')
    text+=(f'连续性中，100/100实例的纯目标最大两两差都<0.05rad（最大{max(target_ranges):.9f}rad），'
           '失败来自3个实例的规划方法切换，其中1次RRT路径被碰撞/q7检查拒绝；不能把目标平滑误写成完整连续性达标。'
           '每坐标扰动为±2mm，对应欧氏位移3.464mm；测量关节每坐标为±0.001rad，五组规则在运行前固定。\n\n')
    text+='| 连续性失败实例 | 最大目标差 rad | 方法（基准+5次） | 失败情况 |\n|---|---:|---|---|\n'
    for row in continuity_failures:
        text+=f"| {row['task']} ep{row['episode']} step{row['timestep']} | {row['goal_max_pairwise_difference_rad']:.9f} | {row['methods']} | {row['statuses']} |\n"
    text+=(f'\nreach目标与固定rest的q1绝对差P90={validation["independent_motion"]["reach_goal_q1_distance_to_rest_p90"]:.9f}rad；'
           'carry目标均已逐段核实等于rest。固定归位与reach目标之间的大角度距离与实测摆幅一致；'
           '本轮没有为通过门槛更改rest、权重、摆幅口径或绕行姿态。\n\n')
    text+=('D2三个布局在固定动作下均未观测到物理分叉，首次分叉步数、对象和幅度均为“无”。'
           '原D2已有记录中，ep0/1/3的首次不同动作分别位于0起始索引156/187/211，均为第二颗果实的reach；'
           '执行该动作后才出现已记录机器人状态差，即已执行157/188/212步时。'
           '原D2对应规划trace包含RRT第1候选成功与退到第2候选直线的分支差异。'
           '本次结果不支持把原分叉直接归因于物理微差，规划后备路径的不确定性是更直接的解释；'
           '后者仍应标作由动作、trace和固定回放共同支持的推断，原D2没有记录的速度/姿态不能倒推出数值。\n\n')
    text+=('settle静止监督命中由v2的1734/6564降到0/1080，但release从0/4376升到1700/4320；'
           '全阶段命中应一起看，不能只凭settle单项认定总体等待问题解决。'
           '三方总体命中率见audit.json/comparison各版本static_delayed_supervision/all。\n')
    (out/'report.md').write_text(text)
    tracked=ECM/'reports/lowlevel_rescue_D2_v3_20261102';tracked.mkdir(exist_ok=True)
    atomic_json(tracked/'summary.json',audit)
    prefix='../results/'+out.name+'/'
    (ECM/'reports/lowlevel_rescue_D2_v3_20261102.md').write_text(text.replace(
        '](episode_lengths.png)',']('+prefix+'episode_lengths.png)').replace('](videos/',']('+prefix+'videos/'))
    root_status=read_json(ECM/'reports/lowlevel_rescue_status.json')
    root_status.update(phase=2,instruction_revision=2,state='awaiting_phase2_v3_review',updated_utc=timestamp(),
        all_core_gates_passed=False,core_gates=audit['core_gates'],decision='Stop under §10.1/10.5; no phase3 or v3 iteration without user review.')
    root_status['links'].update(D2_v3_report=str(out/'report.md'),D2_v3_audit=str(out/'audit.json'),
        D2_v3_summary=str(out/'summary.json'),D2_v3_validation=str(out/'independent_validation.json'),
        D2_v3_versioned_report='reports/lowlevel_rescue_D2_v3_20261102.md',
        D2_v3_versioned_summary='reports/lowlevel_rescue_D2_v3_20261102/summary.json')
    atomic_json(ECM/'reports/lowlevel_rescue_status.json',root_status)
    print({'validated':True,'all_core_gates_passed':False,'core_gates':audit['core_gates'],
           'continuity_failures':continuity_failures,'physics_checks':physics_checks},flush=True)


if __name__=='__main__': main()
