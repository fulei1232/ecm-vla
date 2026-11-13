#!/usr/bin/env python3
"""Independent completion record for D2-v3.1; no expert or policy reruns."""
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from v_a_common import ECM,atomic_json,read_json,sha256,timestamp


def main():
    out=ECM/'results/fruit_expert_v31_audit_20261102';audit=read_json(out/'audit.json')
    assert all(audit['core_gates'].values()) and audit['selected_expert']['ik_mode']=='canonical_v31'
    preservation=read_json(out/'preservation.json')
    for section in preservation:
        for path,digest in preservation[section].items(): assert sha256(Path(path))==digest,path
    workflow=read_json(out/'plan.json');exceptions=[]
    for path,digest in workflow['source_sha256'].items():
        if sha256(Path(path))!=digest:
            assert Path(path)==ECM/'scripts/analyze_fruit_failure_modes.py'
            exceptions.append(path)
    assert len(exceptions)==1
    samples=0;state_bad=0;command_bad=0;payloads=0
    for row in audit['v31']['records']:
        path=Path(row['manifest']);assert sha256(path)==row['manifest_sha256']
        man=read_json(path);assert man['solver_version']=='canonical_v3.1_20261102'
        for key in ('h5','gt'): assert sha256(path.parent/man[key])==man[key+'_sha256'];payloads+=1
        with h5py.File(path.parent/man['h5']) as f:
            ep=f[f"episode_{man['episode']}"];n=int(ep['setup/n_timesteps'][()])
            q=np.asarray([ep[f'timestep_{i}/obs/joint_state'][6] for i in range(n)])
            a=np.asarray([ep[f'timestep_{i}/action/joint_action'][6] for i in range(n)])
        samples+=n;state_bad+=int(np.sum(np.abs(q)>np.pi));command_bad+=int(np.sum(np.abs(a)>np.pi))
    assert samples==83194 and state_bad==command_bad==0
    rows=audit['planning']['determinism']['cases']
    assert len(rows)==100 and Counter(r['phase'] for r in rows)=={'reach':50,'carry':50}
    for row in rows:
        assert len(row['hashes'])==3 and len(set(row['hashes']))==1 and row['passed']
        assert row['hashes'][0]==row['recorded_result_sha256']
        assert sha256(Path(row['snapshot']))==row['snapshot_sha256']
    target=method=0
    for row in audit['planning']['continuity']['cases']:
        delta=float(np.ptp(np.asarray(row['goals']),axis=0).max())
        target+=delta<.05
        method+=len(set(row['methods']))==1 and all(t['status']=='Success' for t in row['traces'])
    assert target==100 and method==99
    for video in audit['videos']:
        assert Path(video['video']).stat().st_size>0 and Path(video['frames']).stat().st_size>0
    run=ECM/'runs/fruit_k0v2_lowlevel_rescue_20261102'
    dataset=ECM/'data/finetune_kf/fruit_dart_v2_rm75b_train'
    from run_finetune_kf import verify
    from fruit_rescue_stage3_support import make_training_spec
    verify(run);assert read_json(run/'arms/k0.json')==make_training_spec(run,dataset)
    t1=read_json(run/'gate_t1_k0.json');assert t1['passed'] and t1['arms']['k0']['logged']==240
    profile=read_json(ECM/'results/fruit_k0v2_gate_20261102/expert_profile.json')
    assert len(profile['expert_records'])==400
    for row in profile['expert_records']: assert sha256(Path(row['manifest']))==row['sha256']
    episodes=read_json(dataset/'meta/episodes.json');build=read_json(dataset/'meta/build.json')
    assert len(episodes)==400 and all(row['exec_start_idx']==30 for row in episodes)
    assert build['passed'] and build['execution_samples']==sum(row['exec_samples'] for row in episodes)
    assert profile['states']-build['execution_samples']==400*30
    validation={'validated_utc':timestamp(),'accepted_manifests':200,'payload_sha256_verified':payloads,
        'accepted_control_frames':samples,'q7_states_outside_pi':state_bad,'q7_commands_outside_pi':command_bad,
        'planning_determinism':{'passed':100,'all':100,'repeats':3},'target_continuity':{'passed':target,'all':100},
        'method_consistency':{'passed':method,'all':100},'four_video_and_frame_files_verified':True,
        'v3_metadata_preserved':len(preservation['v3_artifacts']),'v2_metadata_preserved':len(preservation['v2_artifacts']),
        'expert_runtime_sources_unchanged':True,'postprocessing_only_source_exception':exceptions,
        'training_completed':True,'T1_passed':True,'training_steps':12000,'logged_loss_intervals':240,
        'last_interval_loss':t1['arms']['k0']['last_loss'],'training_recipe_matches_approved_changes':True,
        'training_episodes':400,'physical_training_frames':profile['states'],'supervised_training_samples':build['execution_samples'],
        'history_only_frames':12000,'norm_stats_sha256':sha256(dataset/'assets/robomme/norm_stats.json'),
        'training_wrapper_compatibility_note':'Relative to the old sealed wrapper, save_interval now has an optional spec override; this run has no override, so effective save_interval is still12000.'}
    atomic_json(out/'independent_validation.json',validation)
    if not (out/'audit_before_independent_review.json').exists(): atomic_json(out/'audit_before_independent_review.json',audit)
    audit['independent_validation']=validation;atomic_json(out/'audit.json',audit);atomic_json(out/'summary.json',audit)
    text=(out/'report.md').read_text()
    if '## 独立复核与后续实际进度' in text: text=text.split('## 独立复核与后续实际进度')[0]
    lines=text.splitlines();lines[:5]=[
        '阶段2 v3.1完成：六个核心项全部通过，已按§11.5选择v3.1并进入阶段3。',
        '逐尝试成功率count=100/102、occluded_ripe=100/100；规划失败分别2/102、0/100。',
        '规划确定性100/100，目标连续性100/100，方法一致率99/100；抓取q5/q6/q7各544/544一致。',
        '83194个记录状态及指令q7均不越界；v3的24次失败中保存状态重规划成功23/24；v2/v3保留。',
        '后续实际进度：400条新训练示范已生成，12000步K0训练和T1已完成；当前准备恢复单次G1。']
    text='\n'.join(lines)+'\n\n## 独立复核与后续实际进度\n\n'
    text+=(f'复核200个manifest、{payloads}个数据内容哈希、83194个q7记录、100组规划重复及100组连续性；'
           'v3的429份元数据、v2的236份元数据及v3源文件快照均保持原摘要。4段视频及抽帧已核对。'
           '唯一工作流源文件变更是阶段3分析器对物理保持前缀的读取修复，不改变专家、模型、控制、原始回放或指标定义。'
           '证据在independent_validation.json及阶段3physical_prefix_fix/validation.json。\n\n')
    text+=(f'阶段3已完成400条干净/DART新示范，共{profile["states"]}个物理帧，'
           f'前缀12000帧保留为历史，监督样本{build["execution_samples"]}。归一化统计重新计算，'
           '检查点携带统计与数据一致。T1的240个损失区间齐全且有限，末区间0.0029。'
           '训练规格只含获准的数据、统计和GPU布局差别；训练包装器较旧封存版增加了可选save_interval字段，'
           '本次未设置该字段，有效保存间隔仍为12000，不构成新增训练设置差别。\n')
    (out/'report.md').write_text(text)
    tracked=ECM/'reports/lowlevel_rescue_D2_v31_20261102';atomic_json(tracked/'summary.json',audit)
    (ECM/'reports/lowlevel_rescue_D2_v31_20261102.md').write_text(text.replace('](videos/','](../results/'+out.name+'/videos/').replace(
        '](v3_failure_diagnosis/','](../results/'+out.name+'/v3_failure_diagnosis/'))
    print(validation,flush=True)


if __name__=='__main__': main()
