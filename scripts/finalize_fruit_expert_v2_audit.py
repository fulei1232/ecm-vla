#!/usr/bin/env python3
"""Independent D2 artifact review; no expert reruns, training or controller changes."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from report_fruit_expert_v2 import trajectory
from fruit_expert_audit_stats import load_episode
from v_a_common import ECM,atomic_json,read_json,sha256,timestamp


def main():
    out=ECM/'results/fruit_expert_v2_audit_20261102'
    status=read_json(out/'status.json');plan=read_json(out/'plan.json');audit=read_json(out/'audit.json')
    assert status['state'] in ('awaiting_phase2_review','passed_phase2')
    for path,digest in plan['source_sha256'].items(): assert sha256(Path(path))==digest,path
    assert sha256(ECM/'reports/split_registry_20261102.json')==plan['frozen_registry_sha256']
    assert sha256(ECM/'reports/HANDOFF_lowlevel_rescue_20261102.md')==plan['handoff_sha256']
    assert audit['nearest']['episodes']==200 and len(audit['videos'])==4
    source_files=0;frames=0;outside_states=0;outside_commands=0;by_phase=Counter();witnesses=[]
    for record in audit['nearest']['records']:
        path=Path(record['manifest']);man=read_json(path)
        assert sha256(path)==record['manifest_sha256']
        assert man['solver_version']=='nearest_v2_20261102'
        for key in ('h5','gt'):
            assert sha256(path.parent/man[key])==man[key+'_sha256'];source_files+=1
        _,q,a,supervised,gt=load_episode(path)
        frames+=len(q);bad=np.flatnonzero(np.abs(q[:,6])>np.pi);bad_commands=np.flatnonzero(np.abs(a[:,6])>np.pi)
        outside_states+=len(bad);outside_commands+=len(bad_commands)
        by_phase.update(gt[i]['phase'] for i in bad)
        if len(bad):
            i=int(bad[0]);witnesses.append({'manifest':str(path),'timestep':i,'phase':gt[i]['phase'],
                                          'q7_state':float(q[i,6]),'q7_command':float(a[i,6])})
    assert frames==audit['nearest']['state_samples']
    assert outside_states==audit['nearest']['q7_states_outside_pi']['numerator']
    cases=[];max_common=0.
    for case in audit['determinism']['cases']:
        runs=[trajectory(Path(p)) for p in case['results']]
        assert all(q is not None for r,q,a in runs)
        shared=min(len(q) for r,q,a in runs)
        difference=max(max(float(np.max(np.abs(q[:shared]-runs[0][1][:shared]))),
                           float(np.max(np.abs(a[:shared,:7]-runs[0][2][:shared,:7])))) for r,q,a in runs[1:])
        max_common=max(max_common,difference)
        cases.append({'task':case['task'],'episode':case['episode'],'lengths':[len(q) for r,q,a in runs],
                      'shared_control_steps':shared,'max_common_prefix_difference_rad':difference,'passed':case['passed'],
                      'methods_per_repeat':[dict(Counter(t.get('method') for t in r.get('planning_audit',[]) if t.get('status')=='Success')) for r,q,a in runs]})
    for video in audit['videos']:
        assert Path(video['video']).stat().st_size>0 and Path(video['frames']).stat().st_size>0
    validation={'validated_utc':timestamp(),'manifests':200,'payload_sha256_files':source_files,
                'runtime_sources_unchanged':True,'frozen_registry_unchanged':True,'full_determinism_runs':30,
                'deterministic_layouts':{'numerator':sum(c['passed'] for c in cases),'denominator':10},
                'max_common_prefix_difference_rad':max_common,'determinism_details':cases,
                'q7_state_outside_pi':{'numerator':outside_states,'denominator':frames},
                'q7_command_outside_pi':{'numerator':outside_commands,'denominator':frames},
                'q7_state_outside_by_phase':dict(by_phase),'q7_witnesses':witnesses,
                'videos_verified':4,'D1_old_metrics_unchanged':audit['d1_recomputed_with_revision1_metrics']['old_metrics_unchanged']}
    atomic_json(out/'independent_validation.json',validation)
    # Keep the original automatic output. Its 0.0 maximum covered only equal-
    # length cases; it must not be presented as the full determinism result.
    if not (out/'audit_before_independent_review.json').exists(): atomic_json(out/'audit_before_independent_review.json',audit)
    audit['independent_validation']=validation
    audit['determinism']['same_length_cases_max_difference_rad']=audit['determinism'].get(
        'same_length_cases_max_difference_rad',audit['determinism']['max_difference_rad'])
    if any(len(set(case['lengths']))>1 for case in cases): audit['determinism']['max_difference_rad']=None
    audit['determinism']['max_common_prefix_difference_rad']=max_common
    audit['determinism']['passed_layouts']=validation['deterministic_layouts']
    audit['determinism']['max_difference_scope']='Original max_difference_rad includes only identical-shape runs; length-mismatched layouts fail regardless. Common-prefix max and lengths are independently reported.'
    atomic_json(out/'audit.json',audit);atomic_json(out/'summary.json',audit)
    text=(out/'report.md').read_text()
    lines=text.splitlines()
    lines[1]='已确认：200条被接受示范来自217次尝试；逐尝试成功率 count=100/111、occluded_ripe=100/106，均未达95%。'
    lines[2]='已确认：确定性仅7/10布局通过，3个布局轨迹长度不一致；抓取q7偏离为279/547，q7越界状态286/91349。'
    lines[3]='已通过：reach/carry的q1摆幅P90为0.766/0.690rad，q7摆幅>π均为0/547；修订1补充项已交付。'
    lines[4]='决策：按§4.2停下等用户确认，不进入阶段3；平台保持A0，v1作废。'
    text='\n'.join(lines)+'\n'
    text=text.replace('| 10布局×3次轨迹最大差异 rad | 未审计 | 0.0 | <1e-6；False |',
                      f'| 10布局×3次确定性 | 未审计 | 7/10通过；共同控制步最大差异 {max_common:.9f} rad，另有长度不一致 | 全部10布局<1e-6且长度一致；不通过 |')
    text+='\n## 独立复核与确定性数字说明\n\n'
    text+=(f'复核200个manifest、{source_files}个内容文件哈希及全部运行源文件；冻结划分未变。'
           f'30次重复都完成，但只有7/10布局满足确定性，3/10轨迹长度不一致；共同控制步最大差异为{max_common:.9f}rad。'
           '自动汇总原先的0.0仅来自等长的7个布局，不能代表全部10个布局；原始自动汇总已另存audit_before_independent_review.json。'
           '独立复核没有改动作、轨迹、IK权重、控制器或审计判定。\n\n')
    text+='| 布局 | 三次长度 | 共同控制步最大差异 rad |\n|---|---|---:|\n'
    for c in cases:
        if not c['passed']: text+=f"| {c['task']} ep{c['episode']} | {c['lengths']} | {c['max_common_prefix_difference_rad']:.9f} |\n"
    text+=(f'\nq7命令超出[−π,π]为{outside_commands}/{frames}，实测状态超出为{outside_states}/{frames}；'
           f'实测越界的阶段计数为{dict(by_phase)}。逐回合首个样例及来源在independent_validation.json。'
           '该诊断用于区分规划目标与实测跟踪，不替代既定审计；不能仅凭越界断言都是PD滞后造成。\n')
    (out/'report.md').write_text(text)
    tracked=ECM/'reports/lowlevel_rescue_D2_20261102';tracked.mkdir(exist_ok=True)
    atomic_json(tracked/'summary.json',audit)
    versioned=text.replace('](episode_lengths.png)','](../results/fruit_expert_v2_audit_20261102/episode_lengths.png)').replace('](videos/','](../results/fruit_expert_v2_audit_20261102/videos/')
    (ECM/'reports/lowlevel_rescue_D2_20261102.md').write_text(versioned+'\n相对数据路径的基目录：`results/fruit_expert_v2_audit_20261102`。\n')
    state=read_json(ECM/'reports/lowlevel_rescue_status.json')
    state.update(phase=2,state='awaiting_phase2_review',updated_utc=timestamp(),platform={'condition':'A0','joint_guard':'off','exec_horizon':16},deprecated_joint_guards=['v1'])
    state['links'].update(D2_versioned_report='reports/lowlevel_rescue_D2_20261102.md',
                          D2_versioned_summary='reports/lowlevel_rescue_D2_20261102/summary.json',D2_validation=str(out/'independent_validation.json'))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json',state)
    print(validation,flush=True)


if __name__=='__main__': main()
