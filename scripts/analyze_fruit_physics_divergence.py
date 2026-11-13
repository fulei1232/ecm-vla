#!/usr/bin/env python3
"""Locate first exact and material divergence; never confuse actions with physics."""
import argparse
import itertools
from pathlib import Path

import numpy as np

from v_a_common import ECM,atomic_json,read_json,timestamp
from report_fruit_expert_v2 import trajectory


def first_divergence(first,second,threshold=0.):
    for step,(a,b) in enumerate(zip(first,second)):
        differences=[]
        for key in sorted(set(a)&set(b)):
            if not (key.startswith('robot_') or key.startswith('fruit_') or key.startswith('leaf_')): continue
            aa,bb=np.asarray(a[key],dtype=np.float64),np.asarray(b[key],dtype=np.float64)
            if aa.shape!=bb.shape:
                differences.append({'field':key,'object':key.split('_')[0],'reason':'shape mismatch'})
                continue
            if not aa.size: continue
            diff=np.abs(aa-bb)
            if np.any(diff>threshold):
                index=list(np.unravel_index(int(np.argmax(diff)),diff.shape))
                differences.append({'field':key,'object':key.split('_')[0],'max_abs_difference':float(diff.max()),
                    'max_index':[int(i) for i in index],'first_value':float(aa[tuple(index)]),
                    'second_value':float(bb[tuple(index)]),
                    'units':('rad' if index[0]<7 else 'm') if key=='robot_qpos' else
                            ('rad/s' if index[0]<7 else 'm/s') if key=='robot_qvel' else
                            'm_and_quaternion_components' if 'pose' in key else 'field units'})
        if differences:
            return {'state_index':step,'after_control_steps':step,'next_control_step_1based':step+1,
                    'differences':differences}
    return None


def original_d2_cases():
    cases=[]
    for e in (0,1,3):
        runs=[]
        for repeat in range(3):
            path=ECM/f'results/fruit_expert_v2_audit_20261102/determinism/occluded_ripe/ep{e}/repeat{repeat}/result.json'
            row,q,a=trajectory(path)
            gt=[__import__('json').loads(line) for line in (path.parent/row['gt']).read_text().splitlines()]
            states=[{'robot_qpos':qq.tolist()+r['gripper_qpos'],'robot_tcp_pose':r['tcp_pos']+r['tcp_quat'],
                'fruit_pose':[f['pos'] for f in r['fruits']],
                'fruit_attached':[f['attached'] for f in r['fruits']],
                'fruit_in_basket':[f['in_basket'] for f in r['fruits']],
                'leaf_push':r['occluder_push']} for qq,r in zip(q,gt)]
            runs.append((states,a,row,gt))
        pairs=[]
        for i,j in itertools.combinations(range(3),2):
            action_diff=np.max(np.abs(runs[i][1][:min(len(runs[i][1]),len(runs[j][1]))]-
                                      runs[j][1][:min(len(runs[i][1]),len(runs[j][1]))]),axis=1)
            changed=np.flatnonzero(action_diff>0)
            pairs.append({'pair':[i,j],'first_state_difference_exact':first_divergence(runs[i][0],runs[j][0]),
                'first_state_difference_above_1e_6':first_divergence(runs[i][0],runs[j][0],1e-6),
                'first_action_difference_timestep':int(changed[0]) if len(changed) else None,
                'first_action_phase':[runs[k][3][int(changed[0])]['phase'] for k in (i,j)] if len(changed) else None,
                'first_action_pick_index':[runs[k][3][int(changed[0])]['pick_index'] for k in (i,j)] if len(changed) else None,
                'first_action_difference_rad':float(action_diff[changed[0]]) if len(changed) else None})
        cases.append({'episode':e,'pairs':pairs,'note':'D2 sidecars have leaf_push but no leaf pose/velocity; replay below records actual poses'})
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--original-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    original=original_d2_cases();atomic_json(a.out/'original_d2_divergence.json',original)
    if a.original_only: return
    cases=[]
    for e in (0,1,3):
        paths=[a.out/f'ep{e}/repeat{j}/trace.json' for j in range(3)]
        if not all(p.exists() for p in paths):
            cases.append({'episode':e,'complete':False,'missing':[str(p) for p in paths if not p.exists()]});continue
        runs=[read_json(p) for p in paths]
        assert len({r['action_sha256'] for r in runs})==1
        pairs=[{'pair':[i,j],'first_exact':first_divergence(runs[i]['states'],runs[j]['states']),
                'first_above_1e_6':first_divergence(runs[i]['states'],runs[j]['states'],1e-6)}
                for i,j in itertools.combinations(range(3),2)]
        cases.append({'episode':e,'complete':True,'fixed_actions_sha256':runs[0]['action_sha256'],
            'source':runs[0]['source'],'steps':runs[0]['steps'],'pairs':pairs,
            'final_success':[r['final_success'] for r in runs]})
    atomic_json(a.out/'summary.json',{'generated_utc':timestamp(),'report_only':True,
        'original_d2_replanning_runs':original,'fixed_action_replay':cases,
        'threshold_note':'exact IEEE comparison plus separate >1e-6 field-unit comparison; qvel rad/s, pose translation m and quaternion components; state index k is after k actions'})


if __name__=='__main__': main()
