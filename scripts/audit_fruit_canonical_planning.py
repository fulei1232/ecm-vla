#!/usr/bin/env python3
"""Repeat native planning from saved inputs; assess perturbations without physics."""
import argparse
import json
from pathlib import Path
import sys

import mplib
import numpy as np

from v_a_common import ECM,atomic_json,read_json,sha256,timestamp
sys.path.insert(0,str(ECM))
from fruit_picking.canonical_planning import canonical_plan,result_digest
from fruit_picking.solver import PRE_GRASP,horizontal_grasp_pose


def load_snapshot(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k].copy() for k in ('current','reference','goal_pose','point_cloud','fruit_center')},json.loads(str(data['metadata']))


def make_planner(meta):
    planner=mplib.Planner(urdf=meta['urdf'],srdf=meta['urdf'].replace('.urdf','.srdf'),
        user_link_names=meta['user_link_names'],user_joint_names=meta['user_joint_names'],move_group='tcp')
    planner.set_base_pose(np.array([0,0,0,1.,0,0,0]))
    planner.joint_vel_limits=np.asarray(meta['velocity_limits'])
    planner.joint_acc_limits=np.asarray(meta['acceleration_limits'])
    if meta['attach']:
        planner.update_attached_sphere(meta['attached_radius'],[0,0,0,1.,0,0,0])
    return planner


def replan(planner,data,meta,offset=None):
    current,pose=data['current'].copy(),data['goal_pose'].copy()
    if offset is not None:
        fruit=data['fruit_center']+offset
        direction=np.array([fruit[0],fruit[1],0.])/np.linalg.norm(fruit[:2])
        pre=horizontal_grasp_pose(fruit-PRE_GRASP*direction,direction)
        pose=np.concatenate([pre.p,pre.q])
        # Each measured arm coordinate is perturbed by exactly 1e-3 radians.
        current[:7]+=np.where(np.arange(7)%2,1.,-1.)*.001*(1 if np.sum(offset)>=0 else -1)
    planner.update_point_cloud(data['point_cloud'],radius=meta['point_radius'])
    return canonical_plan(planner,pose,current,data['reference'],meta['dt'],meta['attach'],
                          meta['phase'],meta['ik_seed'],meta['ik_candidates'])


def saved_instances(root):
    # Deterministic interleaving covers both tasks, then all picks in each episode.
    result={phase:[] for phase in ('reach','carry')}
    for e in range(100):
        for task in ('count','occluded_ripe'):
            path=root/task/f'ep{e}_manifest.json'
            if not path.exists(): continue
            man=read_json(path)
            for trace in man['planning_audit']:
                phase=trace.get('phase')
                if phase in result and trace.get('snapshot') and trace.get('status')=='Success':
                    result[phase].append({'snapshot':trace['snapshot'],'manifest':str(path),
                        'manifest_sha256':sha256(path),'recorded_result_sha256':trace['result_sha256'],
                        'task':task,'episode':e,'timestep':trace['timestep']})
    return result


def continuity_case(base_trace,other_traces):
    traces=[base_trace]+other_traces
    valid=all(row.get('status')=='Success' and row.get('goal_arm_qpos') is not None for row in traces)
    delta=float(np.max(np.ptp(np.asarray([t['goal_arm_qpos'] for t in traces]),axis=0))) if valid else None
    methods=[row.get('method') for row in traces]
    # Also report fixed waypoint changes, although §10.4 groups both as detour.
    return {'passed':bool(valid and delta<.05 and len(set(methods))==1),
            'max_pairwise_goal_difference_rad':delta,'methods':methods,
            'waypoints':[row.get('waypoint') for row in traces],
            'goals':[row.get('goal_arm_qpos') for row in traces]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    instances=saved_instances(a.data)
    deterministic=[]
    for phase in ('reach','carry'):
        for i,row in enumerate(instances[phase][:50]):
            dest=a.out/f'determinism_{phase}_{i:03d}.json'
            if dest.exists(): deterministic.append(read_json(dest));continue
            path=Path(row['snapshot']);data,meta=load_snapshot(path);planner=make_planner(meta)
            runs=[replan(planner,data,meta) for _ in range(3)]
            hashes=[result_digest(result) for result,trace in runs]
            traces=[trace for result,trace in runs]
            item={**row,'snapshot_sha256':sha256(path),'phase':phase,'hashes':hashes,'traces':traces,
                  'matches_recorded_path':hashes[0]==row['recorded_result_sha256'],
                  'passed':all(result is not None for result,trace in runs) and len(set(hashes))==1 and
                           all(trace==traces[0] for trace in traces)}
            atomic_json(dest,item);deterministic.append(item)
            print({'audit':'determinism','phase':phase,'case':i,'passed':item['passed']},flush=True)
    # Five cube-corner perturbations cover both signs on every Cartesian axis.
    offsets=[np.array(v)*.002 for v in ([1,1,1],[-1,-1,-1],[1,-1,1],[-1,1,-1],[1,1,-1])]
    continuous=[]
    for i,row in enumerate(instances['reach'][:100]):
        dest=a.out/f'continuity_{i:03d}.json'
        if dest.exists(): continuous.append(read_json(dest));continue
        path=Path(row['snapshot']);data,meta=load_snapshot(path);planner=make_planner(meta)
        _,base=replan(planner,data,meta)
        traces=[replan(planner,data,meta,offset)[1] for offset in offsets]
        item={**row,'snapshot_sha256':sha256(path),'perturbation_offsets_m':[x.tolist() for x in offsets],
              'joint_perturbation_abs_rad':.001,**continuity_case(base,traces),'traces':[base]+traces}
        atomic_json(dest,item);continuous.append(item)
        print({'audit':'continuity','case':i,'passed':item['passed']},flush=True)
    exact={phase:sum(r['phase']==phase for r in deterministic)==50 for phase in ('reach','carry')}
    summary={'generated_utc':timestamp(),'determinism':{'cases':deterministic,'coverage':exact,
        'passed':all(exact.values()) and all(r['passed'] for r in deterministic),
        'matching_original_paths':sum(r['matches_recorded_path'] for r in deterministic)},
        'continuity':{'cases':continuous,'instances':len(continuous),'passed_instances':sum(r['passed'] for r in continuous),
        'fraction':sum(r['passed'] for r in continuous)/len(continuous) if continuous else None,
        'passed':len(continuous)==100 and sum(r['passed'] for r in continuous)>=98,
        'definition':'baseline plus five +/-2mm fruit-center offsets; recompute azimuth/pre-grasp pose; alternating +/-1e-3rad on all measured arm coordinates; max pairwise goal range <.05rad, method invariant'}}
    atomic_json(a.out/'summary.json',summary)


if __name__=='__main__': main()
