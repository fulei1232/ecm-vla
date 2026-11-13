#!/usr/bin/env python3
"""Revision3 first step: diagnose all24 v3 failures before changing the expert."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sys

import h5py
import mplib
import numpy as np
import trimesh

from v_a_common import ECM,atomic_json,read_json,sha256,timestamp
sys.path.insert(0,str(ECM))
from audit_fruit_canonical_planning import load_snapshot,make_planner
from fruit_picking.canonical_planning import HIGH_READY_ARM,linear_samples
from fruit_picking.solver import segment_mesh,surface_points


def point_partitions(layout,attached,target,cloud):
    """Recover exact original point-cloud ranges, without guessing nearest fruit."""
    tx,ty=layout['trunk']['xy'];trunk=layout['trunk']
    meshes=[('tree','trunk',segment_mesh([tx,ty,0],[tx,ty,trunk['top']],trunk['radius']))]
    meshes += [('tree',f'twig_{i}',segment_mesh(t['start'],t['end'],t['radius'],capsule=True)) for i,t in enumerate(layout['twigs'])]
    for i,f in enumerate(layout['fruits']):
        if i!=target and not attached[i]:
            meshes.append(('other_fruit',f'fruit_{i}',trimesh.creation.icosphere(subdivisions=2,radius=f['radius']).apply_translation(np.asarray(f['pos'],dtype=np.float32))))
    c=np.asarray(layout['basket']['center']);h=layout['basket']['inner_half'];H=layout['basket']['height']
    for i,(sx,sy) in enumerate(((1,0),(-1,0),(0,1),(0,-1))):
        ext=[.006 if sx else 2*h,.006 if sy else 2*h,H]
        meshes.append(('basket',f'basket_wall_{i}',trimesh.creation.box(ext).apply_translation(c+[sx*h,sy*h,H/2])))
    result=[];start=0
    for category,name,mesh in meshes:
        n=len(surface_points(mesh));result.append({'category':category,'name':name,'start':start,'stop':start+n});start+=n
    assert start==len(cloud),(start,len(cloud),target,attached)
    return result


def collision_details(planner,q,cloud,parts,radius,attach):
    planner.update_point_cloud(cloud,radius=radius)
    own=planner.check_for_self_collision(qpos=q)
    env=planner.check_for_env_collision(qpos=q,with_point_cloud=True,use_attach=attach)
    result=[]
    if own:
        result.append({'category':'self_collision','pairs':[[str(c.link_name1),str(c.link_name2)] for c in own]})
    if env:
        for part in parts:
            planner.update_point_cloud(cloud[part['start']:part['stop']],radius=radius)
            collisions=planner.check_for_env_collision(qpos=q,with_point_cloud=True,use_attach=attach)
            if collisions:
                result.append({**part,'pairs':[[str(c.link_name1),str(c.link_name2)] for c in collisions]})
    planner.update_point_cloud(cloud,radius=radius)
    return result


def tcp(planner,q):
    model=planner.pinocchio_model;model.compute_forward_kinematics(q)
    return np.asarray(model.get_link_pose(planner.link_name_2_idx[planner.move_group]))[:3]


def first_collision(planner,arms,current,cloud,parts,radius,attach):
    for index,arm in enumerate(arms):
        q=current.copy();q[:7]=arm
        own=planner.check_for_self_collision(qpos=q)
        env=planner.check_for_env_collision(qpos=q,with_point_cloud=True,use_attach=attach)
        if own or env or abs(q[6])>np.pi:
            return {'waypoint_index':index,'arm_qpos':q[:7].tolist(),'tcp_position':tcp(planner,q).tolist(),
                'q7_outside_pi':bool(abs(q[6])>np.pi),
                'objects':collision_details(planner,q,cloud,parts,radius,attach)}
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    root=ECM/'results/fruit_expert_v3_audit_20261102/demos_settle2/rm75b_eg2/train'
    cases=[]
    for path in sorted(root.glob('*/attempts/ep*_seed*.json')):
        attempt=read_json(path)
        if attempt['ok']: continue
        dest=out/f"{attempt['task']}_ep{attempt['episode']}_seed{attempt['seed']}.json"
        if dest.exists(): cases.append(read_json(dest));continue
        trace=next(t for t in reversed(attempt['planning_audit']) if t.get('snapshot'))
        snapshot=Path(trace['snapshot']);data,meta=load_snapshot(snapshot);planner=make_planner(meta)
        with h5py.File(path.parent.parent/attempt['failed_h5']) as f:
            raw=f[f"episode_{attempt['episode']}/setup/layout_json"][()]
            layout=json.loads(raw.decode() if isinstance(raw,bytes) else raw)
        gt=[json.loads(line) for line in (path.parent.parent/attempt['failed_gt']).read_text().splitlines()]
        attached=[r['attached'] for r in gt[-1]['fruits']]
        target=meta['labels']['target'];parts=point_partitions(layout,attached,target,data['point_cloud'])
        current=data['current'];goal=current.copy();goal[:7]=trace['goal_arm_qpos']
        checks={str(radius):collision_details(planner,goal,data['point_cloud'],parts,radius,meta['attach']) for radius in (.008,.018)}
        rest=current.copy();rest[:7]=data['reference'][:7]
        high=current.copy();high[:7]=HIGH_READY_ARM
        methods=[]
        for name,nodes in [('straight',[current,goal]),('rest',[current,rest,goal]),('high_ready',[current,high,goal])]:
            planner.update_point_cloud(data['point_cloud'],radius=.018)
            arms=np.concatenate([linear_samples(x,y,np.arange(7)) for x,y in zip(nodes,nodes[1:])])
            collision=first_collision(planner,arms,current,data['point_cloud'],parts,.018,meta['attach'])
            stage='joint_interpolation'
            if collision is None:
                try:
                    _,positions,_,_,_=planner.TOPP(arms,meta['dt'])
                    collision=first_collision(planner,positions,current,data['point_cloud'],parts,.018,meta['attach'])
                    stage='TOPP_output'
                except RuntimeError as exc: stage='TOPP_error:'+str(exc)
            methods.append({'method':name,'checked_waypoints':len(arms),'check_stage':stage,'first_collision':collision})
        planner.update_point_cloud(data['point_cloud'],radius=.018)
        try:
            rrt=planner.plan_qpos_to_qpos([goal],current.copy(),time_step=meta['dt'],use_point_cloud=True,
                                        use_attach=meta['attach'],planning_time=2)
            rr={'method':'rrt_replanned_for_diagnosis','status':rrt['status'],'first_collision':None,
                'scope':'Original failed RRT output was not persisted in v3; this is one new diagnostic planning call, not its original trajectory.'}
            if rrt['status']=='Success':
                positions=np.vstack([rrt['position'],goal[:7]])
                dense=np.concatenate([linear_samples(np.r_[x,current[7:]],np.r_[y,current[7:]],np.arange(7)) for x,y in zip(positions,positions[1:])])
                rr['first_collision']=first_collision(planner,dense,current,data['point_cloud'],parts,.018,meta['attach'])
                np.savez_compressed(dest.with_suffix('.rrt.npz'),positions=positions,dense=dense)
        except RuntimeError as exc: rr={'method':'rrt_replanned_for_diagnosis','status':str(exc),'first_collision':None}
        methods.append(rr)
        case={'attempt':str(path),'attempt_sha256':sha256(path),'snapshot':str(snapshot),'snapshot_sha256':sha256(snapshot),
            'task':attempt['task'],'episode':attempt['episode'],'seed':attempt['seed'],'failure_phase':meta['phase'],
            'pick_index':meta['labels']['pick_index'],'target':target,'target_fruit_position':data['fruit_center'].tolist(),
            'target_joint_configuration':goal[:7].tolist(),'target_collisions_by_radius':checks,'methods':methods,
            'point_cloud_partitions':parts,'target_fruit_excluded':True,'leaf_collision_geometry_present':False}
        atomic_json(dest,case);cases.append(case)
        print({'task':attempt['task'],'seed':attempt['seed'],'phase':meta['phase'],'target008':bool(checks['0.008']),
               'target018':bool(checks['0.018'])},flush=True)
    assert len(cases)==24
    summary={'generated_utc':timestamp(),'cases':cases,'failed_attempts':24,
        'failure_phases':dict(Counter(c['failure_phase'] for c in cases)),
        'targets_colliding_by_radius':{r:sum(bool(c['target_collisions_by_radius'][r]) for c in cases) for r in ('0.008','0.018')},
        'native_rrt_source':{'sha256':hashlib.sha256(inspect.getsource(mplib.Planner.plan_qpos_to_qpos).encode()).hexdigest(),
            'finding':'Native plan_qpos_to_qpos does not update_point_cloud or override its radius; it uses the existing planning_world. No observed automatic reset to .008 in this Python entry point.'},
        'limits':'Leaves have no collision geometry and target fruit is excluded in the original snapshots. Original rejected RRT paths were not saved; replanned diagnostic RRT paths are explicitly labeled.'}
    atomic_json(out/'summary.json',summary)
    lines=['v3全部24次失败的碰撞诊断已完成，尚未修改v3.1专家。',
        f"失败阶段：{summary['failure_phases']}。",f"目标姿态碰撞数：{summary['targets_colliding_by_radius']}。",
        '原生RRT Python入口未发现自动把点云半径改回0.008m的代码；不能把该推断当作已确认原因。',
        '原始失败RRT路径没有保存；诊断重规划结果明确区分，原快照和产物均不改动。','',
        '| 任务/种子 | 阶段 | 目标0.008m碰撞 | 目标0.018m碰撞 | 各方法首次碰撞 |','|---|---|---|---|---|']
    for c in cases:
        short=[{'method':m['method'],'first':m.get('first_collision'),'status':m.get('status')} for m in c['methods']]
        lines.append(f"| {c['task']}/{c['seed']} | {c['failure_phase']} | {c['target_collisions_by_radius']['0.008']} | {c['target_collisions_by_radius']['0.018']} | {short} |")
    lines+=['',summary['limits'],'','所有数字及逐路点关节角、TCP位置、碰撞链接和对象分组见summary.json及逐尝试JSON。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
