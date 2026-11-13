"""v3.1 dispatch. v3 execution/holds remain unchanged in canonical_solver.py."""
import json
from pathlib import Path
import numpy as np
import trimesh

from . import canonical_solver as v3
from .canonical_planning import result_digest
from .canonical_v31_planning import SOLVER_VERSION,canonical_v31_plan

configure=v3.configure
move=v3.move
pick=v3.pick
run=v3.run


def initialize(solver):
    v3.initialize(solver);solver.solver_version=SOLVER_VERSION


def refresh_obstacles(solver,skip=()):
    from .solver import segment_mesh,surface_points,POINT_RADIUS
    L=solver.L;tx,ty=L.trunk['xy']
    meshes=[('tree','trunk',segment_mesh([tx,ty,0],[tx,ty,L.trunk['top']],L.trunk['radius']))]
    meshes += [('tree',f'twig_{i}',segment_mesh(t['start'],t['end'],t['radius'],capsule=True)) for i,t in enumerate(L.twigs)]
    positions=solver.u.fruit_positions()
    for i,fruit in enumerate(L.fruits):
        if i not in skip and not solver.u.attached[i]:
            meshes.append(('other_fruit',f'fruit_{i}',trimesh.creation.icosphere(subdivisions=2,radius=fruit['radius']).apply_translation(positions[i])))
    c=np.asarray(L.basket['center']);h=L.basket['inner_half'];H=L.basket['height']
    for i,(sx,sy) in enumerate(((1,0),(-1,0),(0,1),(0,-1))):
        ext=[.006 if sx else 2*h,.006 if sy else 2*h,H]
        meshes.append(('basket',f'basket_wall_{i}',trimesh.creation.box(ext).apply_translation(c+[sx*h,sy*h,H/2])))
    parts=[];clouds=[];start=0
    for category,name,mesh in meshes:
        points=surface_points(mesh);clouds.append(points)
        parts.append({'category':category,'name':name,'start':start,'stop':start+len(points)});start+=len(points)
    solver.canonical_point_cloud=np.vstack(clouds);solver.canonical_obstacle_parts=parts
    solver.planner.update_point_cloud(solver.canonical_point_cloud,radius=POINT_RADIUS)


def plan(solver,pose,attach):
    current=solver.qpos().astype(np.float64);goal=np.r_[pose.p,pose.q];phase=solver.canonical_phase
    target=solver.canonical_labels.get('target');snapshot=None
    if solver.canonical_snapshot_dir is not None:
        snapshot=solver.canonical_snapshot_dir/f'plan_{len(solver.trajectory_audit):04d}_{phase}.npz'
        np.savez_compressed(snapshot,current=current,reference=solver.canonical_reference,goal_pose=goal,
            point_cloud=solver.canonical_point_cloud,fruit_center=solver.u.fruit_positions()[target],
            metadata=json.dumps({'phase':phase,'attach':attach,'attached_radius':solver.canonical_attached_radius,
                'point_radius':.008,'dt':solver.dt,'ik_seed':solver.ik_seed,'ik_candidates':solver.ik_candidates,
                'urdf':solver.u.agent.urdf_path,'user_link_names':solver.planner.user_link_names,
                'user_joint_names':solver.planner.user_joint_names,'obstacle_parts':solver.canonical_obstacle_parts,
                'velocity_limits':np.asarray(solver.planner.joint_vel_limits).tolist(),
                'acceleration_limits':np.asarray(solver.planner.joint_acc_limits).tolist(),
                'timestep':solver.canonical_step_count,'labels':solver.canonical_labels,
                'layout_target_position':solver.L.fruits[target]['pos'],'solver_version':SOLVER_VERSION}))
    result,trace=canonical_v31_plan(solver.planner,goal,current,solver.canonical_reference,solver.dt,attach,phase,
        solver.canonical_point_cloud,solver.ik_seed,solver.ik_candidates,solver.canonical_obstacle_parts)
    trace.update(snapshot=str(snapshot) if snapshot else None,timestep=solver.canonical_step_count,
        result_sha256=result_digest(result),target=target,layout_target_position=solver.L.fruits[target]['pos'])
    solver.trajectory_audit.append(trace)
    return result
