"""Revision3 expert: zoned clearance, target-collision-only IK replacement, tuck."""
import math
import numpy as np

from .canonical_planning import HIGH_READY_ARM,canonical_candidates,canonical_goal,checked_polyline,linear_samples
from .joint_planning import DeterministicIKModel

SOLVER_VERSION='canonical_v3.1_20261102'
BASE_RADIUS=.008
MIDDLE_RADIUS=.018
TERMINAL_DISTANCE=.05
# Native FK probe: max link-origin radius .215m, TCP height .626m at q1=0.
TUCK_Q2_Q7=np.asarray([-1.,-.0061,2.,-.181,.0534,-1.3933])


class ZonedGeometry:
    """Same source cloud, base clearance in the terminal5cm; inflated elsewhere."""
    def __init__(self,planner,cloud,goal_position,attach,parts=None):
        self.native=planner;self.cloud=np.asarray(cloud);self.goal_position=np.asarray(goal_position)
        self.attach=attach;self.parts=parts or [];self.radius=None;self.first_collision=None;self.checks=0
    def __getattr__(self,name): return getattr(self.native,name)
    def set_radius(self,radius,force=False):
        if force or radius!=self.radius:
            self.native.update_point_cloud(self.cloud,radius=radius);self.radius=radius
    def tcp(self,q):
        model=self.native.pinocchio_model;model.compute_forward_kinematics(q)
        return np.asarray(model.get_link_pose(self.native.link_name_2_idx[self.native.move_group]))[:3]
    def radius_at(self,q):
        return BASE_RADIUS if np.linalg.norm(self.tcp(q)-self.goal_position)<=TERMINAL_DISTANCE else MIDDLE_RADIUS
    def witness(self,q,kind,collisions):
        if self.first_collision is None:
            objects=[]
            if kind=='environment':
                radius=self.radius
                for part in self.parts:
                    self.native.update_point_cloud(self.cloud[part['start']:part['stop']],radius=radius)
                    hits=self.native.check_for_env_collision(qpos=q,with_point_cloud=True,use_attach=self.attach)
                    if hits: objects.append({'category':part['category'],'name':part['name']})
                self.set_radius(radius,force=True)
            self.first_collision={'check_index':self.checks,'kind':kind,'arm_qpos':q[:7].tolist(),
                'tcp_position':self.tcp(q).tolist(),'radius':self.radius,'objects':objects,
                'links':[[str(getattr(hit,'link_name1','')),str(getattr(hit,'link_name2',''))] for hit in collisions]}
    def check_for_self_collision(self,qpos):
        collisions=self.native.check_for_self_collision(qpos=qpos)
        if collisions: self.witness(qpos,'self_collision',collisions)
        return collisions
    def check_for_env_collision(self,qpos,**kwargs):
        self.set_radius(self.radius_at(qpos));self.checks+=1
        collisions=self.native.check_for_env_collision(qpos=qpos,**kwargs)
        if collisions: self.witness(qpos,'environment',collisions)
        return collisions
    def clear_witness(self): self.first_collision=None;self.checks=0


def target_collision(planner,goal,attach):
    own=planner.check_for_self_collision(qpos=goal)
    env=planner.check_for_env_collision(qpos=goal,with_point_cloud=True,use_attach=attach)
    return bool(own or env)


def select_goal(planner,pose,reference,geometry,seed,candidates):
    geometry.set_radius(BASE_RADIUS)
    first,trace=canonical_goal(planner,pose,reference,seed,candidates)
    trace.update(goal_replacement_allowed_only='target_collision_at_008m',goal_replaced=False)
    if first is None: return None,trace
    if not target_collision(planner,first,geometry.attach): return first,trace
    # This is the sole extra source of IK candidates relative to v3.
    trace['initial_target_collision_008']=True
    idx=np.asarray(planner.move_group_joint_indices);model=planner.pinocchio_model
    link=planner.link_name_2_idx[planner.move_group]
    sampler=DeterministicIKModel(model,planner.joint_limits,idx,reference,seed)
    raw=[first]
    for _ in range(candidates):
        result=model.compute_IK_CLIK(link,pose,sampler.get_random_configuration(),[])
        q=np.asarray(result[0],dtype=np.float64).copy()
        if not planner.wrap_joint_limit(q): continue
        ranked=canonical_candidates([q],reference,idx,planner.joint_limits)
        if not ranked: continue
        q=ranked[0];model.compute_forward_kinematics(q);reached=np.asarray(model.get_link_pose(link))
        if planner.distance_6D(pose[:3],pose[3:],reached[:3],reached[3:])<1e-3: raw.append(q)
    ranked=canonical_candidates(raw,reference,idx,planner.joint_limits)
    trace['target_candidate_checks']=[]
    for rank,q in enumerate(ranked,1):
        collides=target_collision(planner,q,geometry.attach)
        trace['target_candidate_checks'].append({'rank':rank,'collides_at_008':collides,'arm_qpos':q[idx].tolist()})
        if not collides:
            trace.update(goal_replaced=not np.array_equal(q[idx],first[idx]),selected_target_rank=rank,
                         replacement_reason='initial_target_collision_008')
            return q,trace
    return None,trace


def tuck_waypoint(current,goal,reference,phase,indices):
    tuck=current.copy();tuck[indices[1:]]=TUCK_Q2_Q7
    tuck[indices[6]]=reference[indices[6]]
    tuck[indices[0]]=goal[indices[0]] if phase=='reach' else current[indices[0]]
    return tuck


def validate_positions(geometry,positions,current,indices):
    for a,b in zip(positions,positions[1:]):
        start,end=current.copy(),current.copy();start[indices]=a;end[indices]=b
        for arm in linear_samples(start,end,indices):
            q=current.copy();q[indices]=arm
            if abs(q[indices[6]])>math.pi:
                geometry.witness(q,'q7_outside_pi',[]);return False
            if geometry.check_for_self_collision(qpos=q) or geometry.check_for_env_collision(
                    qpos=q,with_point_cloud=True,use_attach=geometry.attach): return False
    return True


def canonical_v31_plan(planner,goal_pose,current,reference,dt,attach,phase,cloud,seed=0,candidates=64,parts=None):
    current=np.asarray(current,dtype=np.float64).copy();reference=np.asarray(reference,dtype=np.float64).copy()
    indices=np.asarray(planner.move_group_joint_indices)
    assert len(indices)==7
    planner.planning_world.set_use_point_cloud(True);planner.planning_world.set_use_attach(attach)
    planner.robot.set_qpos(current,True);planner.planning_world.set_qpos_all(current[indices])
    geometry=ZonedGeometry(planner,cloud,goal_pose[:3],attach,parts)
    if phase=='carry':
        goal=reference.copy();trace={'ik_initial':'not_used_carry_rest','goal_replaced':False}
    else: goal,trace=select_goal(planner,goal_pose,reference,geometry,seed,candidates)
    trace.update(phase=phase,attempts=[],clearance_policy={'base_radius':BASE_RADIUS,'middle_radius':MIDDLE_RADIUS,
        'tcp_terminal_distance':TERMINAL_DISTANCE,'target_fruit':'excluded as in v3'},
        native_rrt_seed_api='unavailable in mplib0.1.1')
    if goal is None:
        trace.update(status='no collision-free canonical IK target at 008m',method=None,goal_arm_qpos=None)
        geometry.set_radius(BASE_RADIUS,force=True);return None,trace
    trace['goal_arm_qpos']=goal[indices].tolist()
    nonarm=np.ones(len(current),dtype=bool);nonarm[indices]=False;goal[nonarm]=current[nonarm]
    tuck=tuck_waypoint(current,goal,reference,phase,indices)
    high=current.copy();high[indices]=HIGH_READY_ARM
    for method,waypoint,nodes in [('straight',None,[current,goal]),('detour','tuck',[current,tuck,goal]),
                                  ('detour','high_ready',[current,high,goal])]:
        geometry.clear_witness()
        try: result=checked_polyline(geometry,nodes,indices,dt,attach);error=None
        except RuntimeError as exc: result=None;error=str(exc)
        valid=result is not None and np.all(np.abs(np.asarray(result['position'])[:,6])<=math.pi)
        trace['attempts'].append({'method':method,'waypoint':waypoint,'success':bool(valid),
            'error':error,'first_collision':geometry.first_collision})
        if valid:
            trace.update(status='Success',method=method,waypoint=waypoint)
            geometry.set_radius(BASE_RADIUS,force=True);return result,trace
    # The native binding has a single cloud radius, no state-dependent collision
    # callback. Use .018 if both endpoints allow it; otherwise the explicitly
    # permitted .008 native search + .018 middle-only validation (§11.4.4).
    geometry.set_radius(MIDDLE_RADIUS)
    expanded_endpoints_clear=not target_collision(planner,current,attach) and not target_collision(planner,goal,attach)
    native_radius=MIDDLE_RADIUS if expanded_endpoints_clear else BASE_RADIUS
    geometry.set_radius(native_radius);geometry.clear_witness()
    trace['rrt_geometry']={'native_radius':native_radius,'postvalidation':'same zoned geometry as deterministic paths',
        'binding_limit':'Native mplib0.1.1 accepts one cloud radius, no per-state collision callback.'}
    try:
        result=planner.plan_qpos_to_qpos([goal],current.copy(),time_step=dt,use_point_cloud=True,
                                       use_attach=attach,planning_time=2)
        if result['status']=='Success':
            if not np.array_equal(result['position'][-1],goal[indices]):
                result['position']=np.vstack([result['position'],goal[indices]])
                if 'time' in result: result['time']=np.append(result['time'],result['time'][-1]+dt)
                for key in ('velocity','acceleration'):
                    if key in result: result[key]=np.vstack([result[key],np.zeros(7)])
                if 'duration' in result: result['duration']+=dt
            if not validate_positions(geometry,result['position'],current,indices):
                result={'status':'RRT path failed zoned collision/q7 validation'}
    except RuntimeError as exc: result={'status':f'RRT failed: {exc}'}
    trace['attempts'].append({'method':'rrt','status':result['status'],'first_collision':geometry.first_collision})
    trace.update(method='rrt',status=result['status'])
    geometry.set_radius(BASE_RADIUS,force=True)
    return (result if result['status']=='Success' else None),trace
