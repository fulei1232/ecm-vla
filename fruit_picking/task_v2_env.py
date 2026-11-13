"""Revision8 environment: opt-in bounded basket, pooled replenishment and random start."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import sapien
import torch

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from .env import FruitPickEnv, material, STEM
from .layout import FRONT_CAMERA_EYE, FRONT_CAMERA_TARGET, FRONT_CAMERA_FOV
from .task_v2_spec import (TaskV2Spec, task_layout, wind_plan, start_delay, neutral_alpha,
    count_outcome, bounded_inside, early_start)


def lamp_position():
    forward = FRONT_CAMERA_TARGET-FRONT_CAMERA_EYE
    forward = forward/np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]);right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    half = .72*np.tan(FRONT_CAMERA_FOV/2)
    return FRONT_CAMERA_EYE+.72*forward-.76*half*right+.76*half*up


@register_env('FruitPickTaskV2-v0', max_episode_steps=1800)
class FruitPickTaskV2Env(FruitPickEnv):
    def __init__(self, *args, task_version='v1', task_revision=8,
                 r8_budget_stress=False, r9_budget_stress=False,
                 no_lamp_wait=False, **kwargs):
        if task_revision not in (8, 9):raise ValueError('task_revision must be 8 or 9')
        self.task_revision = int(task_revision)
        self.no_lamp_wait = bool(no_lamp_wait)
        self.v2_spec = TaskV2Spec(version=task_version)
        if task_version == 'v2' and task_revision == 9 and kwargs.get('task') == 'count':
            self.v2_spec = replace(self.v2_spec, budget=2800, count_n_max=10, early_start_failure=True)
        if self.no_lamp_wait:
            self.v2_spec = replace(self.v2_spec, start_delay_max=0, early_start_failure=False)
        if kwargs.get('task','occluded_ripe') != 'count':self.v2_spec=replace(self.v2_spec,budget=1300)
        self.r8_budget_stress = bool(r8_budget_stress)
        self.r9_budget_stress = bool(r9_budget_stress)
        if self.r8_budget_stress and (task_version!='v2' or kwargs.get('task')!='count'):
            raise ValueError('Budget stress is only a v2 count diagnostic')
        if self.r9_budget_stress and (task_version != 'v2' or task_revision != 9 or kwargs.get('task') != 'count'):
            raise ValueError('R9 stress requires revision9 v2 count')
        if self.r8_budget_stress and self.r9_budget_stress:raise ValueError('Choose one stress definition')
        super().__init__(*args, **kwargs)

    @property
    def v2_enabled(self):
        return self.v2_spec.version == 'v2'

    def _load_scene(self, options):
        if self.v2_enabled:
            layout=task_layout(self.task,int(self._episode_seed[0]),'v2',self.v2_spec)
            if self.no_lamp_wait and self.task == 'count':
                prefix = 'Wait until the work light turns green, then '
                if layout.instruction.startswith(prefix):
                    layout.instruction = layout.instruction[len(prefix):]
            if self.r8_budget_stress or self.r9_budget_stress:
                old_n=layout.n_pick;layout.n_pick=10 if self.r9_budget_stress else 5
                layout.instruction=layout.instruction.replace(f'exactly {old_n} ', f'exactly {layout.n_pick} ')
            options={**options,'layout':layout}
        super()._load_scene(options)
        self.original_slots=len(self.fruits)
        self.slot_of=list(range(self.original_slots))
        self.original_geometry=deepcopy(self.layout.fruits)
        self.reserve_ids=[[] for _ in range(self.original_slots)]
        self.park_positions={};self.reserve_collision_groups={};self.v2_materials=[]
        if not self.v2_enabled:return
        if self.task=='count':
            self.lamp_material=material((.9,.05,.03),.5)
            builder=self.scene.create_actor_builder()
            builder.add_sphere_visual(radius=.02,material=self.lamp_material)
            builder.initial_pose=sapien.Pose(p=lamp_position())
            self.work_lamp=builder.build_static(name='work_lamp')
            for slot,f in enumerate(self.original_geometry):
                for generation in range(1,self.layout.n_pick+1):
                    i=len(self.fruits);parking=[100.+2*i,100.,10.]
                    builder=self.scene.create_actor_builder()
                    phys=sapien.physx.PhysxMaterial(static_friction=1.,dynamic_friction=1.,restitution=.1)
                    builder.add_sphere_collision(radius=f['radius'],material=phys,density=800)
                    builder.add_sphere_visual(radius=f['radius'],material=material(f['rgb'],.45))
                    builder.add_sphere_visual(sapien.Pose(p=[0,0,f['radius']-.003]),radius=.006,material=material(STEM))
                    builder.initial_pose=sapien.Pose(p=parking)
                    actor=builder.build_kinematic(name=f'reserve_slot{slot}_generation{generation}')
                    self.fruits.append(actor);self.slot_of.append(slot)
                    self.layout.fruits.append(deepcopy(f))
                    self.reserve_ids[slot].append(i);self.park_positions[i]=parking
                    self.reserve_collision_groups[i]=[tuple(s.collision_groups) for s in self._body(actor).collision_shapes]
        else:
            for actor in self.fruits:
                body=actor._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
                self.v2_materials.append([(s.material,np.asarray(s.material.base_color).copy()) for s in body.render_shapes])

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx,options)
        n=len(self.fruits)
        self.wind_detached=[False]*n;self.wind_events=[];self.wind_slots=set()
        self.finish_streak=0;self.finish_requested=False;self.v2_color_alpha=0.
        self.present=np.ones(n,dtype=bool);self.hanging_slot=list(range(self.original_slots))
        self.replenished=set();self.pending_replenish={};self.replenish_events=[]
        self.receipt_streak=np.zeros(n,dtype=np.int64);self.early_start_steps=[]
        self.lamp_green=False;self.green_step=None
        self.rest_tcp_position=self.agent.tcp_pose.p[0].cpu().numpy().copy()
        if not self.v2_enabled:return
        for i,actor in enumerate(self.fruits):
            body=self._body(actor);body.kinematic=True
            body.linear_velocity=[0,0,0];body.angular_velocity=[0,0,0]
            if i in self.park_positions:
                self.present[i]=False;self.attached[i]=True
                actor.set_pose(Pose.create_from_pq(torch.tensor(self.park_positions[i],dtype=torch.float32)))
                for shape in body.collision_shapes:shape.set_collision_groups([0,0,0,0])
        for items in self.v2_materials:
            for mat,original in items:mat.base_color=original
        self.reserve_remaining=[list(v) for v in self.reserve_ids]
        if self.task=='count':
            self.green_step=self.v2_spec.prefix_steps+(600 if self.r8_budget_stress or self.r9_budget_stress else start_delay(int(self._episode_seed[0]),self.v2_spec))
            self.v2_wind_plan=wind_plan(int(self._episode_seed[0]),self.v2_spec)
            self.v2_wind_plan['absolute_step']=self.green_step+self.v2_wind_plan['step']
            self.lamp_material.base_color=[.9,.05,.03,1.]

    def in_basket(self):
        if not self.v2_enabled:return super().in_basket()
        L=self.layout
        inside=bounded_inside(self.fruit_positions(),L.basket['center'],L.basket['inner_half'],L.basket['height'])
        return inside & getattr(self,'present',np.ones(len(inside),dtype=bool))

    def hanging(self):
        return self.present & ~np.asarray(self.attached,dtype=bool)

    def _grasped(self):
        return {int(i) for i in np.flatnonzero(self.present) if bool(self.agent.is_grasping(self.fruits[int(i)])[0])}

    def _replenish(self,grasped,tcp):
        inside=self.in_basket();wind=np.asarray(self.wind_detached)
        for i in np.flatnonzero(self.present & np.asarray(self.attached) & ~wind):
            i=int(i)
            if i in self.replenished:continue
            self.receipt_streak[i]=self.receipt_streak[i]+1 if inside[i] and i not in grasped else 0
            if self.receipt_streak[i]>=self.v2_spec.receipt_steps:
                self.pending_replenish.setdefault(i,self.control_steps)
        for old,ready in list(self.pending_replenish.items()):
            slot=self.slot_of[old]
            if np.linalg.norm(tcp-np.asarray(self.original_geometry[slot]['pos']))<self.v2_spec.replenish_clearance_m:
                continue
            if old in grasped:continue
            if not self.reserve_remaining[slot]:raise RuntimeError('R8_RESERVE_POOL_EXHAUSTED')
            new=self.reserve_remaining[slot].pop(0);actor=self.fruits[new];body=self._body(actor)
            body.kinematic=True;body.linear_velocity=[0,0,0];body.angular_velocity=[0,0,0]
            actor.set_pose(Pose.create_from_pq(torch.tensor(self.original_geometry[slot]['pos'],dtype=torch.float32)))
            for shape,groups in zip(body.collision_shapes,self.reserve_collision_groups[new]):shape.set_collision_groups(groups)
            self.present[new]=True;self.attached[new]=False;self.grasp_streak[new]=0
            self.hanging_slot[slot]=new;self.replenished.add(old);del self.pending_replenish[old]
            self.replenish_events.append({'step':self.control_steps,'ready_step':ready,'delay_steps':self.control_steps-ready,
                'slot':slot,'old_actor':old,'new_actor':new})

    def _after_control_step(self):
        super()._after_control_step()
        if not self.v2_enabled:return
        spec=self.v2_spec
        if self.task=='occluded_ripe':
            self.v2_color_alpha=neutral_alpha(self.control_steps,spec)
            for items in self.v2_materials:
                for mat,original in items:
                    color=original.copy();color[:3]=(1-self.v2_color_alpha)*original[:3]+self.v2_color_alpha*np.asarray(spec.neutral_rgb)
                    mat.base_color=color
            return
        self.lamp_green=self.control_steps>=self.green_step
        self.lamp_material.base_color=[.03,.85,.05,1.] if self.lamp_green else [.9,.05,.03,1.]
        tcp=self.agent.tcp_pose.p[0].cpu().numpy();grasped=self._grasped()
        q=self.agent.robot.get_qpos()[0].cpu().numpy();vel=self.agent.robot.get_qvel()[0].cpu().numpy()
        reference=np.asarray(self.agent.keyframes['rest'].qpos)
        started_early = early_start(q,reference,tcp,self.rest_tcp_position,self.lamp_green,spec)
        if started_early or (not spec.early_start_failure and self.control_steps>30 and not self.lamp_green
                            and np.max(np.abs(q[:7]-reference[:7]))>spec.rest_joint_tolerance_rad):
            self.early_start_steps.append(self.control_steps)
        for slot,i in enumerate(self.hanging_slot):
            if i is not None and self.attached[i]:self.hanging_slot[slot]=None
        if self.lamp_green and self.control_steps>=self.v2_wind_plan['absolute_step'] and not self.wind_events:
            positions=self.fruit_positions()
            available=[int(i) for i in np.flatnonzero(self.hanging())
                if i not in grasped and np.linalg.norm(positions[i]-tcp)>spec.wind_protect_tcp_m]
            if len(available)>=self.v2_wind_plan['count']:
                rng=np.random.default_rng(self.v2_wind_plan['order_seed'])
                selected=sorted(rng.choice(available,self.v2_wind_plan['count'],replace=False).tolist())
                for i in selected:
                    body=self._body(self.fruits[i]);body.kinematic=False
                    body.linear_velocity=[0,0,0];body.angular_velocity=[0,0,0]
                    self.attached[i]=self.wind_detached[i]=True
                    slot=self.slot_of[i];self.hanging_slot[slot]=None;self.wind_slots.add(slot)
                self.wind_events.append({'step':self.control_steps,'fruits':selected,'slots':[self.slot_of[i] for i in selected]})
        self._replenish(grasped,tcp)
        resting=np.max(np.abs(q[:7]-reference[:7]))<=spec.rest_joint_tolerance_rad
        resting &= np.max(np.abs(vel[:7]))<=spec.static_velocity_rad_s
        resting &= bool(np.all(q[7:9]>=spec.open_finger_min_m)) and not grasped
        self.finish_streak=self.finish_streak+1 if self.lamp_green and resting else 0
        self.finish_requested=self.finish_streak>=spec.finish_hold_steps

    def evaluate(self):
        if not self.v2_enabled or self.task!='count':return super().evaluate()
        inside=self.in_basket();wind=np.asarray(self.wind_detached,dtype=bool)
        robot=np.asarray(self.attached,dtype=bool)&~wind&self.present
        dropped=robot&~inside&(self.fruit_positions()[:,2]<.06)
        result=count_outcome(self.control_steps,self.finish_streak,self.layout.n_pick,
            int((inside&~wind).sum()),int(robot.sum()),bool(dropped.any()),self.v2_spec)
        if bool((inside&wind).any()):result.update(success=False,fail=True,reason='wind_basket_anomaly')
        if self.v2_spec.early_start_failure and self.early_start_steps:
            result.update(success=False,fail=True,reason='early_start')
        as_t=lambda value:torch.tensor([value],device=self.device)
        return {'success':as_t(result['success']),'fail':as_t(result['fail']),'n_in_basket':as_t(result['n_in_basket']),
            'is_grasped_any':as_t(bool(self._grasped())),'v2_reason':result['reason'],
            'v2_robot_picked':as_t(result['robot_picked']),'v2_finish_streak':as_t(self.finish_streak),
            'budget_exact_N':as_t(self.control_steps>=self.v2_spec.budget and result['n_in_basket']==self.layout.n_pick)}
