#!/usr/bin/env python3
"""Single fresh R8 dev replay, with no train/val path or learning code."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from v_a_common import ECM,atomic_json,read_json,sha256,timestamp
sys.path[:0]=[str(ECM),str(ECM/'src')]
from fruit_r8_episode_support import R8Replay,EpisodeEnd,TargetGone

OUT=ECM/'results/fruit_task_v2_R0_r8_20261105'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('count','occluded_ripe'),required=True)
    p.add_argument('--level',choices=('current','initial_current','initial_ground','history','clock','clock_onset','nonstop'),required=True)
    p.add_argument('--episode',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--purpose',choices=('audit','calibration','budget_expert','budget_nonstop'),default='audit')
    p.add_argument('--task-version',choices=('v1','v2'),default='v1');p.add_argument('--clock-fit',type=Path)
    a=p.parse_args()
    if a.task_version!='v2':p.error('Explicit --task-version v2 required; default does not enable new behavior')
    if a.purpose=='audit' and not 200<=a.episode<=249:p.error('Audit dev200–249 only')
    if a.purpose!='audit' and not 250<=a.episode<=259:p.error('Calibration/budget dev250–259 only')
    if a.purpose!='audit' and a.task!='count':p.error('Only Count uses calibration/budget stress')
    if a.purpose=='budget_nonstop' and a.level!='nonstop':p.error('Budget nonstop must keep picking')
    if a.purpose=='audit' and a.level=='nonstop':p.error('Nonstop is a budget diagnostic, not an audit cohort')
    if a.purpose in ('calibration','budget_expert') and a.level!='history':p.error('Reference expert required')
    if a.task=='occluded_ripe' and a.level not in ('current','initial_current','history'):p.error('Unknown occluded audit')
    a.out=a.out.resolve()
    if not a.out.is_relative_to(OUT/'replays'):p.error('Only the new R8 replay directory is writable')
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'result.json').exists():p.error('Existing attempts must not be overwritten')
    fit=None
    if a.level in ('clock','clock_onset'):
        if a.clock_fit is None or a.clock_fit.resolve()!=OUT/'clock_fit.json':p.error('Use the frozen calibration fit')
        fits=read_json(a.clock_fit);fit=fits['relative' if a.level=='clock_onset' else 'absolute']
    import gymnasium as gym
    import fruit_picking.task_v2_env  # noqa
    from fruit_picking.solver import FruitPickSolver,PlanningFailure,horizontal_grasp_pose,PRE_GRASP
    from fruit_picking.canonical_solver import configure
    budget=1800 if a.task=='count' else 1300
    env=gym.make('FruitPickTaskV2-v0',task=a.task,task_version='v2',r8_budget_stress=a.purpose.startswith('budget_'),
        robot_uids='rm75b_eg2',obs_mode='rgb+depth+segmentation',control_mode='pd_joint_pos',
        render_mode='rgb_array',sim_backend='cpu',max_episode_steps=budget)
    replay=None;failure=''
    guard_path=ECM/'results/fruit_k0v2_gate_20261102/guard_v2.json'
    try:
        obs,_=env.reset(seed=4200000+16*a.episode);u=env.unwrapped
        replay=R8Replay(env,obs,a.out,a.level,read_json(guard_path),fit)
        solver=FruitPickSolver(env,replay.step,ik_mode='canonical_v31',ik_seed=0,ik_candidates=64)
        configure(solver,{'settle_steps':2})
        fixed=np.r_[u.agent.robot.get_qpos()[0].cpu().numpy()[:7],1.]
        try:
            solver.hold(30,{'phase':'observe','target':None,'subgoal':'observe initial scene'})
            if a.task=='count':
                while replay.provided()['lamp_green'] is not True:
                    solver.step_fn(fixed.copy(),{'phase':'wait_for_green','target':None,'subgoal':'wait at rest for green light'})
            while not replay.ended:
                slot=replay.choose()
                if slot is None and a.task=='occluded_ripe' and a.level=='initial_current':
                    # Restart this search from frame0 every time; no completed-target cache.
                    initial=sorted([f for f in replay.initial['tree'] if f['ripe_visible']],key=lambda f:-f['position'][1])
                    for item in initial:
                        candidate=item['slot'];replay.choose(inspect_slot=candidate)
                        pos=np.asarray(item['position']);direction=np.r_[pos[:2],0.];direction/=np.linalg.norm(direction)
                        solver.refresh_obstacles(skip=(candidate,));solver.gripper=1.
                        solver.move(horizontal_grasp_pose(pos-PRE_GRASP*direction,direction),
                            {'phase':'reach','target':candidate,'subgoal':'inspect a remembered initial red location'})
                        slot=replay.choose()
                        if slot is not None:break
                if slot is None:
                    solver.gripper=1.;solver.refresh_obstacles()
                    solver.move(solver.home,{'phase':'carry','target':0,'subgoal':'return to rest to finish'})
                    solver.hold(budget,{'phase':'finish','target':None,'subgoal':'hold open at rest until termination'})
                else:
                    actor=u.hanging_slot[slot]
                    if actor is None:
                        solver.hold(1,{'phase':'inspect','target':None,'subgoal':'observe current tree occupancy'});continue
                    try:
                        solver.pick(actor,0,u.layout.n_pick)
                        replay.releases.append({'step':u.control_steps,'actor':actor,'slot':slot,'event':'own_release_attempt'})
                        if a.task=='count':
                            while actor not in u.replenished:
                                solver.hold(1,{'phase':'await_replenishment','target':None,'subgoal':'confirm current placement and replenishment'})
                    except TargetGone:
                        solver.hold(1,{'phase':'inspect','target':None,'subgoal':'observe wind-lost current target'})
        except EpisodeEnd:pass
        except PlanningFailure as exc:failure=str(exc);replay.reason='planning_or_grasp_failure'
        actions=replay.actions[30:]
        points=[{'step':d['step'],'completed_releases':len(d['provided']['release_events'])}
            for d in replay.decisions if a.level=='history' and d['purpose']=='pick_or_finish']
        final=u.evaluate();robot=np.asarray(u.attached)&~np.asarray(u.wind_detached)&u.present
        result={'revision':8,'task':a.task,'level':a.level,'purpose':a.purpose,'episode':a.episode,
            'seed':4200000+16*a.episode,'success':replay.success,'outcome':replay.reason,'planning_failure':failure,
            'steps':u.control_steps,'N':u.layout.n_pick,'M':u.original_slots,'green_step':u.green_step,
            'budget':budget,'remaining_steps':budget-u.control_steps,'early_start_steps':u.early_start_steps,
            'guard_checked_steps':len(actions),'guard_clipped_steps':sum(x['guard_clipped'] for x in actions),
            'guard_sha256':sha256(guard_path),'max_visibility':replay.max_visibility,'snapshots':replay.snapshots,
            'appearance_checks':replay.appearance,'appearance_final_hold_available':'finish_hold' in replay.snapshots,
            'wind_plan':getattr(u,'v2_wind_plan',None),'wind_events':u.wind_events,
            'replenish_events':u.replenish_events,'basket_entries':replay.basket_entries,'own_release_events':replay.releases,
            'slot_geometry':json.loads(json.dumps(u.original_geometry,default=lambda v:np.asarray(v).tolist())),
            'slot_of':u.slot_of,'decision_points':points,
            'final_robot_picked':int(robot.sum()),'final_robot_in_basket':int(final['n_in_basket'][0]),
            'budget_exhausted_exact_N':bool(u.control_steps>=budget and int(final['n_in_basket'][0])==u.layout.n_pick),
            'solver_version':solver.solver_version,'clock_fit_sha256':sha256(a.clock_fit) if a.clock_fit else None,
            'created_utc':timestamp(),'diagnostic_only':True,'training_data':False}
        atomic_json(a.out/'result.json',result);atomic_json(a.out/'actions.json',replay.actions)
        atomic_json(a.out/'decisions.json',replay.decisions)
        print({k:result[k] for k in ['task','level','purpose','episode','N','success','outcome','steps']},flush=True)
    finally:
        if replay is not None:replay.trace.close()
        env.close()


if __name__=='__main__':main()
