#!/usr/bin/env python3
"""Fixed D2 actions, with no planner calls; detailed leaf/fruit/robot state trace."""
import argparse
from pathlib import Path
import sys

import h5py
import numpy as np

from v_a_common import ECM,atomic_json,read_json,sha256,timestamp
sys.path.insert(0,str(ECM))


def snapshot(u):
    def array(tensor): return tensor[0].cpu().numpy().astype(np.float64).tolist()
    return {'robot_qpos':array(u.agent.robot.get_qpos()),'robot_qvel':array(u.agent.robot.get_qvel()),
        'robot_tcp_pose':array(u.agent.tcp_pose.raw_pose),
        'fruit_pose':[array(actor.pose.raw_pose) for actor in u.fruits],
        'fruit_linear_velocity':[np.asarray(u._body(actor).linear_velocity,dtype=np.float64).tolist() for actor in u.fruits],
        'fruit_angular_velocity':[np.asarray(u._body(actor).angular_velocity,dtype=np.float64).tolist() for actor in u.fruits],
        'fruit_attached':list(u.attached),'fruit_in_basket':u.in_basket().tolist(),
        'leaf_pose':[array(actor.pose.raw_pose) for actor in u.occluders],
        'leaf_push':list(u.occluder_push)}


def main():
    import gymnasium as gym
    import fruit_picking.env  # noqa
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=int,choices=[0,1,3],required=True)
    p.add_argument('--repeat',type=int,choices=[0,1,2],required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    source=ECM/f'results/fruit_expert_v2_audit_20261102/determinism/occluded_ripe/ep{a.episode}/repeat0/result.json'
    row=read_json(source);h5=source.parent/row['h5']
    with h5py.File(h5) as f:
        ep=f[f'episode_{a.episode}'];n=int(ep['setup/n_timesteps'][()])
        actions=[ep[f'timestep_{i}/action/joint_action'][()] for i in range(n)]
    env=gym.make('FruitPick-v1',task='occluded_ripe',robot_uids='rm75b_eg2',obs_mode='rgb+depth+segmentation',
                 control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='cpu')
    try:
        env.reset(seed=row['seed'])
        # nearest-v2 seeds these immediately after reset in its constructor.
        import random
        random.seed(0);np.random.seed(0)
        states=[]
        for action in actions:
            states.append(snapshot(env.unwrapped));env.step(action)
        states.append(snapshot(env.unwrapped))
        atomic_json(a.out/'trace.json',{'source':str(source),'source_h5':str(h5),'source_sha256':sha256(h5),
            'seed':row['seed'],'episode':a.episode,'repeat':a.repeat,'steps':n,'states':states,
            'action_sha256':__import__('hashlib').sha256(np.stack(actions).tobytes()).hexdigest(),
            'source_solver_mode':'nearest','source_global_seed_after_reset':0,
            'final_success':bool(env.unwrapped.evaluate()['success'][0]),'generated_utc':timestamp()})
    finally: env.close()


if __name__=='__main__': main()
