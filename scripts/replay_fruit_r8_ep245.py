#!/usr/bin/env python3
"""89 original executed actions in the original R7 scene; only v2 basket predicate changes."""
import sys
import types
from pathlib import Path

import numpy as np

from v_a_common import ECM,read_json,atomic_json,sha256,timestamp
sys.path.insert(0,str(ECM))


def main():
    import gymnasium as gym
    from mani_skill.utils.registration import register_env
    from fruit_picking.task_v2_env import FruitPickTaskV2Env
    out=ECM/'results/fruit_task_v2_R0_r8_20261105';snap=out/'r7_regression_sources'
    manifest=read_json(snap/'manifest.json')
    for n,h in manifest['files'].items():assert sha256(snap/n)==h
    spec_name='fruit_picking._r7_spec_ep245_r8'
    spec=types.ModuleType(spec_name);sys.modules[spec_name]=spec
    exec(compile((snap/'task_v2_spec.py').read_text(),str(snap/'task_v2_spec.py'),'exec'),spec.__dict__)
    env_name='fruit_picking._r7_env_ep245_r8'
    module=types.ModuleType(env_name);sys.modules[env_name]=module
    text=(snap/'task_v2_env.py').read_text().replace('from .task_v2_spec import','from ._r7_spec_ep245_r8 import')
    text='\n'.join(l for l in text.splitlines() if not l.startswith('@register_env('))
    exec(compile(text,str(snap/'task_v2_env.py'),'exec'),module.__dict__)
    cls=module.FruitPickTaskV2Env
    cls.in_basket=FruitPickTaskV2Env.in_basket
    register_env('FruitBasketRegressionR8-v0',max_episode_steps=1300)(cls)
    old=ECM/'results/fruit_task_v2_R0_20261105/replays/count/initial_current/ep245'
    result=read_json(old/'result.json');actions=read_json(old/'actions.json')
    env=gym.make('FruitBasketRegressionR8-v0',task='count',task_version='v2',robot_uids='rm75b_eg2',
        obs_mode='rgb+depth+segmentation',control_mode='pd_joint_pos',sim_backend='cpu',render_mode='rgb_array')
    rows=[]
    try:
        env.reset(seed=result['seed'])
        for action in actions:
            _,_,terminated,truncated,info=env.step(np.asarray(action['executed']))
            u=env.unwrapped
            rows.append({'step':u.control_steps,'positions':u.fruit_positions().tolist(),
                'inside':u.in_basket().tolist(),'reason':info.get('v2_reason')})
            assert info.get('v2_reason')!='wind_basket_anomaly'
        assert env.unwrapped.wind_events==result['wind_events']
        assert len(rows)==89
        atomic_json(out/'ep245_native_replay.json',{'passed':True,'unit_check':'16.3(d)',
            'steps':len(rows),'seed':result['seed'],'wind_events':env.unwrapped.wind_events,
            'old_source_commit':'6eb6197','only_modified_behavior':'v2 in_basket lower z bound',
            'actions_sha256':sha256(old/'actions.json'),'created_utc':timestamp(),'trace':rows})
        print('ep245 native replay passed: 89 original actions, same wind events, no false anomaly',flush=True)
    finally:env.close()


if __name__=='__main__':main()
