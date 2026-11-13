#!/usr/bin/env python3
"""CPU preflight in the actual simulator environment; never creates a scene or renderer."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from v_a_common import ECM, atomic_json, sha256, timestamp
sys.path.insert(0,str(ECM))
from fruit_picking.env import FruitPickEnv
from fruit_picking.task_v2_env import FruitPickTaskV2Env
from fruit_picking.task_v2_spec import TaskV2Spec


def main():
    u=FruitPickTaskV2Env.__new__(FruitPickTaskV2Env)
    rest=np.array([-1.7306,-.5484,-.0061,2.0666,-.181,.0534,-1.3933,.04,.04])
    q=rest.copy();vel=np.zeros(9)
    u.v2_spec=TaskV2Spec(version='v2');u.task='count';u.device=torch.device('cpu')
    u.layout=SimpleNamespace(n_pick=2);u.fruits=[0,1,2]
    u.attached=[True,True,True];u.wind_detached=[True,False,False]
    u.wind_events=[{'step':64,'fruits':[0]}];u.v2_wind_plan={'step':64}
    u.finish_streak=0;u.control_steps=0
    u.agent=SimpleNamespace(keyframes={'rest':SimpleNamespace(qpos=rest)},
        robot=SimpleNamespace(get_qpos=lambda:torch.tensor(q[None,:]),get_qvel=lambda:torch.tensor(vel[None,:])),
        is_grasping=lambda _:torch.tensor([False]))
    u.in_basket=lambda:np.array([False,True,True])
    u.fruit_positions=lambda:np.array([[.4,0,-.4],[-.06,-.34,.2],[-.05,-.34,.2]])
    def tick(self):self.control_steps+=1
    with patch.object(FruitPickEnv,'_after_control_step',tick):
        for _ in range(30):u._after_control_step()
        assert u.finish_streak==0 and not u.evaluate()['success'][0] and not u.evaluate()['fail'][0]
        for _ in range(39):u._after_control_step()
        assert u.finish_streak==39 and not u.evaluate()['success'][0]
        u._after_control_step();assert u.evaluate()['success'][0]
        q[0]+=.06;u._after_control_step();assert u.finish_streak==0
        q[:]=rest;vel[0]=.06;u._after_control_step();assert u.finish_streak==0
        vel[:]=0;q[7]=.02;u._after_control_step();assert u.finish_streak==0
    u.wind_detached=[False,False,False]
    assert u.evaluate()['fail'][0] and u.evaluate()['v2_reason']=='robot_drop'
    proof={'passed':True,'kind':'CPU mocked-physics preflight; NOT R0 physical feasibility',
        'default_off_and_native_import':True,'prefix_not_counted_as_completion':True,
        '40_consecutive_steps_required':True,'rest_velocity_open_finger_thresholds_checked':True,
        'windfall_not_robot_drop_or_pick':True,'source_sha256':sha256(Path(__file__)), 'generated_utc':timestamp()}
    out=ECM/'results/fruit_task_v2_R0_20261105';atomic_json(out/'native_preflight.json',proof)
    print(proof)


if __name__=='__main__':main()
