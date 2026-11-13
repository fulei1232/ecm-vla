#!/usr/bin/env python3
"""R8 CPU geometry and state-replay checks. Native ep245 replay is a separate GPU gate."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from v_a_common import ECM,atomic_json,read_json,read_jsonl,sha256,timestamp
sys.path.insert(0,str(ECM))
from fruit_picking.task_v2_spec import TaskV2Spec,bounded_inside

OUT=ECM/'results/fruit_task_v2_R0_r8_20261105'


def main():
    reading=read_json(OUT/'geometry_reading.json');assert reading['confirmed']
    assert sha256(ECM/'fruit_picking/env.py')==reading['source_sha256']
    center=np.array([-.06,-.34,.15]);radius=.026
    positions=np.array([center+[0,0,.006+radius],[-.06,-.34,-.4+radius],center+[0,0,.12+radius]])
    result=bounded_inside(positions,center,.09,.12).tolist()
    assert result==[True,False,False]
    source=ECM/'results/fruit_task_v2_R0_20261105/replays/count/initial_current/ep245'
    rows=read_jsonl(source/'trace.jsonl');mismatches=[]
    for row in rows:
        flags=bounded_inside(row['fruit_positions'],center,.09,.12)
        if any(flags[i] for i,b in enumerate(row['wind_detached']) if b):mismatches.append(row['physical_step'])
    assert not mismatches
    native=None
    try:
        from fruit_picking.task_v2_env import FruitPickTaskV2Env
        u=FruitPickTaskV2Env.__new__(FruitPickTaskV2Env)
        u.v2_spec=TaskV2Spec(version='v2');u.layout=SimpleNamespace(basket={'center':center,'inner_half':.09,'height':.12})
        u.present=np.ones(3,dtype=bool);u.fruit_positions=lambda:positions
        assert u.in_basket().tolist()==result
        u.v2_spec=TaskV2Spec(version='v1')
        assert u.in_basket().tolist()==[True,True,False]
        native=True
    except ModuleNotFoundError:
        native=False
    proof={'revision':8,'passed':True,'created_utc':timestamp(),'geometry_confirmed':reading,
        'unit_checks':{'a_resting_on_floor':result[0],'b_under_basket_excluded':not result[1],
            'c_above_rim_excluded':not result[2],
            'd_ep245_recorded_state_replay_no_wind_basket_flag':not mismatches},
        'native_method_and_v1_compatibility_checked':native,'ep245_frames_replayed':len(rows),
        'native_physics_ep245_replay':'pending; must pass before new R0 audit',
        'ep245_trace_sha256':sha256(source/'trace.jsonl'),'source_sha256':sha256(Path(__file__))}
    atomic_json(OUT/'geometry_unit_checks.json',proof);print(proof,flush=True)


if __name__=='__main__':main()
