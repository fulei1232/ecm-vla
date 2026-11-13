#!/usr/bin/env python3
"""Reuse v3 sampling/restoration, with revision3's separate continuity gates."""
from pathlib import Path
import sys
import numpy as np

from v_a_common import ECM,atomic_json,read_json
sys.path.insert(0,str(ECM))
import audit_fruit_canonical_planning as v3
from fruit_picking.canonical_v31_planning import canonical_v31_plan


def replan(planner,data,meta,offset=None):
    current,pose=data['current'].copy(),data['goal_pose'].copy()
    if offset is not None:
        fruit=data['fruit_center']+offset;direction=np.r_[fruit[:2],0.]/np.linalg.norm(fruit[:2])
        pre=v3.horizontal_grasp_pose(fruit-v3.PRE_GRASP*direction,direction);pose=np.r_[pre.p,pre.q]
        current[:7]+=np.where(np.arange(7)%2,1.,-1.)*.001*(1 if np.sum(offset)>=0 else -1)
    return canonical_v31_plan(planner,pose,current,data['reference'],meta['dt'],meta['attach'],meta['phase'],
        data['point_cloud'],meta['ik_seed'],meta['ik_candidates'],meta.get('obstacle_parts'))


def continuity_case(base,others):
    traces=[base]+others;goals=[row.get('goal_arm_qpos') for row in traces]
    delta=float(np.ptp(np.asarray(goals),axis=0).max()) if all(g is not None for g in goals) else None
    target=delta is not None and delta<.05
    methods=[row.get('method') for row in traces]
    method=all(row.get('status')=='Success' for row in traces) and len(set(methods))==1
    return {'passed':bool(target and method),'target_passed':bool(target),'method_passed':bool(method),
        'max_pairwise_goal_difference_rad':delta,'methods':methods,
        'waypoints':[row.get('waypoint') for row in traces],'goals':goals}


if __name__=='__main__':
    v3.replan=replan;v3.continuity_case=continuity_case;v3.main()
    out=Path(sys.argv[sys.argv.index('--out')+1]);summary=read_json(out/'summary.json')
    rows=summary['continuity']['cases'];n=len(rows)
    summary['target_continuity']={'instances':n,'passed_instances':sum(r['target_passed'] for r in rows),
        'passed':n==100 and sum(r['target_passed'] for r in rows)>=98}
    summary['method_consistency']={'instances':n,'passed_instances':sum(r['method_passed'] for r in rows),
        'passed':n==100 and sum(r['method_passed'] for r in rows)>=95}
    summary['revision']=3
    atomic_json(out/'summary.json',summary)
