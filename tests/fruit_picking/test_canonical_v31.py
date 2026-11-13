from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fruit_picking.canonical_v31_planning import (BASE_RADIUS,MIDDLE_RADIUS,TUCK_Q2_Q7,
    ZonedGeometry,tuck_waypoint,canonical_v31_plan)
from test_canonical_planning import FakePlanner
from fruit_picking.rescue_generation import mark_history


class V31Tests(unittest.TestCase):
    def setUp(self): self.ref=np.zeros(9);self.ref[6]=-1.3933

    def test_terminal_zone_includes_exact_5cm_and_goal(self):
        p=FakePlanner();p.update_point_cloud=lambda cloud,radius:None
        geometry=ZonedGeometry(p,np.zeros((1,3)),np.array([.2,0,0]),False)
        q=self.ref.copy();q[0]=.2;self.assertEqual(geometry.radius_at(q),BASE_RADIUS)
        q[0]=.15;self.assertEqual(geometry.radius_at(q),MIDDLE_RADIUS if .2-.15>.05 else BASE_RADIUS)
        q[0]=.149;self.assertEqual(geometry.radius_at(q),MIDDLE_RADIUS)

    def test_tuck_has_fixed_six_angles_and_monotone_base_choice(self):
        current=self.ref.copy();current[0]=-1.7;goal=self.ref.copy();goal[0]=.4
        reach=tuck_waypoint(current,goal,self.ref,'reach',np.arange(7))
        carry=tuck_waypoint(current,goal,self.ref,'carry',np.arange(7))
        self.assertEqual(reach[0],goal[0]);self.assertEqual(carry[0],current[0])
        np.testing.assert_array_equal(reach[1:7],TUCK_Q2_Q7)
        np.testing.assert_array_equal(carry[1:7],TUCK_Q2_Q7)

    def test_carry_direct_rest_never_calls_ik(self):
        p=FakePlanner();radii=[];p.update_point_cloud=lambda cloud,radius:radii.append(radius)
        current=self.ref.copy();current[0]=.3
        result,trace=canonical_v31_plan(p,np.array([0,0,0,1,0,0,0]),current,self.ref,.05,False,'carry',np.zeros((1,3)))
        self.assertEqual(p.starts,[]);np.testing.assert_array_equal(result['position'][-1],self.ref[:7])
        self.assertEqual(trace['method'],'straight');self.assertFalse(trace['goal_replaced'])
        self.assertIn(BASE_RADIUS,radii);self.assertIn(MIDDLE_RADIUS,radii)

    def test_path_failure_never_switches_ik_goal(self):
        p=FakePlanner();p.update_point_cloud=lambda cloud,radius:None
        # Only path points at x~.1 are blocked. The .2 IK target is valid.
        p.check_for_env_collision=lambda qpos,**kwargs:['wall'] if .08<qpos[0]<.12 else []
        _,trace=canonical_v31_plan(p,np.array([.2,0,0,1,0,0,0]),self.ref,self.ref,.05,False,'reach',np.zeros((1,3)))
        self.assertFalse(trace['goal_replaced']);self.assertEqual(len(p.starts),1)
        self.assertEqual(trace['goal_arm_qpos'][0],.2)
        if p.rrt_calls: self.assertEqual(p.rrt_calls[0][0][0],.2)

    def test_target_collision_at_base_radius_allows_another_ik_candidate(self):
        p=FakePlanner();p.update_point_cloud=lambda cloud,radius:setattr(p,'cloud_radius',radius)
        def collisions(qpos,**kwargs):
            blocked=abs(qpos[0]-.2)<1e-12 and abs(qpos[1])<1e-12
            if blocked: self.assertEqual(p.cloud_radius,BASE_RADIUS)
            return ['blocked_goal'] if blocked else []
        p.check_for_env_collision=collisions
        result,trace=canonical_v31_plan(p,np.array([.2,0,0,1,0,0,0]),self.ref,self.ref,.05,False,'reach',np.zeros((1,3)),candidates=16)
        self.assertIsNotNone(result);self.assertTrue(trace['goal_replaced'])
        self.assertEqual(trace['replacement_reason'],'initial_target_collision_008')
        self.assertNotEqual(trace['goal_arm_qpos'][1],0.)

    def test_prefix_marks_history_without_removing_steps(self):
        values={f'timestep_{i}/info/is_video_demo':np.array(False) for i in range(32)}
        rec=SimpleNamespace(g=values,t=0)
        for i in range(32):
            rec.t=i+1;mark_history(rec,{'phase':'observe' if i<30 else 'reach'},True)
        self.assertTrue(all(values[f'timestep_{i}/info/is_video_demo'] for i in range(30)))
        self.assertFalse(values['timestep_30/info/is_video_demo']);self.assertEqual(len(values),32)


if __name__=='__main__': unittest.main()
