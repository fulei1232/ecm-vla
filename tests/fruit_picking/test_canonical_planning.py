"""Core semantic tests: fixed targets, ordered fallbacks and honest audit gates."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from fruit_picking.canonical_planning import canonical_candidates,canonical_goal,canonical_plan,result_digest
from analyze_fruit_physics_divergence import first_divergence
from report_fruit_expert_v3 import basket_miss_exceeds_three_percent,core_gates


class FakePlanner:
    move_group_joint_indices=list(range(7))
    joint_limits=np.tile([-6.28,6.28],(9,1))
    link_name_2_idx={'tcp':0}
    move_group='tcp'

    def __init__(self):
        self.starts=[];self.rrt_calls=[];self.path_checks=[]
        self.robot=SimpleNamespace(set_qpos=lambda *args:None)
        self.planning_world=SimpleNamespace(set_use_point_cloud=lambda *a:None,set_use_attach=lambda *a:None,
                                             set_qpos_all=lambda *a:None)
        planner=self
        class Model:
            def compute_IK_CLIK(self,link,pose,start,mask):
                planner.starts.append(start.copy());q=start.copy();q[0]=pose[0];return (q,True,None)
            def compute_forward_kinematics(self,q): self.q=q
            def get_link_pose(self,link): return np.r_[self.q[0],0,0,1,0,0,0]
        self.pinocchio_model=Model()
    def wrap_joint_limit(self,q): return True
    def distance_6D(self,a,b,c,d): return np.linalg.norm(a-c)
    def check_for_self_collision(self,qpos): return []
    def check_for_env_collision(self,**kwargs):
        self.path_checks.append(kwargs['qpos'].copy());return []
    def TOPP(self,path,dt): return np.arange(len(path))*dt,path,path*0,path*0,(len(path)-1)*dt
    def plan_qpos_to_qpos(self,goals,start,**kwargs):
        self.rrt_calls.append(goals);return {'status':'Success','position':np.stack([start[:7],goals[0][:7]])}


class CanonicalTests(unittest.TestCase):
    def setUp(self):
        self.ref=np.zeros(9);self.ref[6]=-1.4

    def test_carry_uses_exact_rest_without_any_ik_call(self):
        p=FakePlanner();current=self.ref.copy();current[0]=.3
        result,trace=canonical_plan(p,np.zeros(7),current,self.ref,.05,True,'carry')
        self.assertEqual(p.starts,[]);self.assertEqual(trace['ik_initial'],'not_used_carry_rest')
        np.testing.assert_array_equal(result['position'][-1],self.ref[:7])

    def test_reach_target_is_independent_of_current_and_rest_clik_is_first(self):
        p=FakePlanner();pose=np.array([.2,0,0,1,0,0,0]);current=self.ref.copy()
        _,first=canonical_plan(p,pose,current,self.ref,.05,False,'reach')
        current[:7]+=.01
        _,second=canonical_plan(p,pose,current,self.ref,.05,False,'reach')
        self.assertEqual(first['goal_arm_qpos'],second['goal_arm_qpos'])
        self.assertEqual(len(p.starts),2)
        for start in p.starts: np.testing.assert_array_equal(start,self.ref)
        self.assertEqual(first['random_initializations'],0)

    def test_fixed_rest_detour_before_rrt_and_target_never_switches(self):
        p=FakePlanner();current=self.ref.copy();current[:2]=[.3,.3]
        # A wall blocks the direct diagonal; both legs via fixed rest avoid it.
        def collide(**kwargs):
            q=kwargs['qpos'];return ['wall'] if abs(q[0]-.25)<.015 and abs(q[1]-.15)<.035 else []
        p.check_for_env_collision=collide
        result,trace=canonical_plan(p,np.array([.2,0,0,1,0,0,0]),current,self.ref,.05,False,'reach')
        self.assertEqual(trace['method'],'detour');self.assertEqual(trace['waypoint'],'rest')
        self.assertEqual([t['method'] for t in trace['attempts']],['straight','detour'])
        self.assertEqual(p.rrt_calls,[])
        np.testing.assert_array_equal(result['position'][-1],trace['goal_arm_qpos'])

    def test_rrt_is_last_and_cannot_switch_to_another_goal(self):
        p=FakePlanner();p.check_for_env_collision=lambda **kwargs:['blocked']
        result,trace=canonical_plan(p,np.array([.2,0,0,1,0,0,0]),self.ref,self.ref,.05,False,'reach')
        self.assertEqual([t['method'] for t in trace['attempts']],['straight','detour','detour','rrt'])
        self.assertEqual(len(p.rrt_calls),1);self.assertEqual(len(p.rrt_calls[0]),1)
        np.testing.assert_array_equal(p.rrt_calls[0][0][:7],trace['goal_arm_qpos'])
        self.assertIsNone(result)  # A colliding native RRT result is not accepted.

    def test_q7_is_mapped_near_reference_and_weights_are_equal(self):
        a=self.ref.copy();a[0]=.2;a[6]=2*np.pi-1.4
        b=self.ref.copy();b[4]=.3
        ranked=canonical_candidates([b,a],self.ref,np.arange(7),FakePlanner.joint_limits)
        self.assertAlmostEqual(ranked[0][0],.2)
        self.assertAlmostEqual(ranked[0][6],-1.4)

    def test_exact_digest_detects_a_single_float_change(self):
        a={'status':'Success','position':np.zeros((2,7))}
        b={'status':'Success','position':a['position'].copy()};b['position'][1,6]=np.nextafter(0.,1.)
        self.assertNotEqual(result_digest(a),result_digest(b))

    def test_physical_divergence_reports_earliest_object_even_when_tiny(self):
        a=[{'robot_qpos':[0.],'fruit_pose':[[0.,0.,0.]],'leaf_push':[0.]} for _ in range(3)]
        import copy
        b=copy.deepcopy(a);b[1]['fruit_pose'][0][2]=1e-9;b[2]['robot_qpos'][0]=.1
        exact=first_divergence(a,b);material=first_divergence(a,b,1e-6)
        self.assertEqual(exact['after_control_steps'],1)
        self.assertEqual(exact['differences'][0]['object'],'fruit')
        self.assertEqual(material['after_control_steps'],2)

    def test_core_gates_require_full_scale_and_both_q7_measures(self):
        stats={'episodes':200,'approach_end_wrist':{q:{'within_median_plusminus_1rad':{'fraction':.95}} for q in ('q5','q6','q7')},
            'motion':{p:{'q1_swing_p90':.99} for p in ('reach','carry')},
            'q7_commands_outside_pi':{'numerator':0,'denominator':10},'q7_states_outside_pi':{'numerator':0}}
        planning={'determinism':{'passed':True},'continuity':{'passed':True}}
        self.assertTrue(all(core_gates(stats,planning).values()))
        stats['q7_commands_outside_pi']['numerator']=1
        self.assertFalse(core_gates(stats,planning)['q7_command_and_state_range'])
        stats['episodes']=199;self.assertFalse(any(core_gates(stats,planning).values()))

    def test_settle_fallback_uses_strict_three_percent_and_real_attempts(self):
        stats={'by_task':{t:{'success':97,'failed':3} for t in ('count','occluded_ripe')},
            'failed_attempt_records':[{'manifest':'/d/count/ep0_manifest.json',
                'failures':[{'reason':'fruit 0 missed the basket'}]*3}]}
        self.assertFalse(basket_miss_exceeds_three_percent(stats))
        stats['by_task']['count']={'success':96,'failed':4}
        stats['failed_attempt_records'][0]['failures'].append({'reason':'fruit 0 missed the basket'})
        self.assertTrue(basket_miss_exceeds_three_percent(stats))


if __name__=='__main__': unittest.main()
