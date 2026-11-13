"""Current R0 contracts follow revision8; revision7 tests remain in commit6eb6197."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import tempfile
import json
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
from fruit_picking.layout import sample_layout
from fruit_picking.task_v2_spec import (TaskV2Spec,task_layout,start_delay,wind_plan,count_outcome,
    bounded_inside,choose_count)
from fruit_picking.task_v2_audit_r8 import appearance_check,fit_clock,filtered_view
from memory_management.fruit_v2_distance import any_hanging,maturity_sensitive


class TaskR8Tests(unittest.TestCase):
    def setUp(self):self.spec=TaskV2Spec(version='v2')

    def test_default_layout_preserves_v1(self):
        self.assertEqual(TaskV2Spec().version,'v1')
        for task in ('count','occluded_ripe'):
            self.assertEqual(task_layout(task,4203200).to_json(),sample_layout(task,4203200).to_json())

    def test_basket_boundaries(self):
        p=np.array([[0,0,.182],[0,0,-.374],[0,0,.30],[0,0,.15],[0,0,.27],[.09,0,.20]])
        self.assertEqual(bounded_inside(p,[0,0,.15],.09,.12).tolist(),[True,False,False,False,False,False])

    def test_budget_always_fails_even_with_N_and_finish_streak(self):
        r=count_outcome(1800,40,2,2,2,False,self.spec)
        self.assertTrue(r['fail']);self.assertFalse(r['success']);self.assertEqual(r['reason'],'budget')
        self.assertTrue(count_outcome(1799,40,2,2,2,False,self.spec)['success'])
        self.assertFalse(count_outcome(300,39,2,2,2,False,self.spec)['terminated'])
        self.assertFalse(count_outcome(300,0,2,2,2,False,self.spec)['success'])

    def test_overpick_and_robot_drop_are_immediate(self):
        self.assertEqual(count_outcome(300,0,2,2,3,False,self.spec)['reason'],'overpick')
        self.assertEqual(count_outcome(300,0,2,2,2,True,self.spec)['reason'],'robot_drop')

    def test_private_random_streams_and_unconstrained_N(self):
        samples=[task_layout('count',4200000+16*e,'v2') for e in range(200,250)]
        self.assertTrue(all(2<=x.n_pick<=5 for x in samples))
        self.assertTrue(any(x.n_pick>len(x.fruits) for x in samples))
        for e in range(200,250):
            seed=4200000+16*e;d=start_delay(seed,self.spec);w=wind_plan(seed,self.spec)
            self.assertTrue(0<=d<=600);self.assertTrue(34<=w['step']<=66);self.assertIn(w['count'],[1,2])
            self.assertEqual(d,start_delay(seed,self.spec))

    def test_count_shortcuts_and_clock_inputs(self):
        tree=[{'id':i,'slot':i,'position':[.4,i*.1,.5]} for i in range(4)]
        self.assertIsNone(choose_count('initial_current',2,tree[:2],tree))
        self.assertEqual(choose_count('initial_ground',2,tree[:2],tree,ground_count=2),1)
        self.assertEqual(choose_count('nonstop',2,tree,visible_basket_count=2),3)
        fit={'a':30.,'b':100.,'bias':0.}
        self.assertIsNone(choose_count('clock',2,tree,current_step=230,clock=fit))
        self.assertEqual(choose_count('clock_onset',2,tree,current_step=230,clock=fit,onset=100),3)
        with self.assertRaises(ValueError):choose_count('clock',2,tree)
        with self.assertRaises(ValueError):choose_count('clock_onset',2,tree,current_step=30,clock=fit)

    def test_pixel_equivalence_exact_and_failure_cases(self):
        rgb=np.full((8,8,3),100,dtype=np.uint8);mask=np.zeros((8,8),bool);mask[2:6,2:6]=True
        result=appearance_check(rgb,{0:mask},rgb.copy(),{0:mask.copy()},set())
        self.assertTrue(result['passed']);self.assertEqual(result['mask_iou'],1);self.assertEqual(result['max_abs_color_diff_255'],0)
        changed=rgb.copy();changed[mask]+=3
        self.assertFalse(appearance_check(rgb,{0:mask},changed,{0:mask},set())['passed'])
        missing=np.zeros_like(mask)
        self.assertFalse(appearance_check(rgb,{0:mask},rgb,{0:missing},set())['passed'])

    def test_signal_without_maturity_and_inactive_exclusion(self):
        p=np.array([[.01,0,0],[.1,0,0],[0,0,0]])
        self.assertEqual(any_hanging([0,0,0],p,[True,True,False]),.01)
        self.assertEqual(maturity_sensitive([0,0,0],p,[False,True,True],[True,True,False]),.1)
        self.assertEqual(any_hanging([0,0,0],p,[False,False,False]),10.)

    def test_clock_fit_uses_calibration_only_and_onset_variant(self):
        cases=[]
        for e in range(250,260):
            onset=30+25*(e-250);N=2+(e-250)%4
            cases.append({'N':N,'green_step':onset,'reference_success':True,
                'basket_entry_steps':[onset+50+100*k for k in range(1,N+1)],
                'decision_points':[{'step':onset+50+100*k,'completed_releases':k} for k in range(N+1)]})
        relative=fit_clock(cases,True);absolute=fit_clock(cases,False)
        self.assertAlmostEqual(relative['a'],50.,places=8);self.assertAlmostEqual(relative['b'],100.,places=8)
        self.assertEqual(relative['calibration_successes'],10);self.assertEqual(absolute['calibration_episodes'],10)
        self.assertEqual(relative['time_basis'],'t-green_step')

    def test_spare_entity_ids_and_order_never_reach_selector(self):
        u=SimpleNamespace(task='count',v2_spec=self.spec,v2_color_alpha=0.,lamp_green=True,
            present=np.array([True,True,True,False]),slot_of=[0,1,0,1],occluders=[],
            layout=SimpleNamespace(fruits=[{'ripe':True,'radius':.026} for _ in range(4)],occluders=[]),
            fruits=[SimpleNamespace(per_scene_id=[i]) for i in [11,22,33,44]],
            work_lamp=SimpleNamespace(per_scene_id=[99]),
            fruit_positions=lambda:np.array([[-.06,-.34,.2],[.4,-.1,.5],[.4,.1,.5],[100,100,10]]),
            in_basket=lambda:np.array([True,False,False,False]),hanging=lambda:np.array([False,True,True,False]))
        counts={'front_camera':{22:100,33:100,99:80},'hand_camera':{}}
        v=filtered_view(u,counts)
        self.assertEqual([f['id'] for f in v['tree']],[0,1])
        self.assertEqual([f['slot'] for f in v['tree']],[0,1])
        self.assertTrue(v['lamp_green']);self.assertEqual(v['visible_basket_count'],0)

    def test_audit_gate_and_reporting_variant_boundaries(self):
        from fruit_r8_r0_support import gate
        rows=[{'success':i<15,'guard_checked_steps':100,'guard_clipped_steps':0} for i in range(50)]
        self.assertIsNone(gate('count','clock',rows[:-1]));self.assertTrue(gate('count','clock',rows))
        rows[15]['success']=True;self.assertFalse(gate('count','clock',rows))
        self.assertTrue(gate('count','clock_onset',rows))
        for i,r in enumerate(rows):r['success']=i<45
        self.assertTrue(gate('count','history',rows))
        rows[0]['guard_clipped_steps']=25;self.assertFalse(gate('count','history',rows))

    def test_default_entry_only_prepares_and_never_starts_GPU_work(self):
        import run_fruit_task_v2_r8_r0 as runner
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'reports').mkdir()
            (root/'reports/lowlevel_rescue_status.json').write_text('{"links":{}}')
            state={'state':'preparing_r8'}
            with patch.object(runner,'OUT',root),patch.object(runner,'ECM',root),\
                 patch.object(runner,'prepare',return_value=(state,{})),patch.object(runner,'report'),\
                 patch.object(runner.subprocess,'Popen') as start,patch.object(sys,'argv',['runner']):
                runner.main()
            start.assert_not_called()
            self.assertEqual(json.loads((root/'status.json').read_text())['state'],'ready_for_r8_gpu_checks')


try:
    from fruit_picking.task_v2_env import FruitPickTaskV2Env
    import torch
except ModuleNotFoundError:
    FruitPickTaskV2Env=None


@unittest.skipIf(FruitPickTaskV2Env is None,'Native simulator package required')
class NativePoolTests(unittest.TestCase):
    def fixture(self):
        u=FruitPickTaskV2Env.__new__(FruitPickTaskV2Env);u.v2_spec=TaskV2Spec(version='v2')
        poses=np.array([[-.06,-.34,.19],[102.,100.,10.],[104.,100.,10.]])
        class Shape:
            def set_collision_groups(self,g):self.collision_groups=tuple(g)
        class Actor:
            def __init__(self,i):self.i=i;self.body=SimpleNamespace(kinematic=False,collision_shapes=[Shape()])
            def set_pose(self,pose):poses[self.i]=pose.p[0].cpu().numpy()
        u.layout=SimpleNamespace(basket={'center':[-.06,-.34,.15],'inner_half':.09,'height':.12})
        u.fruits=[Actor(i) for i in range(3)];u.fruit_positions=lambda:poses
        u._body=lambda a:a.body;u.present=np.array([True,False,False]);u.attached=[True,True,True]
        u.wind_detached=[False]*3;u.replenished=set();u.pending_replenish={};u.replenish_events=[]
        u.receipt_streak=np.zeros(3,dtype=int);u.reserve_remaining=[[1,2]];u.slot_of=[0,0,0]
        u.original_geometry=[{'pos':[.4,0,.5]}];u.reserve_collision_groups={1:[(1,1,0,0)],2:[(1,1,0,0)]}
        u.grasp_streak=[0]*3;u.hanging_slot=[None]
        return u,poses

    def test_two_step_receipt_and_TCP_delay(self):
        u,p=self.fixture()
        for t in [1,2,3]:u.control_steps=t;u._replenish(set(),np.array([.41,0,.5]))
        self.assertFalse(u.replenish_events)
        u.control_steps=4;u._replenish(set(),np.array([.6,0,.5]))
        self.assertEqual(len(u.replenish_events),1);self.assertEqual(u.replenish_events[0]['delay_steps'],2)
        self.assertTrue(u.present[1]);self.assertFalse(u.attached[1]);self.assertTrue(u.attached[2])
        self.assertTrue(u.in_basket()[0]);np.testing.assert_allclose(p[1],[.4,0,.5],atol=1e-7)
        self.assertEqual(u.fruits[1].body.collision_shapes[0].collision_groups,(1,1,0,0))
        u.control_steps=5;u._replenish(set(),np.array([.6,0,.5]));self.assertEqual(len(u.replenish_events),1)

    def test_windfall_never_replenishes_and_inactive_never_counts(self):
        u,p=self.fixture();u.wind_detached[0]=True
        for t in [1,2,3]:u.control_steps=t;u._replenish(set(),np.array([.6,0,.5]))
        self.assertFalse(u.replenish_events)
        p[2]=[-.06,-.34,.19];self.assertFalse(u.in_basket()[2])


if __name__=='__main__':unittest.main()
