"""CPU checks for new R11 rules only; no previously accepted replay is rerun."""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ECM = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ECM/'scripts'), str(ECM), str(ECM/'src')]
from fruit_r11_rules import hidden_content_visible, intersection_stop, AttemptLedger


class Revision11Tests(unittest.TestCase):
    def test_body_is_descriptive_but_content_or_reserve_remains_fatal(self):
        self.assertFalse(hidden_content_visible({'front': {'basket': 999, 'basket_fruit': 0, 'inactive_reserve': 0}}))
        for key in ('basket_fruit', 'inactive_reserve'):
            self.assertTrue(hidden_content_visible({'front': {'basket': 0}, 'hand': {key: 1}}))

    def test_exact_global_stop_boundaries_and_both_conditions(self):
        self.assertFalse(intersection_stop(2, 2))
        self.assertFalse(intersection_stop(3, 150))
        self.assertTrue(intersection_stop(3, 149))
        self.assertTrue(intersection_stop(4, 199))
        self.assertFalse(intersection_stop(4, 200))

    def test_concurrent_attempts_count_once_and_failure_blocks_new_starts(self):
        snapshots = []
        ledger = AttemptLedger(lambda v:snapshots.append(v))
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i:ledger.start(str(i), {'task': 'count' if i%2 else 'occluded_ripe'}), range(150)))
        self.assertEqual(ledger.snapshot()['started_attempts'], 150)
        self.assertFalse(ledger.finish('0','shield_intersection',False))
        self.assertFalse(ledger.finish('1','shield_intersection',False))
        self.assertFalse(ledger.finish('2','shield_intersection',False))
        self.assertTrue(ledger.finish('3','shield_intersection',False))
        with self.assertRaises(RuntimeError):ledger.start('151', {})
        restored = AttemptLedger(lambda v:None, snapshots[-1])
        self.assertTrue(restored.stopped)
        self.assertEqual(restored.snapshot()['started_attempts'], 150)
        with self.assertRaises(RuntimeError):restored.finish('0','robot_drop',False)

    def test_old_validity_audit_changes_only_body_predicate_and_name(self):
        from fruit_r11_validity import audit_unchanged_fields
        import inspect
        old = ast.parse((ECM/'scripts/fruit_r9_r0_support.py').read_text())
        node = next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='audit_case')
        source = ast.get_source_segment((ECM/'scripts/fruit_r9_r0_support.py').read_text(),node)
        expected = source.replace('def audit_case(', 'def audit_unchanged_fields(').replace(
            "('basket', 'basket_fruit', 'inactive_reserve')", "('basket_fruit', 'inactive_reserve')")
        self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(inspect.getsource(audit_unchanged_fields))))

    def test_exact_finger_and_tcp_distances_and_no_shield(self):
        import torch
        from fruit_r11_metrics import GripperDistance
        urdf = ECM/'fruit_picking/assets/rm75b_eg2/rm75b_eg2.urdf'
        def pose(x):
            t=torch.eye(4,dtype=torch.float64)[None];t[0,0,3]=x
            return SimpleNamespace(to_transformation_matrix=lambda:t,
                q=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64))
        agent = SimpleNamespace(tcp_pose=SimpleNamespace(p=torch.tensor([[3.,0.,0.]],dtype=torch.float64)),
            finger1_link=SimpleNamespace(pose=pose(2.)),finger2_link=SimpleNamespace(pose=pose(2.)))
        geometry = GripperDistance([{'center':[0.,0.,0.],'half':[1.,1.,1.]}],urdf)
        result = geometry.measure(agent)
        # The bundled native FCL build returns float32-scale distance values.
        # This fixture tolerates 0.2 micrometres; task collision thresholds are unchanged.
        self.assertAlmostEqual(result['minimum_distance_m'], .992, delta=2e-7)
        self.assertFalse(result['entered'])
        agent.finger1_link.pose = pose(0.)
        self.assertTrue(geometry.measure(agent)['finger_entered'])
        agent.finger1_link.pose=pose(2.);agent.tcp_pose.p[0,0]=0.
        self.assertTrue(geometry.measure(agent)['tcp_entered'])
        off=GripperDistance([],urdf).measure(agent)
        self.assertIsNone(off['minimum_distance_m']);self.assertFalse(off['entered'])

    def test_frame_metrics_keep_body_pixels_and_fatal_content_evidence(self):
        from fruit_r11_metrics import FrameMetrics
        from v_a_common import read_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            distance=SimpleNamespace(measure=lambda agent:{'entered':True,'finger_entered':True,
                'tcp_entered':False,'minimum_distance_m':0.,'finger_distance_m':0.,'tcp_distance_m':.1})
            m=FrameMetrics(Path(tmp),distance)
            replay=SimpleNamespace(u=SimpleNamespace(control_steps=0,agent=None), snapshots={})
            replay.capture=lambda stage,replace:replay.snapshots.update({stage:{'files':{'front_camera':'peak.png'}}})
            m.observe(replay,{'front_camera':{'basket':4,'basket_fruit':0,'inactive_reserve':0}})
            m.observe(replay,{'front_camera':{'basket':4,'basket_fruit':0,'inactive_reserve':0}})
            replay.u.control_steps=1
            m.observe(replay,{'front_camera':{'basket':0,'basket_fruit':1,'inactive_reserve':0}})
            m.log.close();s=m.summary()
            self.assertEqual(s['observations'],2)
            self.assertEqual(s['basket_body']['visible_frames_either_camera'],1)
            self.assertEqual(s['basket_body']['total_pixels_both_cameras'],4)
            self.assertEqual(s['basket_body']['by_camera']['front_camera']['max_basket_fruit_pixels'],1)
            self.assertEqual(s['gripper_shield']['entered_frames'],2)
            self.assertEqual(len(read_jsonl(Path(tmp)/'frame_metrics.jsonl')),2)

    def test_data_and_R0_intersection_branch_and_formal_env_bindings(self):
        from fruit_r11_runtime import Revision11Replay
        from fruit_r9_episode_support import EpisodeEnd,CheckFailed
        with tempfile.TemporaryDirectory() as tmp:
            replay=Revision11Replay.__new__(Revision11Replay)
            replay.u=SimpleNamespace(count_shield_audit_boxes=[{'center':[0,0,0],'half':[.1,.1,.1]}],
                present=np.array([True]),fruit_positions=lambda:np.array([[0.,0.,0.]]),
                layout=SimpleNamespace(fruits=[{'radius':.02}]),control_steps=31)
            replay.directory=Path(tmp);replay.obs=None;replay.shield_intersections_checked=0
            replay.visibility=lambda:{};replay.capture=lambda stage:None
            replay.metrics=SimpleNamespace(observe=lambda *args:None)
            replay.fail=lambda reason,details:(_ for _ in ()).throw(CheckFailed(reason))
            replay.is_data=False
            with patch('fruit_r11_runtime.R9Replay.record'),patch('fruit_r11_runtime.save_forensics'):
                with self.assertRaises(CheckFailed):replay.record('carry')
                replay.is_data=True
                with self.assertRaises(EpisodeEnd):replay.record('carry')
            self.assertFalse(replay.success);self.assertTrue(replay.ended)
            self.assertEqual(replay.reason,'shield_intersection')
            self.assertEqual(json.loads((Path(tmp)/'attempt_failure.json').read_text())['actors'],[0])
        from fruit_r11_common import activate,OUT,DATASET,OLD
        activate()
        import fruit_r9_generation as gen
        import fruit_r9_data as rec
        self.assertEqual(gen.OUT,OUT);self.assertEqual(gen.DATASET,DATASET);self.assertEqual(rec.OUT,OUT)
        from v_a_common import read_json,sha256
        previous=read_json(OLD/'plan.json')['source_sha256']
        self.assertEqual(len(previous),76)
        for name,expected in previous.items():self.assertEqual(sha256(name),expected,name)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Revision11Tests))
    from fruit_r11_common import proof
    from v_a_common import timestamp
    proof('revision11_tests.json',{'passed':result.wasSuccessful(),'tests_run':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'created_utc':timestamp(),
        'scope':'New R11 rules, concurrent ledger, native FCL distances, metrics, unchanged old validity and 76 sources; no GPU replay',
        'accepted_previous_checks_repeated':False})
    sys.exit(0 if result.wasSuccessful() else 1)
