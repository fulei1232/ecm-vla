"""Only new enclosure geometry, port clearance, bindings and legacy preservation."""
from pathlib import Path
import ast
import json
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT)]
from fruit_picking.count_shield_spec import CountShieldSpec, shield_boxes, sphere_intersections, segment_intersects_boxes


class ShieldTests(unittest.TestCase):
    def test_default_off_and_explicit_enable(self):
        self.assertEqual(shield_boxes([-.06,-.34,.15],CountShieldSpec()),[])
        with self.assertRaises(ValueError):CountShieldSpec(mode='automatic')
        tree=ast.parse((ROOT/'fruit_picking/count_shield_env.py').read_text())
        init=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        defaults=dict(zip([a.arg for a in init.args.kwonlyargs],init.args.kw_defaults))
        self.assertEqual(ast.literal_eval(defaults['count_shield']),'off')

    def test_fixed_port_remains_open_and_shell_is_closed_elsewhere(self):
        boxes=shield_boxes([-.06,-.34,.15],CountShieldSpec(mode='fixed_v1'))
        self.assertEqual(len(boxes),13)
        points=[[-.06,-.34,.363],[.035,-.34,.363],[-.06,-.34,.097],[.083,-.34,.25],[.4,0,.5]]
        self.assertEqual(sphere_intersections(points,[.029]*5,boxes).tolist(),[False,True,True,True,False])

    def test_port_baffle_blocks_oblique_sight_line_to_far_rim(self):
        boxes=shield_boxes([-.06,-.34,.15],CountShieldSpec(mode='fixed_v1'))
        camera=[-.03816,-.29404,.51497]
        far_rim=[-.06,-.433,.27]
        self.assertFalse(segment_intersects_boxes(camera,far_rim,boxes[:9]))
        self.assertTrue(segment_intersects_boxes(camera,far_rim,boxes))

    def test_original_failed_prefix_has_drop_port_clearance(self):
        from fruit_picking.layout import sample_layout
        layout=sample_layout('count',4101792);M=len(layout.fruits);N=7
        boxes=shield_boxes(layout.basket['center'],CountShieldSpec(mode='fixed_v1'))
        source=ROOT/'results/fruit_task_v2_R0_r10_20261106/replays/train_dart/count/history/ep112_attempt0/demo_gt.jsonl'
        for line in source.read_text().splitlines():
            row=json.loads(line);fruits=[(i,f) for i,f in enumerate(row['fruits']) if f['present']]
            radii=[layout.fruits[i if i<M else (i-M)//N]['radius'] for i,f in fruits]
            self.assertFalse(sphere_intersections([f['pos'] for i,f in fruits],radii,boxes).any(),row['timestep'])

    def test_new_bindings_never_point_to_archived_results_or_data(self):
        import fruit_shield_common as common
        support=common.activate()
        import fruit_r9_generation as generation
        import fruit_r9_data as recorder
        import fruit_r10_data_timing as timing
        self.assertEqual(support.OUT,common.OUT)
        self.assertEqual(generation.OUT,common.OUT)
        self.assertEqual(generation.DATASET,common.DATASET)
        self.assertEqual(recorder.OUT,common.OUT)
        self.assertEqual(timing.OUT,common.OUT)
        self.assertNotEqual(common.OUT,common.PREVIOUS)

    def test_all_48_archived_source_files_are_unchanged(self):
        from v_a_common import sha256,read_json
        plan=read_json(ROOT/'results/fruit_task_v2_R0_r10_20261106/plan.json')
        self.assertEqual(len(plan['source_sha256']),48)
        for name,h in plan['source_sha256'].items():self.assertEqual(sha256(name),h,name)


if __name__=='__main__':unittest.main()
