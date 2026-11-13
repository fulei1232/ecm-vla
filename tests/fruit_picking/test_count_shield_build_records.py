"""Construction-order regression using real SAPIEN poses and ManiSkill records."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import sapien
from sapien.wrapper.actor_builder import VisualShapeRecord,CollisionShapeRecord
from mani_skill.utils.building.actor_builder import ActorBuilder

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT)]
from fruit_picking.count_shield_build_records import prebuild_basket_offsets,record_signature
from fruit_picking.count_shield_v2_spec import CountShieldV2Spec,basket_boxes,basket_visual_shifts


class RecordScene:
    """Real builder and records; only final GPU entity creation is substituted."""
    def __init__(self,fail=False):self.built=[];self.fail=fail
    def create_actor_builder(self):
        builder=ActorBuilder()
        for box in basket_boxes([0,0,0]):
            builder.visual_records.append(VisualShapeRecord(type='box',pose=sapien.Pose(p=box['center']),scale=tuple(box['half'])))
            builder.collision_records.append(CollisionShapeRecord(type='box',pose=sapien.Pose(p=box['center']),scale=tuple(box['half'])))
        def final_build(name):
            # Called by the real ActorBuilder.build_static after interception.
            if self.fail:raise RuntimeError('deliberate entity-build failure')
            value={'name':name,'visual':record_signature(builder.visual_records),
                   'collision':record_signature(builder.collision_records),'type':builder.physx_body_type}
            self.built.append(value);return value
        builder.build=final_build
        return builder


class BuildRecordTests(unittest.TestCase):
    def test_real_builder_receives_offsets_before_static_build_without_collision_changes(self):
        scene=RecordScene();shifts=basket_visual_shifts(CountShieldV2Spec(mode='fixed_v2'))
        original=scene.create_actor_builder
        with prebuild_basket_offsets(scene,shifts) as audit:
            builder=scene.create_actor_builder();before=record_signature(builder.visual_records)
            result=builder.build_static(name='basket')
            self.assertEqual(result['type'],'static')
            self.assertEqual(result['collision'],audit['collision_before'])
            for old,new,shift in zip(before,result['visual'],shifts):
                expected=sapien.Pose(p=np.asarray(old['pose'][:3])+shift,q=old['pose'][3:])
                np.testing.assert_array_equal(new['pose'],np.r_[expected.p,expected.q])
            self.assertEqual(audit['basket_builds'],1)
        self.assertEqual(scene.create_actor_builder,original)
        self.assertNotIn('create_actor_builder',vars(scene))

    def test_other_actors_and_collision_records_are_untouched(self):
        scene=RecordScene();shifts=basket_visual_shifts(CountShieldV2Spec(mode='fixed_v2'))
        with prebuild_basket_offsets(scene,shifts) as audit:
            builder=scene.create_actor_builder();before=record_signature(builder.visual_records)
            result=builder.build_static('work_lamp')
            self.assertEqual(before,result['visual']);self.assertEqual(audit['basket_builds'],0)

    def test_factory_restores_on_build_exception(self):
        scene=RecordScene(fail=True);original=scene.create_actor_builder
        with self.assertRaisesRegex(RuntimeError,'deliberate entity-build failure'):
            with prebuild_basket_offsets(scene,basket_visual_shifts(CountShieldV2Spec(mode='fixed_v2'))):
                scene.create_actor_builder().build_static('basket')
        self.assertEqual(scene.create_actor_builder,original)
        self.assertNotIn('create_actor_builder',vars(scene))

    def test_geometry_is_identical_and_old_68_sources_are_preserved(self):
        from v_a_common import read_json,sha256
        from fruit_shield_buildfix_common import OLD_V2,activate,OUT,DATASET
        old=read_json(OLD_V2/'plan.json');self.assertEqual(len(old['source_sha256']),68)
        for name,h in old['source_sha256'].items():self.assertEqual(sha256(name),h,name)
        activate()
        import fruit_r9_generation as gen
        import fruit_r9_data as rec
        self.assertEqual(gen.OUT,OUT);self.assertEqual(gen.DATASET,DATASET);self.assertEqual(rec.OUT,OUT)
        from fruit_picking.count_shield_v2_spec import optical_geometry
        geometry=optical_geometry([-.06,-.34,.15],CountShieldV2Spec(mode='fixed_v2'))
        self.assertEqual(len(geometry['render']),14);self.assertEqual(len(geometry['new_obstacle_audit']),13)


if __name__=='__main__':unittest.main()
