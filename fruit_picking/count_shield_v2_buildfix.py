"""Constructor-only correction; fixed_v2 geometry and archived source stay intact."""
import numpy as np
import sapien

from .env import material
from .task_v2_env import FruitPickTaskV2Env
from .count_shield_v2_spec import optical_geometry,basket_visual_shifts
from .count_shield_build_records import prebuild_basket_offsets


def load_scene_before_attachment(self,options):
    self.count_shield=None;self.basket_decoration_shifts=[]
    self.basket_build_records_audit=None
    if self.count_shield_spec.mode=='off':
        FruitPickTaskV2Env._load_scene(self,options)
        self.count_shield_boxes=[];self.count_shield_audit_boxes=[]
        return
    shifts=basket_visual_shifts(self.count_shield_spec)
    with prebuild_basket_offsets(self.scene,shifts) as audit:
        FruitPickTaskV2Env._load_scene(self,options)
    assert audit['basket_builds']==1
    self.basket_build_records_audit=audit
    self.basket_decoration_shifts=audit['applied_shifts']
    geometry=optical_geometry(self.layout.basket['center'],self.count_shield_spec,self.layout.basket['height'])
    self.count_shield_boxes=geometry['render'];self.count_shield_audit_boxes=geometry['new_obstacle_audit']
    assert self.layout.basket['inner_half']==.09 and self.layout.basket['height']==.12
    entity=self.basket._objs[0]
    physics=entity.find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
    actual=[np.r_[s.local_pose.p,s.local_pose.q,s.half_size].astype(np.float32) for s in physics.collision_shapes]
    expected=[np.asarray(r['pose']+r['scale'],dtype=np.float32) for r in audit['collision_before']]
    np.testing.assert_array_equal(actual,expected)
    self.basket_collision_signature=[r.tolist() for r in actual]
    body=entity.find_component_by_type(sapien.render.RenderBodyComponent)
    np.testing.assert_array_equal([np.r_[s.local_pose.p,s.local_pose.q] for s in body.render_shapes],
                                 np.asarray([r['pose'] for r in audit['visual_after']],dtype=np.float32))
    builder=self.scene.create_actor_builder()
    for box in self.count_shield_boxes:
        builder.add_box_visual(sapien.Pose(p=box['center']),half_size=box['half'],material=material(self.count_shield_spec.rgb,1.))
    assert not builder.collision_records
    builder.initial_pose=sapien.Pose()
    self.count_shield=builder.build_static(name='count_fixed_optical_shield')
    assert not self.count_shield._objs[0].find_component_by_type(sapien.physx.PhysxRigidStaticComponent).collision_shapes


def install():
    """Patch the implementation in this explicitly launched process only."""
    from .count_shield_v2_env import FruitPickCountShieldV2Env
    FruitPickCountShieldV2Env._load_scene=load_scene_before_attachment
    import fruit_shield_v2_runtime as runtime
    previous=runtime.scene_manifest
    if not getattr(previous,'buildfix_metadata',False):
        def scene_manifest(u):
            value=previous(u)
            value.update(implementation_revision='visual_records_before_attachment_v1',
                         basket_build_records_audit=getattr(u,'basket_build_records_audit',None))
            return value
        scene_manifest.buildfix_metadata=True
        runtime.scene_manifest=scene_manifest
