"""Independent fixed-v2 optical repair; old environments and collisions stay intact."""
import numpy as np
import sapien

from mani_skill.utils.registration import register_env

from .env import material
from .task_v2_env import FruitPickTaskV2Env
from .count_shield_v2_spec import CountShieldV2Spec, optical_geometry, basket_visual_shifts


@register_env('FruitPickCountShieldV2-v0',max_episode_steps=2800)
class FruitPickCountShieldV2Env(FruitPickTaskV2Env):
    def __init__(self,*args,count_shield='off',**kwargs):
        self.count_shield_spec=CountShieldV2Spec(mode=count_shield)
        if count_shield!='off' and (kwargs.get('task')!='count' or kwargs.get('task_version','v1')!='v2'
                                    or kwargs.get('task_revision',8)!=9):
            raise ValueError('fixed_v2 requires explicit revision9 Count v2')
        self.count_shield=None;self.count_shield_boxes=[];self.count_shield_audit_boxes=[]
        self.basket_decoration_shifts=[]
        super().__init__(*args,**kwargs)

    def _load_scene(self,options):
        super()._load_scene(options)
        self.count_shield=None;self.basket_decoration_shifts=[]
        geometry=optical_geometry(self.layout.basket['center'],self.count_shield_spec,self.layout.basket['height'])
        self.count_shield_boxes=geometry['render'];self.count_shield_audit_boxes=geometry['new_obstacle_audit']
        if not self.count_shield_boxes:return
        assert self.layout.basket['inner_half']==.09 and self.layout.basket['height']==.12
        physics=self.basket._objs[0].find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
        signature=lambda:[np.r_[s.local_pose.p,s.local_pose.q,s.half_size].tolist() for s in physics.collision_shapes]
        before=signature()
        body=self.basket._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
        shapes=body.render_shapes;assert len(shapes)==5
        for shape,shift in zip(shapes,basket_visual_shifts(self.count_shield_spec)):
            old=shape.local_pose
            shape.local_pose=sapien.Pose(p=np.asarray(old.p)+shift,q=old.q)
            self.basket_decoration_shifts.append(shift)
        assert signature()==before,'Moving decoration must not alter basket collisions'
        self.basket_collision_signature=before
        builder=self.scene.create_actor_builder()
        for box in self.count_shield_boxes:
            builder.add_box_visual(sapien.Pose(p=box['center']),half_size=box['half'],
                                   material=material(self.count_shield_spec.rgb,1.))
        assert not builder.collision_records
        builder.initial_pose=sapien.Pose()
        self.count_shield=builder.build_static(name='count_fixed_optical_shield')
        physics=self.count_shield._objs[0].find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
        assert len(physics.collision_shapes)==0
