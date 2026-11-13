"""Process-local R12 false bottom, implemented as render geometry only."""
import numpy as np
import sapien

from .env import material
from .count_shield_r12_spec import CountShieldR12Spec, false_bottom_box


def install():
    """Install over the build-fixed fixed_v2 environment in this child process."""
    from .count_shield_v2_env import FruitPickCountShieldV2Env
    previous = FruitPickCountShieldV2Env._load_scene
    if getattr(previous, 'r12_false_bottom', False):
        return

    def load_scene(self, options):
        previous(self, options)
        if self.count_shield_spec.mode != 'fixed_v2':
            return
        box = false_bottom_box(self.layout.basket['center'],
                               CountShieldR12Spec(mode='fixed_v2_r12'))
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(sapien.Pose(p=box['center']), half_size=box['half'],
                               material=material(tuple(box['rgb']), 1.))
        assert not builder.collision_records
        builder.initial_pose = sapien.Pose()
        actor = builder.build_static(name='count_fixed_false_bottom')
        component = actor._objs[0].find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
        assert len(component.collision_shapes) == 0
        self.count_shield_false_bottom_box = {k: box[k] for k in ('center', 'half')}
        self.count_shield_false_bottom_spec = CountShieldR12Spec(mode='fixed_v2_r12').to_dict()
        self.count_shield_false_bottom_actor = actor
        # Keep this render-only volume in report-only hand/TCP distance checks.
        self.count_shield_boxes = list(self.count_shield_boxes) + [self.count_shield_false_bottom_box]

    load_scene.r12_false_bottom = True
    FruitPickCountShieldV2Env._load_scene = load_scene
