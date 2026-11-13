"""An independent opt-in environment; all archived environment code stays intact."""
import sapien

from mani_skill.utils.registration import register_env

from .env import material
from .task_v2_env import FruitPickTaskV2Env
from .count_shield_spec import CountShieldSpec, shield_boxes


@register_env('FruitPickCountShield-v0', max_episode_steps=2800)
class FruitPickCountShieldEnv(FruitPickTaskV2Env):
    def __init__(self, *args, count_shield='off', **kwargs):
        self.count_shield_spec = CountShieldSpec(mode=count_shield)
        if count_shield != 'off' and (kwargs.get('task') != 'count'
                or kwargs.get('task_version', 'v1') != 'v2'
                or kwargs.get('task_revision', 8) != 9):
            raise ValueError('Fixed shield requires explicit revision9 Count v2')
        self.count_shield = None
        self.count_shield_boxes = []
        super().__init__(*args, **kwargs)

    def _load_scene(self, options):
        super()._load_scene(options)
        self.count_shield = None
        self.count_shield_boxes = shield_boxes(self.layout.basket['center'], self.count_shield_spec)
        if not self.count_shield_boxes:
            return
        builder = self.scene.create_actor_builder()
        for box in self.count_shield_boxes:
            builder.add_box_visual(sapien.Pose(p=box['center']), half_size=box['half'],
                                   material=material(self.count_shield_spec.rgb, 1.))
        assert not builder.collision_records
        builder.initial_pose = sapien.Pose()
        self.count_shield = builder.build_static(name='count_fixed_optical_shield')
        component = self.count_shield._objs[0].find_component_by_type(sapien.physx.PhysxRigidStaticComponent)
        assert len(component.collision_shapes) == 0
