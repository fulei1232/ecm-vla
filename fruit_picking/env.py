"""`FruitPick-v1`: a RealMan arm (RM75-B by default) + EG2 gripper picks fruits that hang from a tree into a basket (ManiSkill3, CPU sim).

The scene is rebuilt from `layout.sample_layout(task, seed)` on every `reset(seed=...)`. Fruits are kinematic
(fixed to their stems) until both jaws hold them for `DETACH_STEPS` consecutive control steps, after which they are
dynamic and can be carried off. The occluding leaf clusters of `occluded_ripe` swing down in front of their fruits
during `layout.gust` (control steps, counted from reset). They have no collision; instead the gripper pushes them
aside (up) as the tcp comes within `PUSH_START` of a cluster and they spring back when it leaves, so finding a
hidden fruit takes memory but the final approach is visible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sapien
import torch
import trimesh

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from . import agent as rm_agent
from .layout import FRONT_CAMERA_EYE, FRONT_CAMERA_FOV, FRONT_CAMERA_TARGET, GROUND_Z, Layout, sample_layout

LEAF_MESH = Path(__file__).resolve().parent / "assets/leaf.obj"
DETACH_STEPS = 2
PUSH_START, PUSH_FULL, PUSH_LIFT = 0.15, 0.07, 0.10
BARK = (0.36, 0.24, 0.14)
STEM = (0.30, 0.38, 0.12)
SOIL = (0.42, 0.34, 0.24)
SKY = (0.70, 0.80, 0.90)
WICKER = (0.62, 0.46, 0.26)
STEEL = (0.45, 0.47, 0.50)


def leaf_mesh() -> str:
    """Flat pointed leaf, 8 cm x 4 cm, double sided, in its local xy plane."""
    if not LEAF_MESH.exists():
        t = np.linspace(-1, 1, 25)
        half_width = 0.02 * (1 - t**2) ** 0.7 * (1 - 0.25 * t)
        outline = np.concatenate([np.stack([0.04 * t, half_width], 1), np.stack([0.04 * t[::-1], -half_width[::-1]], 1)[1:-1]])
        verts = np.vstack([[0.0, 0.0, 0.0], np.column_stack([outline, np.zeros(len(outline))])])
        ring = np.arange(1, len(verts))
        front = [(0, a, b) for a, b in zip(ring, np.roll(ring, -1))]
        faces = front + [(0, b, a) for _, a, b in front]
        trimesh.Trimesh(verts, faces, process=False).export(LEAF_MESH)
    return str(LEAF_MESH)


def material(rgb, roughness=0.8):
    return sapien.render.RenderMaterial(base_color=[*rgb, 1.0], roughness=roughness)


def segment_pose(a, b):
    """Pose whose x axis runs from a to b (SAPIEN cylinders and capsules lie along x)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    x = (b - a) / np.linalg.norm(b - a)
    helper = np.array([0, 0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0, 0])
    y = np.cross(helper, x)
    y /= np.linalg.norm(y)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, np.cross(x, y)], 1)
    T[:3, 3] = (a + b) / 2
    return sapien.Pose(T), float(np.linalg.norm(b - a) / 2)


@register_env("FruitPick-v1", max_episode_steps=2000)
class FruitPickEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["rm75b_eg2", "rm65b_eg2"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    agent: rm_agent.RealManEG2

    def __init__(self, *args, task="occluded_ripe", robot_uids="rm75b_eg2", **kwargs):
        self.task = task
        self.layout: Layout | None = None
        kwargs.setdefault("reconfiguration_freq", 1)
        kwargs.setdefault("reward_mode", "sparse")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=FRONT_CAMERA_EYE.tolist(), target=FRONT_CAMERA_TARGET.tolist())
        return [CameraConfig("front_camera", pose, 256, 256, FRONT_CAMERA_FOV, 0.01, 10)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.25, -1.1, 0.95], target=[0.35, 0.0, 0.45])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 10)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose())

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.45, 0.45, 0.45])
        self.scene.add_directional_light([0.5, 0.3, -1], [1.0, 0.97, 0.9], shadow=self.enable_shadow,
                                         shadow_scale=3, shadow_map_size=2048)
        self.scene.add_directional_light([-0.5, -0.2, -1], [0.35, 0.35, 0.4])

    def _load_scene(self, options: dict):
        seed = int(self._episode_seed[0])
        self.layout = options.get("layout") or sample_layout(self.task, seed)
        L = self.layout

        c = np.asarray(L.basket["center"])
        builder = self.scene.create_actor_builder()
        builder.add_plane_collision(sapien.Pose(p=[0, 0, GROUND_Z], q=[0.7071068, 0, -0.7071068, 0]))
        builder.add_box_visual(sapien.Pose(p=[0.5, 0, GROUND_Z - 0.01]), half_size=[2.5, 2.5, 0.01], material=material(SOIL))
        for p, half in (([2.4, 0, 1.0], [0.02, 3.0, 1.6]), ([-2.4, 0, 1.0], [0.02, 3.0, 1.6]),
                        ([0, 2.4, 1.0], [3.0, 0.02, 1.6]), ([0, -2.4, 1.0], [3.0, 0.02, 1.6])):
            builder.add_box_visual(sapien.Pose(p=p), half_size=half, material=material(SKY, 1.0))
        for p, half in (([-0.02, 0, GROUND_Z / 2 - 0.001], [0.16, 0.14, -GROUND_Z / 2]),
                        ([c[0], c[1], (GROUND_Z + c[2]) / 2], [0.03, 0.03, (c[2] - GROUND_Z) / 2])):
            builder.add_box_collision(sapien.Pose(p=p), half_size=half)
            builder.add_box_visual(sapien.Pose(p=p), half_size=half, material=material(STEEL, 0.5))
        builder.initial_pose = sapien.Pose()
        self.ground = builder.build_static(name="ground")

        leaf = leaf_mesh()
        builder = self.scene.create_actor_builder()
        tx, ty = L.trunk["xy"]
        pose, half = segment_pose([tx, ty, L.trunk["bottom"]], [tx, ty, L.trunk["top"]])
        builder.add_cylinder_collision(pose, radius=L.trunk["radius"], half_length=half)
        builder.add_cylinder_visual(pose, radius=L.trunk["radius"], half_length=half, material=material(BARK))
        for twig, fruit in zip(L.twigs, L.fruits):
            pose, half = segment_pose(twig["start"], twig["end"])
            builder.add_capsule_collision(pose, radius=twig["radius"], half_length=half)
            builder.add_capsule_visual(pose, radius=twig["radius"], half_length=half, material=material(BARK))
            top = np.asarray(fruit["pos"]) + [0, 0, fruit["radius"] - 0.004]
            pose, half = segment_pose(top, twig["stem_top"])
            builder.add_cylinder_visual(pose, radius=0.0025, half_length=half, material=material(STEM))
        for l in L.leaves:
            builder.add_visual_from_file(leaf, sapien.Pose(p=l["pos"], q=l["quat"]), scale=[l["scale"]] * 3,
                                         material=material(l["rgb"], 0.6))
        builder.initial_pose = sapien.Pose()
        self.tree = builder.build_static(name="tree")

        self.fruits = []
        for i, f in enumerate(L.fruits):
            builder = self.scene.create_actor_builder()
            phys = sapien.physx.PhysxMaterial(static_friction=1.0, dynamic_friction=1.0, restitution=0.1)
            builder.add_sphere_collision(radius=f["radius"], material=phys, density=800)
            builder.add_sphere_visual(radius=f["radius"], material=material(f["rgb"], 0.45))
            builder.add_sphere_visual(sapien.Pose(p=[0, 0, f["radius"] - 0.003]), radius=0.006, material=material(STEM))
            builder.initial_pose = sapien.Pose(p=f["pos"])
            self.fruits.append(builder.build_kinematic(name=f"fruit_{i}"))

        self.occluders = []
        for k, occ in enumerate(L.occluders):
            builder = self.scene.create_actor_builder()
            n = np.asarray(occ["normal"])
            u = np.cross([0, 0, 1.0], n)
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            for l in occ["leaves"]:
                du, dv, dn = l["offset"]
                builder.add_visual_from_file(leaf, sapien.Pose(p=du * u + dv * v + dn * n, q=l["quat"]),
                                             scale=[l["scale"]] * 3, material=material(l["rgb"], 0.6))
            builder.initial_pose = sapien.Pose(p=occ["open"])
            self.occluders.append(builder.build_kinematic(name=f"occluder_{k}"))

        h, wall = L.basket["inner_half"], 0.006
        builder = self.scene.create_actor_builder()
        boxes = [([0, 0, wall / 2], [h + wall, h + wall, wall / 2])]
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            half = [wall / 2 if sx else h + wall, wall / 2 if sy else h + wall, L.basket["height"] / 2]
            boxes.append(([sx * (h + wall / 2), sy * (h + wall / 2), L.basket["height"] / 2], half))
        for p, half in boxes:
            builder.add_box_collision(sapien.Pose(p=p), half_size=half)
            builder.add_box_visual(sapien.Pose(p=p), half_size=half, material=material(WICKER))
        builder.initial_pose = sapien.Pose(p=c)
        self.basket = builder.build_static(name="basket")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        L = self.layout
        qpos = torch.as_tensor(self.agent.keyframes["rest"].qpos, dtype=torch.float32)
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose())
        for actor, f in zip(self.fruits, L.fruits):
            actor.set_pose(Pose.create_from_pq(torch.tensor(f["pos"], dtype=torch.float32)))
        for actor, occ in zip(self.occluders, L.occluders):
            actor.set_pose(Pose.create_from_pq(torch.tensor(occ["open"], dtype=torch.float32)))
        self.control_steps = 0
        self.attached = [False] * len(self.fruits)
        self.grasp_streak = [0] * len(self.fruits)
        self.occluder_push = [0.0] * len(self.occluders)

    def _body(self, actor):
        return actor._objs[0].find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)

    def _after_control_step(self):
        self.control_steps += 1
        g0, g1 = self.layout.gust
        alpha = float(np.clip((self.control_steps - g0) / (g1 - g0), 0.0, 1.0))
        alpha = alpha * alpha * (3 - 2 * alpha)
        tcp = self.agent.tcp_pose.p[0].cpu().numpy()
        for k, (actor, occ) in enumerate(zip(self.occluders, self.layout.occluders)):
            p = (1 - alpha) * np.asarray(occ["open"]) + alpha * np.asarray(occ["closed"])
            push = float(np.clip((PUSH_START - np.linalg.norm(tcp - p)) / (PUSH_START - PUSH_FULL), 0.0, 1.0))
            self.occluder_push[k] = push * push * (3 - 2 * push)
            p = p + [0, 0, PUSH_LIFT * self.occluder_push[k]]
            actor.set_pose(Pose.create_from_pq(torch.tensor(p, dtype=torch.float32)))
        for i, actor in enumerate(self.fruits):
            if self.attached[i]:
                continue
            self.grasp_streak[i] = self.grasp_streak[i] + 1 if bool(self.agent.is_grasping(actor)[0]) else 0
            if self.grasp_streak[i] >= DETACH_STEPS:
                body = self._body(actor)
                body.kinematic = False
                body.linear_velocity = [0, 0, 0]
                body.angular_velocity = [0, 0, 0]
                self.attached[i] = True

    def fruit_positions(self) -> np.ndarray:
        return np.stack([a.pose.p[0].cpu().numpy() for a in self.fruits])

    def in_basket(self) -> np.ndarray:
        L = self.layout
        rel = self.fruit_positions() - np.asarray(L.basket["center"])
        return (np.abs(rel[:, 0]) < L.basket["inner_half"]) & (np.abs(rel[:, 1]) < L.basket["inner_half"]) \
            & (rel[:, 2] < L.basket["height"])

    def evaluate(self):
        L = self.layout
        inside = self.in_basket()
        pos = self.fruit_positions()
        dropped = np.array(self.attached) & ~inside & (pos[:, 2] < 0.06)
        targets = np.zeros(len(self.fruits), bool)
        targets[L.targets] = True
        if self.task == "occluded_ripe":
            success = bool(inside[targets].all() and not inside[~targets].any())
            fail = bool(inside[~targets].any() or dropped.any())
        else:
            success = int(inside.sum()) == L.n_pick
            fail = int(inside.sum()) > L.n_pick or bool(dropped.any())
        as_t = lambda x: torch.tensor([x], device=self.device)
        return {"success": as_t(success), "fail": as_t(fail), "n_in_basket": as_t(int(inside.sum())),
                "is_grasped_any": as_t(any(bool(self.agent.is_grasping(a)[0]) for a in self.fruits))}

    def _get_obs_extra(self, info: dict):
        return dict(tcp_pose=self.agent.tcp_pose.raw_pose)
