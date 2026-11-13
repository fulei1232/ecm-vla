"""RealMan arms with an EG2 two-finger gripper as ManiSkill agents: `rm75b_eg2` (7 joints, the default; same
8-d joint+gripper interface as the Panda MME-VLA was trained on) and `rm65b_eg2` (6 joints).

Build the assets first: `python ecm_vla/fruit_picking/assets/build_rm_eg2.py --arm RM75-B` (and `--arm RM65-B`).
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (PDEEPoseControllerConfig, PDJointPosControllerConfig,
                                           PDJointPosMimicControllerConfig, deepcopy_dict)
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils

ASSETS = Path(__file__).resolve().parent / "assets"
OPEN_FINGERS = [0.04, 0.04]


class RealManEG2(BaseAgent):
    """Rest pose (`keyframes["rest"]`): tcp above the basket at (-0.06, -0.34, 0.47), pointing away from the tree
    so the wrist camera cannot see it."""
    urdf_config = dict(
        _materials=dict(gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)),
        link=dict(left_finger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
                  right_finger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1)),
    )
    gripper_joint_names = ["left_finger_joint", "right_finger_joint"]
    ee_link_name = "tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names, lower=None, upper=None, stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, normalize_action=False)
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names, lower=-0.1, upper=0.1, stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, use_delta=True)
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names, pos_lower=-0.1, pos_upper=0.1, rot_lower=-0.1, rot_upper=0.1,
            stiffness=self.arm_stiffness, damping=self.arm_damping, force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name, urdf_path=self.urdf_path)
        # -1 -> -0.01 (squeeze past closed so thin objects still get force), +1 -> fully open
        gripper = PDJointPosMimicControllerConfig(
            self.gripper_joint_names, lower=-0.01, upper=0.04, stiffness=self.gripper_stiffness,
            damping=self.gripper_damping, force_limit=self.gripper_force_limit,
            mimic={"right_finger_joint": {"joint": "left_finger_joint"}})
        return deepcopy_dict(dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper),
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=deepcopy(gripper)),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=deepcopy(gripper)),
        ))

    @property
    def _sensor_configs(self):
        return [CameraConfig(
            uid="hand_camera",
            pose=sapien_utils.look_at(eye=[0, -0.06, 0.075], target=[0, -0.015, 0.30], up=[0, -1, 0]),
            width=256, height=256, fov=np.pi / 2, near=0.01, far=10,
            mount=self.robot.links_map["eg2_base"])]

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_finger")
        self.finger2_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_finger")
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)

    def is_grasping(self, object, min_force=0.5, max_angle=85):
        l_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        # jaws open along +x (left) / -x (right) of their link frames
        l_dir = self.finger1_link.pose.to_transformation_matrix()[..., :3, 0]
        r_dir = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 0]
        l_ok = (torch.linalg.norm(l_forces, axis=1) >= min_force) & (
            torch.rad2deg(common.compute_angle_between(l_dir, l_forces)) <= max_angle)
        r_ok = (torch.linalg.norm(r_forces, axis=1) >= min_force) & (
            torch.rad2deg(common.compute_angle_between(r_dir, r_forces)) <= max_angle)
        return l_ok & r_ok

    def is_static(self, threshold: float = 0.2):
        return torch.max(torch.abs(self.robot.get_qvel()[..., :-2]), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """tcp pose with z = approaching and y = closing."""
        approaching, closing = np.asarray(approaching, float), np.asarray(closing, float)
        assert abs(1 - np.linalg.norm(approaching)) < 1e-3 and abs(1 - np.linalg.norm(closing)) < 1e-3
        assert abs(approaching @ closing) <= 1e-3
        T = np.eye(4)
        T[:3, :3] = np.stack([np.cross(closing, approaching), closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)


@register_agent()
class RM75BEG2(RealManEG2):
    uid = "rm75b_eg2"
    urdf_path = str(ASSETS / "rm75b_eg2/rm75b_eg2.urdf")
    arm_joint_names = [f"joint_{i}" for i in range(1, 8)]
    keyframes = dict(rest=Keyframe(
        qpos=np.array([-1.7306, -0.5484, -0.0061, 2.0666, -0.181, 0.0534, -1.3933, *OPEN_FINGERS]), pose=sapien.Pose()))


@register_agent()
class RM65BEG2(RealManEG2):
    uid = "rm65b_eg2"
    urdf_path = str(ASSETS / "rm65b_eg2/rm65b_eg2.urdf")
    arm_joint_names = [f"joint_{i}" for i in range(1, 7)]
    keyframes = dict(rest=Keyframe(
        qpos=np.array([1.3961, -0.5484, 2.0666, 0.0, 0.0526, -1.5708, *OPEN_FINGERS]), pose=sapien.Pose()))
