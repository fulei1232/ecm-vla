"""Scripted mplib demonstrator for `FruitPick-v1`.

Per target fruit: RRT to a pre-grasp 0.12 m behind the fruit along its azimuth, straight-line approach, close until
the fruit detaches, straight-line pull back, RRT to the home pose above the basket (fruit attached to the planner as a
sphere), open, wait for it to settle. The planner avoids a point cloud of the trunk, twigs, basket and the fruits
still on the tree. Every control step goes through `step_fn(action, labels)` so a recorder can log it.
"""
from __future__ import annotations

from typing import Callable

import mplib
import numpy as np
import sapien
import trimesh

from .agent import RealManEG2

OPEN, CLOSED = 1.0, -1.0
PRE_GRASP = 0.12
PULL_BACK = 0.12
LIFT = 0.03
CLOSE_STEPS = 10
OPEN_STEPS = 8
SETTLE_STEPS = 12
POINT_RADIUS = 0.008


class PlanningFailure(RuntimeError):
    pass


def horizontal_grasp_pose(center, approach) -> sapien.Pose:
    a = np.array([approach[0], approach[1], 0.0])
    a /= np.linalg.norm(a)
    closing = np.cross([0, 0, 1.0], a)
    return RealManEG2.build_grasp_pose(a, closing / np.linalg.norm(closing), np.asarray(center, float))


def describe(pos, ripe=None) -> str:
    side = "left" if pos[1] > 0.08 else "right" if pos[1] < -0.08 else "middle"
    level = "upper " if pos[2] > 0.57 else "lower " if pos[2] < 0.51 else ""
    color = "" if ripe is None else ("red " if ripe else "green ")
    return f"the {color}fruit on the {level}{side}"


def surface_points(mesh: trimesh.Trimesh, spacing=0.012) -> np.ndarray:
    n = max(16, int(mesh.area / spacing**2))
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    return np.asarray(pts)


def segment_mesh(a, b, radius, capsule=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    mesh = trimesh.creation.capsule(height=length, radius=radius) if capsule else \
        trimesh.creation.cylinder(radius=radius, height=length)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], (b - a) / length))
    mesh.apply_translation((a + b) / 2)
    return mesh


class FruitPickSolver:
    def __init__(self, env, step_fn: Callable[[np.ndarray, dict], tuple], joint_vel_scale=2.0, joint_acc_scale=2.0,
                 ik_mode="legacy", ik_seed=0, ik_candidates=64):
        if ik_mode not in ("legacy", "nearest", "canonical", "canonical_v31") or ik_candidates < 1:
            raise ValueError("invalid expert IK mode or candidate count")
        self.ik_mode, self.ik_seed, self.ik_candidates = ik_mode, ik_seed, ik_candidates
        self.trajectory_audit = []
        self.solver_version = "legacy"
        if ik_mode == "nearest":
            import random
            from .joint_planning import SOLVER_VERSION
            self.solver_version = SOLVER_VERSION
            random.seed(ik_seed)
            np.random.seed(ik_seed)
        self.env = env
        self.u = env.unwrapped
        self.L = self.u.layout
        self.step_fn = step_fn
        agent = self.u.agent
        robot = agent.robot
        self.robot = robot
        self.n_arm = len(agent.arm_joint_names)
        self.planner = mplib.Planner(
            urdf=agent.urdf_path, srdf=agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=[l.get_name() for l in robot.get_links()],
            user_joint_names=[j.get_name() for j in robot.get_active_joints()], move_group="tcp")
        self.planner.set_base_pose(np.array([0, 0, 0, 1.0, 0, 0, 0]))
        self.planner.joint_vel_limits = np.asarray(self.planner.joint_vel_limits) * joint_vel_scale
        self.planner.joint_acc_limits = np.asarray(self.planner.joint_acc_limits) * joint_acc_scale
        self.dt = self.u.control_timestep
        self.gripper = OPEN
        self.home = self.fk(agent.keyframes["rest"].qpos)
        if ik_mode == "canonical":
            from .canonical_solver import initialize
            initialize(self)
        if ik_mode == "canonical_v31":
            from .canonical_v31_solver import initialize
            initialize(self)

    def qpos(self) -> np.ndarray:
        return self.robot.get_qpos()[0].cpu().numpy()

    def fk(self, qpos) -> sapien.Pose:
        model = self.planner.robot.get_pinocchio_model()
        model.compute_forward_kinematics(np.asarray(qpos, float))
        p = np.asarray(model.get_link_pose(self.planner.user_link_names.index("tcp")))
        return sapien.Pose(p[:3], p[3:])

    def obstacles(self, skip=()) -> np.ndarray:
        L = self.L
        tx, ty = L.trunk["xy"]
        meshes = [segment_mesh([tx, ty, 0.0], [tx, ty, L.trunk["top"]], L.trunk["radius"])]
        meshes += [segment_mesh(t["start"], t["end"], t["radius"], capsule=True) for t in L.twigs]
        positions = self.u.fruit_positions()
        for i, f in enumerate(L.fruits):
            if i not in skip and not self.u.attached[i]:
                meshes.append(trimesh.creation.icosphere(subdivisions=2, radius=f["radius"]).apply_translation(positions[i]))
        c, h, H = np.asarray(L.basket["center"]), L.basket["inner_half"], L.basket["height"]
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ext = [0.006 if sx else 2 * h, 0.006 if sy else 2 * h, H]
            meshes.append(trimesh.creation.box(ext).apply_translation(c + [sx * h, sy * h, H / 2]))
        return np.vstack([surface_points(m) for m in meshes])

    def refresh_obstacles(self, skip=()):
        if self.ik_mode == "canonical_v31":
            from .canonical_v31_solver import refresh_obstacles
            return refresh_obstacles(self, skip)
        if self.ik_mode == "canonical":
            from .canonical_solver import refresh_obstacles
            return refresh_obstacles(self, skip)
        self.planner.update_point_cloud(self.obstacles(skip), radius=POINT_RADIUS)

    def action(self, arm_qpos) -> np.ndarray:
        return np.concatenate([np.asarray(arm_qpos, float)[:self.n_arm], [self.gripper]])

    def follow(self, result, labels):
        for q in result["position"]:
            self.step_fn(self.action(q), labels)

    def hold(self, n, labels):
        q = self.qpos()[:self.n_arm]
        for _ in range(n):
            self.step_fn(self.action(q), labels)

    def plan(self, pose: sapien.Pose, screw: bool, attach: bool):
        if self.ik_mode == "canonical_v31" and not screw:
            from .canonical_v31_solver import plan
            return plan(self, pose, attach)
        if self.ik_mode == "canonical" and not screw:
            from .canonical_solver import plan
            return plan(self, pose, attach)
        goal = np.concatenate([pose.p, pose.q])
        if self.ik_mode == "nearest" and not screw:
            from .joint_planning import nearest_plan
            result, audit = nearest_plan(self.planner, goal, self.qpos(), self.dt, attach,
                                         self.ik_seed, self.ik_candidates)
            self.trajectory_audit.append(audit)
            return result
        for attempt in range(3):
            if screw:
                result = self.planner.plan_screw(goal, self.qpos(), time_step=self.dt, use_point_cloud=True,
                                                 use_attach=attach)
            else:
                result = self.planner.plan_qpos_to_pose(goal, self.qpos(), time_step=self.dt, use_point_cloud=True,
                                                        use_attach=attach, planning_time=2, wrt_world=True)
            if result["status"] == "Success":
                if self.ik_mode == "nearest":
                    self.trajectory_audit.append({"method": "screw", "status": result["status"], "attempt": attempt})
                return result
        if self.ik_mode == "nearest":
            self.trajectory_audit.append({"method": "screw", "status": result["status"], "attempt": attempt})
        return None

    def move(self, pose, labels, screw=False, attach=False):
        if self.ik_mode == "canonical_v31":
            from .canonical_v31_solver import move
            return move(self, pose, labels, screw, attach)
        if self.ik_mode == "canonical":
            from .canonical_solver import move
            return move(self, pose, labels, screw, attach)
        audit_start = len(self.trajectory_audit)
        result = self.plan(pose, screw, attach)
        if result is None and screw:
            result = self.plan(pose, False, attach)
        if self.ik_mode == "nearest":
            for audit in self.trajectory_audit[audit_start:]:
                audit.update(phase=labels["phase"], subgoal=labels["subgoal"], requested_screw=screw)
        if result is None:
            raise PlanningFailure(f"{labels['subgoal']}: no plan to {np.round(pose.p, 3).tolist()}")
        self.follow(result, labels)

    def pick(self, i, k, n):
        if self.ik_mode == "canonical_v31":
            from .canonical_v31_solver import pick
            return pick(self, i, k, n)
        if self.ik_mode == "canonical":
            from .canonical_solver import pick
            return pick(self, i, k, n)
        L, u = self.L, self.u
        fruit = L.fruits[i]
        pos = u.fruit_positions()[i]
        name = describe(pos, fruit["ripe"] if L.task == "occluded_ripe" else None)
        a = np.array([pos[0], pos[1], 0.0]) / np.linalg.norm(pos[:2])
        grasp = horizontal_grasp_pose(pos, a)
        pre = horizontal_grasp_pose(pos - PRE_GRASP * a, a)
        pick_labels = {"subgoal": f"pick up {name}", "target": i, "pick_index": k, "n_picks": n}
        self.refresh_obstacles(skip=(i,))
        self.gripper = OPEN
        self.move(pre, {**pick_labels, "phase": "reach"})
        self.move(grasp, {**pick_labels, "phase": "approach"}, screw=True)
        self.gripper = CLOSED
        self.hold(CLOSE_STEPS, {**pick_labels, "phase": "grasp"})
        if not u.attached[i]:
            raise PlanningFailure(f"fruit {i} did not detach")
        self.planner.update_attached_sphere(fruit["radius"], [0, 0, 0, 1.0, 0, 0, 0])
        back = horizontal_grasp_pose(pos - PULL_BACK * a + [0, 0, LIFT], a)
        self.move(back, {**pick_labels, "phase": "pull"}, screw=True, attach=True)
        place_labels = {"subgoal": "put the fruit into the basket", "target": i, "pick_index": k, "n_picks": n}
        self.refresh_obstacles(skip=(i,))
        self.move(self.home, {**place_labels, "phase": "carry"}, attach=True)
        self.gripper = OPEN
        self.hold(OPEN_STEPS, {**place_labels, "phase": "release"})
        self.hold(SETTLE_STEPS, {**place_labels, "phase": "settle"})
        if not u.in_basket()[i]:
            raise PlanningFailure(f"fruit {i} missed the basket")

    def run(self):
        if self.ik_mode == "canonical_v31":
            from .canonical_v31_solver import run
            return run(self)
        if self.ik_mode == "canonical":
            from .canonical_solver import run
            return run(self)
        self.gripper = OPEN
        self.hold(self.L.observe_steps, {"subgoal": "observe the tree", "target": None, "phase": "observe"})
        n = len(self.L.targets)
        for k, i in enumerate(self.L.targets):
            self.pick(i, k, n)
        self.hold(5, {"subgoal": "All tasks completed", "target": None, "phase": "done"})
        return bool(self.u.evaluate()["success"][0])
