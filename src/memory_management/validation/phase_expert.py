"""Phase-aware expert for the closed-loop Teacher Oracle, step 2b (phase expert preflight).

Protocol: reports/phase_expert_preflight_protocol_20261024.md. RoboMME's solve() functions are
open-loop primitive sequences that always start at the subgoal's first primitive, so a solve
restarted mid-subgoal reopens the gripper and backs up to the reach pose (step 2, T-rh stuck).
PhaseSolver keeps the waypoints of the original solve() functions but chooses the next primitive
from the current state after every primitive:

  pickup   grasped -> lift; gripper closed -> finish an in-progress close (first primitive only,
           TCP at the grasp pose) or open; on the vertical approach line -> descend and close;
           otherwise -> reach pose
  release  gripper closed -> move over the target, open, retreat; open -> retreat (BinFill bin,
           VideoUnmaskSwap put-down)
  button   on the vertical line above the button -> (close,) press, back up; otherwise -> ready pose

BinFill pickups target the planned cube unless it is already in the bin, then the first cube of
the same colour still on the table; a held cube of that colour is kept. Subgoals not listed in
KINDS run the original solve().

Only numpy at module level; PhaseSolver imports sapien, torch and mani_skill lazily.
"""
from __future__ import annotations

import numpy as np

from . import sim_branch as sb

CORRIDOR = 0.02
MAX_PRIMITIVES = 20
FINGER_LENGTH = 0.025
CUBE_Z = 0.15
BIN_Z = 0.2
RELEASE_Z = 0.07
RISE_Z = 0.15
DROP_Z = 0.2
RETREAT_X = 0.1
BUTTON_Z = 0.15
REMOVED_X = 5.0
KINDS = {("BinFill", "pick up the cube"): "pickup_cube",
         ("BinFill", "put it into the bin"): "drop_bin",
         ("BinFill", "press the button"): "button",
         ("VideoUnmaskSwap", "pick up the container"): "pickup_bin",
         ("VideoUnmaskSwap", "put down the container"): "putdown"}


# --------------------------------------------------------------------------- phase rules (no simulator)

def horizontal(a, b):
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def in_corridor(tcp, bottom, top_z, c=CORRIDOR):
    """On the vertical line above `bottom`, between its height and top_z."""
    return horizontal(tcp, bottom) <= c and float(bottom[2]) - c <= float(tcp[2]) <= top_z + c


def pickup_phase(grasped, closed, tcp, grasp_p, reach_z, first, c=CORRIDOR):
    if grasped:
        return "lift" if closed else "close"
    if closed:
        at_grasp = float(np.linalg.norm(np.subtract(tcp, grasp_p, dtype=np.float64))) <= c
        return "finish_close" if first and at_grasp else "open"
    return "descend" if in_corridor(tcp, grasp_p, reach_z, c) else "reach"


def release_phase(closed):
    return "release" if closed else "retreat"


def button_phase(closed, tcp, button_p, ready_z, c=CORRIDOR):
    if in_corridor(tcp, button_p, ready_z, c):
        return "press" if closed else "close_press"
    return "approach"


def pick_cube(planned, same_colour, removed, grasped):
    live = [c for c in same_colour if not removed(c)]
    held = [c for c in live if grasped(c)]
    if held:
        return held[0]
    if not removed(planned):
        return planned
    return live[0] if live else planned


# --------------------------------------------------------------------------- simulator side

def _vec(x):
    return sb._as_array(x).reshape(-1).astype(np.float64)


class PhaseSolver:
    """solve(k) for ExpertRun.run: one call runs primitives until its subgoal's last move."""

    def __init__(self, base, wrapper, planner, run):
        self.base, self.wrapper, self.planner = base, wrapper, planner
        self.task = base.spec.id
        self.stats = run.stats.setdefault("phase", {"primitives": {}, "fallbacks": 0, "retargets": 0,
                                                    "primitive_limit": 0})

    def solve(self, k):
        task = self.base.task_list[k]
        kind = KINDS.get((self.task, task.get("choice_label")))
        if kind is None:
            self.stats["fallbacks"] += 1
            return task["solve"](self.wrapper, self.planner)
        step = {"pickup_cube": self._pickup, "pickup_bin": self._pickup, "drop_bin": self._drop_bin,
                "putdown": self._putdown, "button": self._button}[kind]
        for i in range(MAX_PRIMITIVES):
            if step(kind, task, i == 0):
                return None
        self.stats["primitive_limit"] += 1
        return None

    # state
    def _tcp(self):
        return _vec(self.base.agent.tcp.pose.p)[:3]

    def _tcp_q(self):
        return self.base.agent.tcp.pose.q.tolist()[0]

    def _closed(self):
        return self.planner.gripper_state == self.planner.CLOSED

    def _grasped(self, obj):
        return bool(_vec(self.base.agent.is_grasping(obj))[0])

    def _removed(self, obj):
        return float(_vec(obj.pose.p)[0]) > REMOVED_X

    def _count(self, kind, phase):
        key = f"{kind}:{phase}"
        self.stats["primitives"][key] = self.stats["primitives"].get(key, 0) + 1

    def _move(self, pose):
        result = self.planner.move_to_pose_with_screw(pose)
        return not (isinstance(result, int) and result == -1)

    # subgoals; each returns True when the solve is finished
    def _target(self, kind, task):
        segment = task.get("segment")
        planned = segment[0] if isinstance(segment, (list, tuple)) else segment
        if kind != "pickup_cube":
            return planned
        base = self.base
        same = next((c for c in (base.red_cubes, base.blue_cubes, base.green_cubes)
                     if any(x is planned for x in c)), [planned])
        obj = pick_cube(planned, same, self._removed, self._grasped)
        if obj is not planned:
            self.stats["retargets"] += 1
        return obj

    def _pickup(self, kind, task, first):
        import sapien
        from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
        base, planner = self.base, self.planner
        obj = self._target(kind, task)
        approaching = np.array([0, 0, -1])
        target_closing = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        info = compute_grasp_info_by_obb(get_actor_obb(obj), approaching=approaching, target_closing=target_closing,
                                         depth=FINGER_LENGTH)
        grasp_pose = base.agent.build_grasp_pose(approaching, info["closing"], obj.pose.sp.p)
        if kind == "pickup_bin":
            reach = grasp_pose * sapien.Pose([0, 0, -0.15])
            reach_p = reach.p.tolist()
            reach_p[2] = BIN_Z
            reach, grasp, top = sapien.Pose(reach_p, reach.q.tolist()), grasp_pose * sapien.Pose([0, 0, -0.01]), BIN_Z
        else:
            reach_p = grasp_pose.p.tolist()
            reach_p[2] = CUBE_Z
            reach, grasp, top = sapien.Pose(p=reach_p, q=grasp_pose.q.tolist()), grasp_pose, CUBE_Z
        phase = pickup_phase(self._grasped(obj), self._closed(), self._tcp(), _vec(grasp.p), top, first)
        self._count(kind, phase)
        if phase == "reach":
            return not self._move(reach)
        if phase == "descend":
            if not self._move(grasp):
                return True
            planner.close_gripper()
            return False
        if phase in ("finish_close", "close"):
            planner.close_gripper()
            return False
        if phase == "open":
            planner.open_gripper()
            return False
        goal = obj.pose.p.tolist()[0]
        goal[2] = top
        self._move(sapien.Pose(goal, grasp_pose.q))
        return True

    def _drop_bin(self, kind, task, first):
        import sapien
        q, target = self._tcp_q(), self.base.board_with_hole
        phase = release_phase(self._closed())
        self._count(kind, phase)
        if phase == "release":
            goal = target.pose.p.tolist()[0]
            goal[2] = DROP_Z
            if not self._move(sapien.Pose(goal, q)):
                return True
            self.planner.open_gripper()
        goal = target.pose.p.tolist()[0]
        goal[0] -= RETREAT_X
        goal[2] = DROP_Z
        self._move(sapien.Pose(goal, q))
        return True

    def _putdown(self, kind, task, first):
        import sapien
        q = self._tcp_q()
        phase = release_phase(self._closed())
        self._count(kind, phase)
        if phase == "release":
            goal = self.base.agent.tcp.pose.p.tolist()[0]
            goal[2] = RELEASE_Z
            if not self._move(sapien.Pose(goal, q)):
                return True
            self.planner.open_gripper()
        goal = self.base.agent.tcp.pose.p.tolist()[0]
        goal[2] = RISE_Z
        self._move(sapien.Pose(goal, q))
        return True

    def _button(self, kind, task, first):
        import sapien
        import torch
        from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix, matrix_to_quaternion
        position = self.base.button.pose.p.tolist()[0]
        ready = position.copy()
        ready[2] = BUTTON_Z
        angles = torch.deg2rad(torch.tensor([180.0, 0.0, 0.0], dtype=torch.float32))
        rotate = matrix_to_quaternion(euler_angles_to_matrix(angles, convention="XYZ"))
        phase = button_phase(self._closed(), self._tcp(), position, BUTTON_Z)
        self._count(kind, phase)
        if phase == "approach":
            return not self._move(sapien.Pose(p=ready, q=rotate))
        if phase == "close_press":
            self.planner.close_gripper()
        if not self._move(sapien.Pose(p=position, q=rotate)):
            return True
        self._move(sapien.Pose(p=ready, q=rotate))
        return True
