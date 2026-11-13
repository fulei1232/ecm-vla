"""Canonical-only execution and saved planning inputs; legacy/v2 stay separate."""
from pathlib import Path
import json
import math

import numpy as np

from .canonical_planning import CANONICAL_SETTLE_STEPS, COLLISION_MARGIN_M, SOLVER_VERSION, canonical_plan, result_digest


def initialize(solver):
    solver.solver_version = SOLVER_VERSION
    solver.canonical_reference = np.asarray(solver.u.agent.keyframes["rest"].qpos,dtype=np.float64).copy()
    solver.canonical_settle_steps = CANONICAL_SETTLE_STEPS
    solver.canonical_snapshot_dir = None
    solver.canonical_step_count = 0
    solver.canonical_pending_basket = None
    solver.canonical_basket_checks = []
    solver.canonical_attached_radius = 0.
    solver.canonical_q7 = {"samples":0,"state_outside_pi":0,"command_outside_pi":0,
                           "state_range":[None,None],"command_range":[None,None],"first_outside":None}
    original_step = solver.step_fn

    def step(action, labels):
        measured, commanded = float(solver.qpos()[6]), float(action[6])
        stats = solver.canonical_q7
        stats["samples"] += 1
        for key,value in (("state",measured),("command",commanded)):
            low,high = stats[key+"_range"]
            stats[key+"_range"] = [value if low is None else min(low,value),value if high is None else max(high,value)]
            stats[key+"_outside_pi"] += int(abs(value)>math.pi)
        if stats["first_outside"] is None and (abs(measured)>math.pi or abs(commanded)>math.pi):
            stats["first_outside"] = {"timestep":solver.canonical_step_count,"phase":labels["phase"],
                                      "state":measured,"command":commanded}
        result = original_step(action,labels)
        solver.canonical_step_count += 1
        return result
    solver.step_fn = step


def configure(solver, options=None):
    options = options or {}
    solver.canonical_settle_steps = int(options.get("settle_steps",CANONICAL_SETTLE_STEPS))
    if solver.canonical_settle_steps not in (2,12):
        raise ValueError("canonical settle_steps must be 2, or mandated fallback 12")
    if options.get("snapshot_dir"):
        solver.canonical_snapshot_dir = Path(options["snapshot_dir"])
        solver.canonical_snapshot_dir.mkdir(parents=True,exist_ok=True)


def refresh_obstacles(solver, skip=()):
    from .solver import POINT_RADIUS
    solver.canonical_point_cloud = solver.obstacles(skip)
    solver.planner.update_point_cloud(solver.canonical_point_cloud,radius=POINT_RADIUS)


def plan(solver, pose, attach):
    from .solver import POINT_RADIUS
    solver.planner.update_point_cloud(solver.canonical_point_cloud,radius=POINT_RADIUS+COLLISION_MARGIN_M)
    current = solver.qpos().astype(np.float64)
    goal = np.concatenate([pose.p,pose.q])
    phase = solver.canonical_phase
    snapshot = None
    if solver.canonical_snapshot_dir is not None:
        snapshot = solver.canonical_snapshot_dir / f"plan_{len(solver.trajectory_audit):04d}_{phase}.npz"
        target = solver.canonical_labels.get("target")
        fruit = solver.u.fruit_positions()[target] if target is not None else np.full(3,np.nan)
        np.savez_compressed(snapshot,current=current,reference=solver.canonical_reference,goal_pose=goal,
            point_cloud=solver.canonical_point_cloud,fruit_center=fruit,
            metadata=json.dumps({"phase":phase,"attach":attach,"attached_radius":solver.canonical_attached_radius,
                "point_radius":POINT_RADIUS+COLLISION_MARGIN_M,"dt":solver.dt,"ik_seed":solver.ik_seed,
                "ik_candidates":solver.ik_candidates,"urdf":solver.u.agent.urdf_path,
                "user_link_names":solver.planner.user_link_names,"user_joint_names":solver.planner.user_joint_names,
                "velocity_limits":np.asarray(solver.planner.joint_vel_limits).tolist(),
                "acceleration_limits":np.asarray(solver.planner.joint_acc_limits).tolist(),
                "timestep":solver.canonical_step_count,"labels":solver.canonical_labels}))
    try:
        result, trace = canonical_plan(solver.planner,goal,current,solver.canonical_reference,
                solver.dt,attach,phase,solver.ik_seed,solver.ik_candidates)
    finally:
        # approach/pull retain their original, non-inflated screw planner.
        solver.planner.update_point_cloud(solver.canonical_point_cloud,radius=POINT_RADIUS)
    trace.update(snapshot=str(snapshot) if snapshot else None,timestep=solver.canonical_step_count,
                 result_sha256=result_digest(result))
    solver.trajectory_audit.append(trace)
    return result


def check_pending_basket(solver, when):
    from .solver import PlanningFailure
    if solver.canonical_pending_basket is None:
        return
    i = solver.canonical_pending_basket
    inside = bool(solver.u.in_basket()[i])
    solver.canonical_basket_checks.append({"fruit":i,"when":when,"timestep":solver.canonical_step_count,"inside":inside})
    solver.canonical_pending_basket = None
    if not inside:
        raise PlanningFailure(f"fruit {i} missed the basket (deferred {when})")


def move(solver, pose, labels, screw=False, attach=False):
    from .solver import PlanningFailure
    solver.canonical_phase, solver.canonical_labels = labels["phase"], dict(labels)
    result = solver.plan(pose,screw,attach)
    if screw:
        solver.trajectory_audit.append({"method":"screw","phase":labels["phase"],
            "status":"Success" if result is not None else "Failed","timestep":solver.canonical_step_count})
    if result is None and screw:
        result = solver.plan(pose,False,attach)
    if result is None:
        raise PlanningFailure(f"{labels['subgoal']}: no plan to {np.round(pose.p,3).tolist()}")
    # Validate, never wrap/clip the original Cartesian screw commands.
    if np.any(np.abs(np.asarray(result["position"])[:,6])>math.pi):
        raise PlanningFailure(f"{labels['phase']}: screw/path q7 command outside [-pi,pi]")
    if labels["phase"] == "reach":
        check_pending_basket(solver,"after_next_reach_planning")
    solver.follow(result,labels)


def pick(solver, i, k, n):
    from .solver import (CLOSED, CLOSE_STEPS, LIFT, OPEN, OPEN_STEPS, PRE_GRASP, PULL_BACK,
                         PlanningFailure, describe, horizontal_grasp_pose)
    L,u = solver.L,solver.u
    fruit = L.fruits[i]
    pos = u.fruit_positions()[i]
    name = describe(pos,fruit["ripe"] if L.task == "occluded_ripe" else None)
    a = np.array([pos[0],pos[1],0.])/np.linalg.norm(pos[:2])
    grasp = horizontal_grasp_pose(pos,a)
    pre = horizontal_grasp_pose(pos-PRE_GRASP*a,a)
    labels = {"subgoal":f"pick up {name}","target":i,"pick_index":k,"n_picks":n}
    solver.refresh_obstacles(skip=(i,)); solver.gripper = OPEN
    solver.move(pre,{**labels,"phase":"reach"})
    solver.move(grasp,{**labels,"phase":"approach"},screw=True)
    solver.gripper = CLOSED
    solver.hold(CLOSE_STEPS,{**labels,"phase":"grasp"})
    if not u.attached[i]:
        raise PlanningFailure(f"fruit {i} did not detach")
    solver.canonical_attached_radius = fruit["radius"]
    solver.planner.update_attached_sphere(fruit["radius"],[0,0,0,1.,0,0,0])
    back = horizontal_grasp_pose(pos-PULL_BACK*a+[0,0,LIFT],a)
    solver.move(back,{**labels,"phase":"pull"},screw=True,attach=True)
    labels = {**labels,"subgoal":"put the fruit into the basket"}
    solver.refresh_obstacles(skip=(i,))
    solver.move(solver.home,{**labels,"phase":"carry"},attach=True)
    solver.gripper = OPEN
    solver.hold(OPEN_STEPS,{**labels,"phase":"release"})
    solver.hold(solver.canonical_settle_steps,{**labels,"phase":"settle"})
    solver.canonical_pending_basket = i


def run(solver):
    from .solver import OPEN
    solver.gripper = OPEN
    solver.hold(solver.L.observe_steps,{"subgoal":"observe the tree","target":None,"phase":"observe"})
    n = len(solver.L.targets)
    for k,i in enumerate(solver.L.targets):
        solver.pick(i,k,n)
    solver.hold(5,{"subgoal":"All tasks completed","target":None,"phase":"done"})
    check_pending_basket(solver,"before_episode_end")
    return bool(solver.u.evaluate()["success"][0])
