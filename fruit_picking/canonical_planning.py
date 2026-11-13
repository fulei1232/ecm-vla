"""Revision-2 expert: fixed-reference IK and fixed-waypoint joint paths.

This is separate from the sealed nearest-v2 implementation. IK chooses a
kinematic goal before path feasibility is considered; current state, gripper
state and obstacle-dependent candidate filtering cannot change that goal.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np

from .joint_planning import COLLISION_STEP_RAD, DeterministicIKModel, collision_free, nearest_equivalent

SOLVER_VERSION = "canonical_v3_20261102"
COLLISION_MARGIN_M = .01
CANONICAL_SETTLE_STEPS = 2
# A second, fixed high preparation pose, used only when going via rest fails.
HIGH_READY_ARM = np.asarray([-1.7306, -.9000, -.0061, 1.4000, -.1810, .0534, -1.3933])


def canonical_candidates(raw, reference, indices, limits):
    """Equal weights; q7 equivalence is relative to rest, never current state."""
    indices = np.asarray(indices)
    limits = np.asarray(limits)
    unique = {}
    for value in raw:
        q = np.asarray(value, dtype=np.float64).copy()
        if q.shape != reference.shape or not np.isfinite(q).all():
            continue
        wrist = indices[6]
        mapped = nearest_equivalent(q[wrist], reference[wrist], *limits[wrist])
        if mapped is None or abs(mapped) > math.pi:
            continue
        q[wrist] = mapped
        if np.any(q[indices] < limits[indices, 0]) or np.any(q[indices] > limits[indices, 1]):
            continue
        nonarm = np.ones(len(q), dtype=bool)
        nonarm[indices] = False
        q[nonarm] = reference[nonarm]
        distance = float(np.sum((q[indices]-reference[indices])**2))
        key = tuple(q[indices].round(10))
        unique[key] = (distance, q)
    return [q for _, q in sorted(unique.values(), key=lambda row: (row[0], tuple(row[1][indices])))]


def canonical_goal(planner, pose, reference, seed=0, candidates=64):
    """One rest-initialized CLIK; deterministic random initializations only on failure.

    Native Planner.IK mixes kinematics and scene collisions. Calling the same
    native CLIK/FK directly keeps the selected target independent of scene
    state. Path validation below still checks all collisions.
    """
    indices = np.asarray(planner.move_group_joint_indices)
    reference = np.asarray(reference, dtype=np.float64).copy()
    model = planner.pinocchio_model
    link = planner.link_name_2_idx[planner.move_group]
    sampler = DeterministicIKModel(model, planner.joint_limits, indices, reference, seed)

    def solve(start):
        result = model.compute_IK_CLIK(link, pose, start.copy(), [])
        q = np.asarray(result[0], dtype=np.float64).copy()
        if not planner.wrap_joint_limit(q):
            return []
        ranked = canonical_candidates([q], reference, indices, planner.joint_limits)
        if not ranked:
            return []
        q = ranked[0]
        if planner.check_for_self_collision(qpos=q):
            return []
        model.compute_forward_kinematics(q)
        reached = np.asarray(model.get_link_pose(link))
        if planner.distance_6D(pose[:3], pose[3:], reached[:3], reached[3:]) >= 1e-3:
            return []
        return [q]

    raw = solve(reference)
    initial_ok = bool(raw)
    if not initial_ok:
        for _ in range(candidates):
            raw.extend(solve(sampler.get_random_configuration()))
    ranked = canonical_candidates(raw, reference, indices, planner.joint_limits)
    trace = {"ik_initial": "rest", "initial_rest_ik_success": initial_ok,
             "random_initializations": 0 if initial_ok else candidates, "sample_seed": seed,
             "eligible_candidates": len(ranked), "weights": [1.]*7,
             "reference_arm_qpos": reference[indices].tolist()}
    return (ranked[0] if ranked else None), trace


def linear_samples(start, end, indices):
    steps = max(2, int(np.ceil(np.max(np.abs(end[indices]-start[indices]))/COLLISION_STEP_RAD))+1)
    return np.linspace(start[indices], end[indices], steps)


def checked_polyline(planner, nodes, indices, dt, attach):
    pieces = [linear_samples(a, b, indices) for a, b in zip(nodes, nodes[1:])]
    path = np.concatenate([pieces[0]]+[p[1:] for p in pieces[1:]])
    # Zero-length legs (e.g. current==rest) must not create repeated TOPP knots.
    keep = np.r_[True,np.max(np.abs(np.diff(path,axis=0)),axis=1)>1e-12]
    path = path[keep]
    current = nodes[0].copy()
    for arm in path:
        q = current.copy(); q[indices] = arm
        if not collision_free(planner, q, attach):
            return None
    if np.max(np.abs(path-path[0])) < 1e-10:
        return {"status":"Success", "position":path[:1], "time":np.zeros(1), "duration":0.}
    times, positions, velocities, accelerations, duration = planner.TOPP(path, dt)
    # Native TOPP samples [0,duration); explicitly command the fixed endpoint.
    if not np.array_equal(positions[-1], path[-1]):
        times = np.append(times, duration)
        positions = np.vstack([positions, path[-1]])
        velocities = np.vstack([velocities, np.zeros(len(indices))])
        accelerations = np.vstack([accelerations, np.zeros(len(indices))])
    for arm in positions:
        q = current.copy(); q[indices] = arm
        if not collision_free(planner, q, attach):
            return None
    return {"status":"Success", "time":times, "position":positions, "velocity":velocities,
            "acceleration":accelerations, "duration":float(duration)}


def canonical_plan(planner, goal_pose, current, reference, dt, attach, phase,
                   seed=0, candidates=64):
    """Point cloud must already use POINT_RADIUS + COLLISION_MARGIN_M."""
    current = np.asarray(current, dtype=np.float64).copy()
    reference = np.asarray(reference, dtype=np.float64).copy()
    indices = np.asarray(planner.move_group_joint_indices)
    if len(indices) != 7:
        raise ValueError("canonical expert requires seven arm joints")
    planner.planning_world.set_use_point_cloud(True)
    planner.planning_world.set_use_attach(attach)
    planner.robot.set_qpos(current, True)
    planner.planning_world.set_qpos_all(current[indices])
    if phase == "carry":
        goal, trace = reference.copy(), {"ik_initial":"not_used_carry_rest", "weights":[1.]*7}
    else:
        goal, trace = canonical_goal(planner, goal_pose, reference, seed, candidates)
    trace.update(phase=phase, collision_margin_m=COLLISION_MARGIN_M, attempts=[],
                 native_rrt_seed_api="unavailable in mplib 0.1.1")
    if goal is None:
        trace.update(status="no canonical IK goal", method=None, goal_arm_qpos=None)
        return None, trace
    trace["goal_arm_qpos"] = goal[indices].tolist()
    # Nonarm state stays fixed during path checking; it never affects goal IK.
    nonarm = np.ones(len(current),dtype=bool); nonarm[indices] = False
    goal[nonarm] = current[nonarm]
    rest = current.copy(); rest[indices] = reference[indices]
    high = current.copy(); high[indices] = HIGH_READY_ARM
    options = [("straight",None,[current,goal]),
               ("detour","rest",[current,rest,goal]),
               ("detour","high_ready",[current,high,goal])]
    for method, waypoint, nodes in options:
        try:
            result = checked_polyline(planner,nodes,indices,dt,attach)
            error = None
        except RuntimeError as exc:
            result, error = None, str(exc)
        trace["attempts"].append({"method":method,"waypoint":waypoint,"success":result is not None,"error":error})
        if result is not None and np.all(np.abs(np.asarray(result["position"])[:,6]) <= math.pi):
            trace.update(status="Success",method=method,waypoint=waypoint)
            return result,trace
    # Never switch IK branches after a path failure. One fixed joint goal only.
    try:
        result = planner.plan_qpos_to_qpos([goal],current.copy(),time_step=dt,
                    use_point_cloud=True,use_attach=attach,planning_time=2)
        if result["status"] == "Success":
            positions = np.asarray(result["position"])
            if not np.array_equal(positions[-1],goal[indices]):
                result["position"] = np.vstack([positions,goal[indices]])
                if "time" in result:
                    result["time"] = np.append(result["time"],result["time"][-1]+dt)
                for key in ("velocity","acceleration"):
                    if key in result: result[key] = np.vstack([result[key],np.zeros(len(indices))])
                if "duration" in result: result["duration"] += dt
            for a,b in zip(result["position"],result["position"][1:]):
                start,end = current.copy(),current.copy();start[indices]=a;end[indices]=b
                for arm in linear_samples(start,end,indices):
                    q = current.copy(); q[indices] = arm
                    if abs(arm[6]) > math.pi or not collision_free(planner,q,attach):
                        result = {"status":"RRT path failed canonical collision/q7 validation"}
                        break
                if result["status"] != "Success": break
    except RuntimeError as exc:
        result = {"status":f"RRT failed: {exc}"}
    trace["attempts"].append({"method":"rrt","status":result["status"]})
    trace.update(method="rrt",status=result["status"])
    return (result if result["status"] == "Success" else None),trace


def result_digest(result):
    """Exact IEEE-array digest, excluding wall time and mutable planner internals."""
    if result is None:
        return None
    digest = hashlib.sha256()
    for key in ("status","position","time","velocity","acceleration","duration"):
        digest.update(key.encode())
        value = result.get(key)
        if isinstance(value,str) or value is None:
            digest.update(str(value).encode())
        else:
            array = np.ascontiguousarray(value,dtype=np.float64)
            digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()
