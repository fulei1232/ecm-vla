"""Deterministic candidate sampling/ranking and collision-checked joint interpolation.

Only the opt-in nearest expert uses this module. The native mplib 0.1.1 IK
implementation is retained; its unseedable random-configuration provider is
temporarily replaced by a local NumPy generator. Native RRT is a logged fallback.
"""
from __future__ import annotations

import math
import numpy as np

SOLVER_VERSION = "nearest_v2_20261102"
IK_WEIGHTS = np.asarray([64., 24., 4., 2., 1., 1., 1.])
IK_CANDIDATES = 64
COLLISION_STEP_RAD = 0.02


class DeterministicIKModel:
    def __init__(self, model, limits, arm_indices, current, seed):
        self.model, self.limits = model, np.asarray(limits)
        self.indices, self.current = np.asarray(arm_indices), np.asarray(current).copy()
        self.rng = np.random.default_rng(seed)

    def __getattr__(self, key):
        return getattr(self.model, key)

    def get_random_configuration(self):
        q = self.current.copy()
        bounds = self.limits[self.indices]
        q[self.indices] = self.rng.uniform(bounds[:, 0], bounds[:, 1])
        return q


def nearest_equivalent(value, current, lower, upper):
    tau = 2 * math.pi
    first, last = math.ceil((lower-value)/tau), math.floor((upper-value)/tau)
    if first > last:
        return None
    values = [value+k*tau for k in range(first, last+1)]
    return min(values, key=lambda q: (abs(q-current), abs(q), q))


def rank_candidates(candidates, current, indices, limits):
    indices, limits = np.asarray(indices), np.asarray(limits)
    wrist = indices[6]
    unique = {}
    for candidate in candidates:
        q = np.asarray(candidate, dtype=np.float64).copy()
        if q.shape != current.shape or not np.isfinite(q).all():
            continue
        mapped = nearest_equivalent(q[wrist], current[wrist], *limits[wrist])
        if mapped is None or not np.isfinite(mapped) or abs(mapped) > math.pi:
            continue
        q[wrist] = mapped
        if np.any(q[indices] < limits[indices, 0]) or np.any(q[indices] > limits[indices, 1]):
            continue
        nonarm = np.ones(len(q), dtype=bool)
        nonarm[indices] = False
        q[nonarm] = current[nonarm]
        distance = float(np.sum(IK_WEIGHTS * (q[indices]-current[indices])**2))
        key = tuple(q[indices].round(10))
        if key not in unique or tuple(q) < tuple(unique[key][1]):
            unique[key] = (distance, q)
    return [q for _, q in sorted(unique.values(), key=lambda item: (item[0], tuple(item[1][indices])))]


def collision_free(planner, q, attach):
    return not planner.check_for_self_collision(qpos=q) and not planner.check_for_env_collision(
        qpos=q, with_point_cloud=True, use_attach=attach)


def straight_plan(planner, current, goal, indices, dt, attach):
    indices = np.asarray(indices)
    steps = max(2, int(np.ceil(np.max(np.abs(goal[indices]-current[indices]))/COLLISION_STEP_RAD))+1)
    path = np.linspace(current[indices], goal[indices], steps)
    for arm in path:
        q = current.copy()
        q[indices] = arm
        if not collision_free(planner, q, attach):
            return None
    if np.max(np.abs(path[-1]-path[0])) < 1e-10:
        return {"status": "Success", "position": path[:1], "time": np.zeros(1), "duration": 0.0}
    times, positions, velocities, accelerations, duration = planner.TOPP(path, dt)
    # Time parameterization is validated too; interpolation must not hide a collision.
    for arm in positions:
        q = current.copy()
        q[indices] = arm
        if not collision_free(planner, q, attach):
            return None
    return {"status": "Success", "time": times, "position": positions, "velocity": velocities,
            "acceleration": accelerations, "duration": duration}


def nearest_plan(planner, goal_pose, current, dt, attach, seed=0, candidates=IK_CANDIDATES):
    current = np.asarray(current, dtype=np.float64).copy()
    indices = np.asarray(planner.move_group_joint_indices)
    assert len(indices) == 7
    planner.planning_world.set_use_point_cloud(True)
    planner.planning_world.set_use_attach(attach)
    original = planner.pinocchio_model
    planner.pinocchio_model = DeterministicIKModel(original, planner.joint_limits, indices, current, seed)
    try:
        status, raw = planner.IK(goal_pose, current.copy(), n_init_qpos=candidates)
    finally:
        planner.pinocchio_model = original
    ranked = rank_candidates(raw, current, indices, planner.joint_limits)
    trace = {"ik_status": status, "sample_seed": seed, "n_init_qpos": candidates,
             "raw_candidates": len(raw), "eligible_candidates": len(ranked), "attempts": [],
             "weights": IK_WEIGHTS.tolist(), "native_rrt_seed_api": "unavailable in mplib 0.1.1"}
    for rank, goal in enumerate(ranked, start=1):
        row = {"rank": rank, "goal_arm_qpos": goal[indices].tolist()}
        try:
            result = straight_plan(planner, current, goal, indices, dt, attach)
        except RuntimeError as exc:
            row["straight_failure"] = str(exc)
            result = None
        method = "straight"
        if result is None:
            method = "rrt"
            try:
                result = planner.plan_qpos_to_qpos([goal], current.copy(), time_step=dt, use_point_cloud=True,
                                                  use_attach=attach, planning_time=2)
            except RuntimeError as exc:
                result = {"status": f"RRT/time-parameterization failed: {exc}"}
        if result["status"] == "Success" and np.any(np.abs(np.asarray(result["position"])[:, 6]) > math.pi):
            result = {"status": "Rejected: wrist path exceeds [-pi,pi]"}
        row.update(method=method, status=result["status"])
        trace["attempts"].append(row)
        if result["status"] == "Success":
            trace.update(method=method, selected_rank=rank, status="Success")
            return result, trace
    trace.update(method=None, status="no feasible ranked joint goal")
    return None, trace
