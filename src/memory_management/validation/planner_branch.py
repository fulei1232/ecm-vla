"""The RoboMME planner inside R branches, closed-loop Teacher Oracle step 2 (planner preflight).

Protocol: reports/planner_branch_preflight_protocol_20261023.md. At every decision point of a
recorded D0a rollout a fresh env instance is replayed to the point (method R, step 1) and the
planner is run from there, built exactly as DemonstrationWrapper builds it for demonstrations:
  full  one planner drives the current subgoal's solve() until the episode stops;
  rh    the same, but every 16 steps the solve is abandoned and restarted from the current
        state, which is how the Teacher Oracle queries it once per decision.
The expert chunk the Oracle scores against is the first 20 steps of `full`.
params["expert"] selects the solve: "original" (default, step 2) runs the task's own solve();
"phase" (step 2b, phase_expert.PhaseSolver) re-decides the next primitive from the current state.

Only numpy is imported at module level; sim-dependent code imports lazily and expects the V-A
sys.path to be set.
"""
from __future__ import annotations

import collections
from pathlib import Path
import statistics
import time
import traceback

import numpy as np

from . import sim_branch as sb

CHUNK_LEN = 20
REPLAN_EVERY = sb.EXEC_HORIZON
MAX_STEPS = 1300
STALL_STEPS = 400
STALL_SOLVES = 3
START_STEPS = 4
MODES = ("full", "rh")
TERMINAL = ("success", "fail", "timeout")
STICK_TASKS = ("PatternLock", "RouteStick")
REMOVED_X = 5.0
DEGENERATE = "Fail to parameterize path"
METRICS = ("l2_expert_policy", "l2_hold_policy", "l2_hold_expert", "direction_cosine", "start_speed_expert",
           "start_speed_policy", "start_speed_ratio", "hold_steps", "gripper_agreement")


class _Signal(BaseException):
    """Leaves solve() early; a BaseException so the planner's `except Exception` retries let it through."""


class Stop(_Signal):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class Replan(_Signal):
    pass


def infer_gripper_state(target, lower, upper, open_state, closed_state):
    return open_state if float(target) > (float(lower) + float(upper)) / 2 else closed_state


# --------------------------------------------------------------------------- expert run (no simulator)

class ExpertRun:
    """Stop rules and per-step record of one expert run starting t executed steps into the episode."""

    def __init__(self, t, replan_every=None, max_steps=MAX_STEPS, stall_steps=STALL_STEPS, stall_solves=STALL_SOLVES):
        self.t, self.replan_every = t, replan_every
        self.max_steps, self.stall_steps, self.stall_solves = max_steps, stall_steps, stall_solves
        self.actions, self.sources, self.subgoals, self.statuses = [], [], [], []
        self.changes, self.gripper_inits = [], []
        self.stats = {"moves": 0, "screw_failures": 0, "rrt_attempts": 0, "rrt_failures": 0, "exhausted": [],
                      "degenerate_moves": 0}
        self.source, self.pending, self.outcome, self.error = "other", None, None, None
        self.start_subgoal = self.subgoal = None
        self.window = self.last_change = self.solve_calls = self.replans = 0
        self.seconds = self.plan_seconds = 0.0
        self.plan_timed = []

    @property
    def n(self):
        return len(self.actions)

    def _raise(self, signal):
        self.pending = signal
        raise signal

    def before_step(self):
        if self.pending is not None:
            raise self.pending

    def after_step(self, action, terminated, truncated, status, subgoal):
        self.actions.append(np.asarray(action, dtype=np.float64).reshape(-1).copy())
        self.sources.append(self.source)
        self.subgoals.append(subgoal)
        self.statuses.append(status)
        self.window += 1
        if subgoal != self.subgoal:
            self.subgoal, self.last_change = subgoal, self.n
            self.changes.append([self.n, subgoal])
        # Same order as eval.py: the step budget is checked before the env's own stop flag.
        if self.t + self.n > self.max_steps:
            self._raise(Stop("timeout"))
        if terminated or truncated:
            self._raise(Stop(status if status in TERMINAL else f"status:{status}"))
        if self.n - self.last_change >= self.stall_steps:
            self._raise(Stop("stuck"))
        if self.replan_every and self.window >= self.replan_every:
            self._raise(Replan())

    def run(self, current_subgoal, solve, prepare):
        """solve(k) runs task k's solve(); prepare() (re)initialises the planner before a fresh plan."""
        self.start_subgoal = self.subgoal = current_subgoal()
        stalled, seen, fresh = 0, 0, True
        while True:
            if fresh:
                prepare()
                self.window, fresh = 0, False
            k = current_subgoal()
            self.solve_calls += 1
            self.source = "other"
            try:
                solve(k)
                self.before_step()  # a signal swallowed by a bare `except:` inside solve()
            except Replan:
                self.pending, fresh = None, True
                self.replans += 1
                continue
            except Stop as stop:
                self.outcome = stop.reason
                return self
            except Exception:
                self.outcome, self.error = "exception", traceback.format_exc()[-3000:]
                return self
            stalled = 0 if len(self.changes) > seen else stalled + 1
            seen = len(self.changes)
            if stalled >= self.stall_solves:
                self.outcome = "stuck"
                return self

    def record(self, chunk_len=CHUNK_LEN):
        actions = np.asarray(self.actions, dtype=np.float64) if self.actions else np.zeros((0, 0))
        mismatches = sum(1 for g in self.gripper_inits if g["agrees_with_last_action"] is False)
        return {"outcome": self.outcome, "error": self.error, "steps": self.n,
                "actions_sha256": sb.array_digest(actions), "chunk": actions[:chunk_len].tolist(),
                "chunk_sources": self.sources[:chunk_len], "start_subgoal": self.start_subgoal,
                "final_subgoal": self.subgoal, "subgoal_changes": self.changes, "solve_calls": self.solve_calls,
                "replans": self.replans, "stats": self.stats, "rrt_used": self.stats["rrt_attempts"] > 0,
                "rrt_in_chunk": "rrt" in self.sources[:chunk_len],
                "source_counts": dict(collections.Counter(self.sources)),
                "gripper_inits": len(self.gripper_inits), "gripper_mismatches": mismatches,
                "first_gripper_init": self.gripper_inits[0] if self.gripper_inits else None,
                "seconds": round(self.seconds, 3), "plan_seconds": round(self.plan_seconds, 3),
                "plan_timed": self.plan_timed}


def instrument_planner(planner, wrapper, run, screw_failure, degenerate_ok=False):
    """DemonstrationWrapper.get_demonstration_trajectory's screw -> RRT* retry, with counters for `run`.

    degenerate_ok: a screw plan whose time parameterisation fails (target = current pose) counts as a
    completed move of zero steps instead of raising; the phase expert relies on this.
    """
    screw, rrt, stats = planner.move_to_pose_with_screw, planner.move_to_pose_with_RRTStar, run.stats

    def move(*args, **kwargs):
        stats["moves"] += 1
        try:
            for _ in range(wrapper._demo_screw_max_attempts):
                run.source = "screw"
                try:
                    result = screw(*args, **kwargs)
                except screw_failure:
                    stats["screw_failures"] += 1
                    continue
                except RuntimeError as exc:
                    if not (degenerate_ok and str(exc) == DEGENERATE):
                        raise
                    stats["degenerate_moves"] += 1
                    return None
                if isinstance(result, int) and result == -1:
                    stats["screw_failures"] += 1
                    continue
                return result
            for _ in range(wrapper._demo_rrt_max_attempts):
                run.source = "rrt"
                stats["rrt_attempts"] += 1
                try:
                    result = rrt(*args, **kwargs)
                except Exception:
                    stats["rrt_failures"] += 1
                    continue
                if isinstance(result, int) and result == -1:
                    stats["rrt_failures"] += 1
                    continue
                return result
            stats["exhausted"].append(run.n)
            return -1
        finally:
            run.source = "other"

    def tagged(method):
        def call(*args, **kwargs):
            run.source = "gripper"
            try:
                return method(*args, **kwargs)
            finally:
                run.source = "other"
        return call

    planner.move_to_pose_with_screw = move
    for name in ("open_gripper", "close_gripper"):
        if hasattr(planner, name):
            setattr(planner, name, tagged(getattr(planner, name)))
    mp = getattr(planner, "planner", None)
    for name in ("plan_screw", "plan_qpos_to_pose"):
        if mp is not None and hasattr(mp, name):
            def timed(*args, _plan=getattr(mp, name), **kwargs):
                start = time.perf_counter()
                try:
                    return _plan(*args, **kwargs)
                finally:
                    run.plan_seconds += time.perf_counter() - start
            try:
                setattr(mp, name, timed)
                run.plan_timed.append(name)
            except AttributeError:
                pass
    return planner


# --------------------------------------------------------------------------- chunk measures (no simulator)

def chunk_available(expert, chunk_len=CHUNK_LEN):
    if not expert:
        return False
    if any(e < chunk_len for e in expert["stats"]["exhausted"]):
        return False
    return len(expert["chunk"]) >= chunk_len or expert["outcome"] in TERMINAL


def chunk_metrics(expert, policy, qpos, sources=(), gripper=True, start_steps=START_STEPS):
    """Expert chunk vs the policy's own chunk at the same decision; arm joints only except gripper_agreement."""
    expert, policy, q = (np.asarray(x, dtype=np.float64) for x in (expert, policy, qpos))
    n = min(len(expert), len(policy))
    if n == 0:
        return None
    arm = sb.ARM_JOINTS
    e, p = expert[:n, :arm], policy[:n, :arm]

    def start_speed(x):
        return float(np.linalg.norm(np.diff(np.vstack([q, x[:start_steps]]), axis=0), axis=1).mean())

    de, dp = e[-1] - q, p[-1] - q
    ne, npol = float(np.linalg.norm(de)), float(np.linalg.norm(dp))
    se, sp = start_speed(e), start_speed(p)
    out = {"steps": n,
           "l2_expert_policy": float(np.linalg.norm(e - p, axis=1).mean()),
           "l2_hold_policy": float(np.linalg.norm(p - q, axis=1).mean()),
           "l2_hold_expert": float(np.linalg.norm(e - q, axis=1).mean()),
           "direction_cosine": float(de @ dp / (ne * npol)) if ne > 0 and npol > 0 else None,
           "start_speed_expert": se, "start_speed_policy": sp,
           "start_speed_ratio": se / sp if sp > 0 else None,
           "hold_steps": sum(1 for s in list(sources)[:n] if s == "gripper"),
           "gripper_agreement": None}
    if gripper and expert.shape[1] > arm and policy.shape[1] > arm:
        out["gripper_agreement"] = float(np.mean(np.sign(expert[:n, arm]) == np.sign(policy[:n, arm])))
    return out


def fresh_points(ref):
    """Decision points whose subgoal index changed within the preceding 16 steps (t=0 included)."""
    index = lambda t: ref.frame_at(t)["gt"].get("current_task_index")
    return [t for t in ref.decision_points() if t == 0 or index(t) != index(t - sb.EXEC_HORIZON)]


def policy_chunk(ref, t):
    return np.load(Path(ref.directory) / "actions" / f"decision_{t // sb.EXEC_HORIZON:04d}.npz")["actions"]


# --------------------------------------------------------------------------- simulator side

def demonstration_wrapper(env):
    while not hasattr(type(env), "get_demonstration_trajectory"):
        env = env.env
    return env


def build_planner(wrapper, run, degenerate_ok=False):
    """The planner of DemonstrationWrapper.get_demonstration_trajectory, instrumented for `run`."""
    from robomme.robomme_env.utils.planner_fail_safe import (
        FailAwarePandaArmMotionPlanningSolver, FailAwarePandaStickMotionPlanningSolver, ScrewPlanFailure)
    base = wrapper.unwrapped
    common = dict(debug=False, vis=False, base_pose=base.agent.robot.pose, visualize_target_grasp_pose=False,
                  print_env_info=False)
    if base.spec.id in STICK_TASKS:
        planner = FailAwarePandaStickMotionPlanningSolver(wrapper, joint_vel_limits=0.3, **common)
    else:
        planner = FailAwarePandaArmMotionPlanningSolver(wrapper, **common)
    return instrument_planner(planner, wrapper, run, ScrewPlanFailure, degenerate_ok)


def target_status(base, k):
    segment = base.task_list[k].get("segment")
    actors = segment if isinstance(segment, (list, tuple)) else [] if segment is None else [segment]
    out = []
    for actor in actors:
        try:
            p = sb._as_array(actor.pose.p).reshape(-1)[:3].astype(np.float64)
        except Exception as exc:
            out.append({"name": getattr(actor, "name", None), "error": f"{type(exc).__name__}: {exc}"})
            continue
        out.append({"name": getattr(actor, "name", None), "position": p.tolist(), "removed": bool(p[0] > REMOVED_X)})
    return out


def run_expert(rp, ref, t, mode, params):
    base, wrapper = rp.base, demonstration_wrapper(rp.runner.env)
    run = ExpertRun(t, params["replan_every"] if mode == "rh" else None, params["max_steps"],
                    params["stall_steps"], params["stall_solves"])
    phase = params.get("expert", "original") == "phase"
    planner = build_planner(wrapper, run, degenerate_ok=phase)
    if phase:
        from .phase_expert import PhaseSolver
        solve = PhaseSolver(base, wrapper, planner, run).solve
    else:
        solve = lambda k: base.task_list[k]["solve"](wrapper, planner)  # noqa: E731
    gripper = base.agent.controller.controllers.get("gripper") if hasattr(planner, "gripper_state") else None
    last_executed = ref.actions[t - 1] if t > 0 else None

    def prepare():
        if gripper is None:
            return
        target = float(gripper._target_qpos[0, 0])
        state = infer_gripper_state(target, gripper.config.lower, gripper.config.upper, planner.OPEN, planner.CLOSED)
        planner.gripper_state = state
        last = run.actions[-1] if run.actions else last_executed
        run.gripper_inits.append({"n": run.n, "target": target, "state": state,
                                  "agrees_with_last_action": None if last is None
                                  else bool((state == planner.OPEN) == (float(last[-1]) > 0))})

    original = wrapper.step

    def step(action):
        run.before_step()
        out = original(action)
        if rp.recorder is not None:
            rp.recorder.pending.clear()
        _, _, terminated, truncated, info = out
        run.after_step(action, bool(terminated), bool(truncated), info.get("status"), int(base.current_task_index))
        return out

    wrapper.step = step
    start = time.monotonic()
    try:
        run.run(lambda: int(base.current_task_index), solve, prepare)
    finally:
        wrapper.step = original
        run.seconds = time.monotonic() - start
    return run


def run_point(ref, t, mode, scratch, params):
    out = {"t": t, "mode": mode, "start_ok": False}
    rp = None
    try:
        start = time.monotonic()
        rp = sb.Replayer(ref, scratch).reset()
        reached = rp.advance_to(t)
        frames = sb.compare_frames(ref.frames, rp.rows)
        expected = ref.exec_start + t + 1
        out["start"] = {"reached": reached, "rows": len(rp.rows), "expected_rows": expected, **frames,
                        "replay_seconds": round(time.monotonic() - start, 3)}
        out["start_ok"] = bool(reached and len(rp.rows) == expected and frames["physics_mismatches"] == 0
                               and frames["render_mismatches"] == 0)
        base = rp.base
        k = int(base.current_task_index)
        q = sb._as_array(base.agent.robot.get_qpos())[0, :sb.ARM_JOINTS].astype(np.float64)
        out.update(subgoal=k, subgoal_name=base.task_list[k].get("name"), qpos=q.tolist(),
                   targets=target_status(base, k))
        if out["start_ok"]:
            out["expert"] = run_expert(rp, ref, t, mode, params).record(params["chunk_len"])
            if mode == "full":
                out["metrics"] = chunk_metrics(out["expert"]["chunk"], policy_chunk(ref, t), q,
                                               out["expert"]["chunk_sources"], ref.task not in STICK_TASKS,
                                               params["start_steps"])
    except Exception:
        out["error"] = traceback.format_exc()[-3000:]
    finally:
        if rp is not None:
            try:
                rp.close()
            except Exception:
                out["close_error"] = traceback.format_exc()[-1000:]
    return out


def job_points(ref, spec, scratch, params):
    return {"points": {str(t): {mode: run_point(ref, t, mode, Path(scratch) / f"t{t:04d}_{mode}", params)
                                for mode in spec["modes"]} for t in spec["points"]}}


JOBS = {"points": job_points}


def run_job(ref, spec, scratch, params):
    return JOBS[spec["kind"]](ref, spec, scratch, params)


# --------------------------------------------------------------------------- evaluation

def _expert(rec):
    return (rec or {}).get("expert")


def _outcome(rec):
    return (_expert(rec) or {}).get("outcome")


def _rate(flags):
    flags = list(flags)
    return sum(bool(f) for f in flags) / len(flags) if flags else None


def _median(values):
    values = [v for v in values if v is not None]
    return {"median": statistics.median(values) if values else None, "n": len(values)}


def collect_task(episodes):
    """Decision rows (main jobs) and repeat pairs of one task; episodes = [{key, info, outputs}]."""
    decisions, pairs, branches, missing, expected = [], [], [], [], 0
    for ep in episodes:
        info, outputs = ep["info"], ep["outputs"]
        main, repeat = {}, {}
        for name, spec in info["jobs"].items():
            expected += len(spec["points"]) * len(spec["modes"])
            o = outputs.get(name)
            if not (o and o.get("completed")):
                missing.append(f"{ep['key']}/{name}")
                continue
            for t, modes in o["points"].items():
                (repeat if spec["repeat"] else main)[int(t)] = modes
        fresh = set(info["fresh_points"])
        for t in info["points"]:
            rec = main.get(t, {})
            decisions.append({"episode": ep["key"], "t": t, "success_episode": info["outcome"] == "success",
                              "fresh": t in fresh, "n_steps": info["executed_steps"],
                              "full": rec.get("full"), "rh": rec.get("rh")})
        for t, modes in repeat.items():
            for mode, rec in modes.items():
                pairs.append({"episode": ep["key"], "t": t, "mode": mode,
                              "main": main.get(t, {}).get(mode), "repeat": rec})
        branches += [rec for modes in list(main.values()) + list(repeat.values()) for rec in modes.values()]
    return decisions, pairs, branches, missing, expected


def same_run(a, b):
    keys = ("actions_sha256", "outcome", "steps")
    return all(a.get(k) == b.get(k) for k in keys)


def evaluate_task(episodes, params):
    decisions, pairs, branches, missing, expected = collect_task(episodes)
    success = [d for d in decisions if d["success_episode"]]
    comparable = [p for p in pairs if _expert(p["main"]) and _expert(p["repeat"])
                  and not _expert(p["main"])["rrt_used"] and not _expert(p["repeat"])["rrt_used"]]
    rates = {"P1_chunk_available": _rate(chunk_available(_expert(d["full"]), params["chunk_len"]) for d in decisions),
             "P2_full_success_on_success_episodes": _rate(_outcome(d["full"]) == "success" for d in success),
             "P3_rh_success_on_success_episodes": _rate(_outcome(d["rh"]) == "success" for d in success)}
    gates = {"P0_branch_start": not missing and len(branches) == expected and all(b.get("start_ok") for b in branches),
             "P1_chunk_available": (rates["P1_chunk_available"] or 0) >= params["p1_min"],
             "P2_full_competent": (rates["P2_full_success_on_success_episodes"] or 0) >= params["p2_min"],
             "P3_rh_competent": (rates["P3_rh_success_on_success_episodes"] or 0) >= params["p3_min"],
             "P4_deterministic": bool(comparable) and all(same_run(_expert(p["main"]), _expert(p["repeat"]))
                                                          for p in comparable)}
    return {"gates": gates, "rates": rates, "usable": all(gates.values()),
            "counts": {"decisions": len(decisions), "success_episode_decisions": len(success),
                       "branches": len(branches), "expected_branches": expected, "missing_jobs": missing,
                       "repeat_pairs": len(pairs), "comparable_pairs": len(comparable)},
            "diagnostics": diagnostics(decisions, pairs, branches, comparable)}


def diagnostics(decisions, pairs, branches, comparable):
    experts = [_expert(b) for b in branches if _expert(b)]
    outcomes = {f"{'success' if s else 'fail'}_episode/{mode}":
                dict(collections.Counter(_outcome(d[mode]) for d in decisions if d["success_episode"] == s))
                for s in (True, False) for mode in MODES}
    groups = {}
    for d in decisions:
        m = (d["full"] or {}).get("metrics")
        if m:
            groups.setdefault(f"{'success' if d['success_episode'] else 'fail'}/{'fresh' if d['fresh'] else 'mid'}",
                              []).append(m)
    metrics = {g: {k: _median(m[k] for m in ms) for k in METRICS} for g, ms in sorted(groups.items())}
    recover = {}
    for d in decisions:
        if not d["success_episode"]:
            recover.setdefault(d["episode"], {"n_steps": d["n_steps"], "points": []})["points"].append(
                [d["t"], _outcome(d["full"]), _outcome(d["rh"])])
    for e in recover.values():
        for i, mode in enumerate(MODES, start=1):
            ok = [p[0] for p in e["points"] if p[i] == "success"]
            e[f"{mode}_recoverable"] = _rate(p[i] == "success" for p in e["points"])
            e[f"{mode}_last_recoverable_t"] = max(ok) if ok else None
    moves = sum(x["stats"]["moves"] for x in experts)
    mismatched = [{"episode": p["episode"], "t": p["t"], "mode": p["mode"]} for p in comparable
                  if not same_run(_expert(p["main"]), _expert(p["repeat"]))]
    rrt_pairs = [p for p in pairs if p not in comparable and _expert(p["main"]) and _expert(p["repeat"])]
    phase_stats = [x["stats"]["phase"] for x in experts if x["stats"].get("phase")]
    phase = None
    if phase_stats:
        primitives = collections.Counter()
        for s in phase_stats:
            primitives.update(s["primitives"])
        phase = {"degenerate_moves": sum(x["stats"].get("degenerate_moves", 0) for x in experts),
                 "retargets": sum(s["retargets"] for s in phase_stats),
                 "fallbacks": sum(s["fallbacks"] for s in phase_stats),
                 "primitive_limit": sum(s["primitive_limit"] for s in phase_stats),
                 "primitives": dict(primitives)}
    return {
        "phase": phase,
        "outcomes": outcomes,
        "chunk_metrics": metrics,
        "recoverability": recover,
        "planner": {"branches_with_rrt": _rate(x["rrt_used"] for x in experts),
                    "chunks_with_rrt": _rate(_expert(d["full"])["rrt_in_chunk"] for d in decisions if _expert(d["full"])),
                    "branches_with_exhausted": _rate(bool(x["stats"]["exhausted"]) for x in experts),
                    "plan_seconds_per_move": sum(x["plan_seconds"] for x in experts) / moves if moves else None,
                    "expert_seconds_mean": statistics.mean(x["seconds"] for x in experts) if experts else None},
        "targets_removed": sum(1 for d in decisions
                               if any(x.get("removed") for x in (d["full"] or {}).get("targets", []))),
        "gripper_init_mismatches": sum(x["gripper_mismatches"] for x in experts),
        "gripper_inits": sum(x["gripper_inits"] for x in experts),
        "determinism": {"mismatched_pairs": mismatched,
                        "rrt_pairs": len(rrt_pairs),
                        "rrt_pairs_equal": sum(1 for p in rrt_pairs if same_run(_expert(p["main"]), _expert(p["repeat"])))},
        "errors": [{"t": b.get("t"), "mode": b.get("mode"), "error": (b.get("error") or "")[-600:]}
                   for b in branches if b.get("error")][:5],
        "exceptions": [{"t": b.get("t"), "mode": b.get("mode"), "error": (_expert(b)["error"] or "")[-600:]}
                       for b in branches if _expert(b) and _expert(b)["outcome"] == "exception"][:5],
    }


def evaluate_overall(per_task):
    return {"usable_tasks": sorted(task for task, r in per_task.items() if r["usable"]),
            "per_task_usable": {task: r["usable"] for task, r in per_task.items()}}
