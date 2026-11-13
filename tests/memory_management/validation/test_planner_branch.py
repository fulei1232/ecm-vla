import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from src.memory_management.validation import planner_branch as pb
from src.memory_management.validation import sim_branch as sb

ECM = Path(__file__).resolve().parents[3]
VA_EPISODES = ECM / "results/V_A_formal_20261023/episodes/D0a"
PARAMS = {"chunk_len": 20, "replan_every": 16, "max_steps": 1300, "stall_steps": 400, "stall_solves": 3,
          "start_steps": 4, "p1_min": 0.9, "p2_min": 0.9, "p3_min": 0.9}


class FakeEnv:
    """Subgoal k completes after needs[k] steps of solve(k); the episode succeeds after the last one."""

    def __init__(self, needs, fail_at=None):
        self.needs, self.fail_at = list(needs), fail_at
        self.subgoal, self.progress, self.steps = 0, 0, 0

    def step(self, run, action):
        run.before_step()
        self.steps += 1
        self.progress += 1
        if self.subgoal < len(self.needs) and self.progress >= self.needs[self.subgoal]:
            self.subgoal, self.progress = self.subgoal + 1, 0
        done = self.subgoal == len(self.needs)
        failed = self.fail_at is not None and self.steps >= self.fail_at
        status = "fail" if failed else "success" if done else "ongoing"
        run.after_step(action, done or failed, False, status, self.subgoal)


def _solver(env, run, length=30, restart=False):
    """solve(k): up to `length` steps; with restart=True, progress is lost whenever solve restarts."""
    def solve(k):
        if restart:
            env.progress = 0
        for _ in range(length):
            env.step(run, np.full(8, float(k)))
    return solve


def _drive(env, run, **kwargs):
    prepared = []
    run.run(lambda: env.subgoal, _solver(env, run, **kwargs), lambda: prepared.append(run.n))
    return prepared


def test_infer_gripper_state():
    assert pb.infer_gripper_state(0.04, -0.01, 0.04, 1, -1) == 1
    assert pb.infer_gripper_state(-0.01, -0.01, 0.04, 1, -1) == -1
    assert pb.infer_gripper_state(0.015, -0.01, 0.04, 1, -1) == -1


def test_full_run_succeeds_and_records_subgoal_changes():
    env, run = FakeEnv([10, 25]), pb.ExpertRun(t=100)
    prepared = _drive(env, run)
    assert run.outcome == "success" and run.n == 35 and prepared == [0]
    assert run.changes == [[10, 1], [35, 2]] and run.replans == 0


def test_rh_replans_every_16_steps_and_reinitialises():
    env, run = FakeEnv([10, 25]), pb.ExpertRun(t=0, replan_every=16)
    prepared = _drive(env, run)
    assert run.outcome == "success" and run.n == 35
    assert prepared == [0, 16, 32] and run.replans == 2


def test_rh_with_restarting_solve_never_finishes_long_subgoal():
    env, run = FakeEnv([20]), pb.ExpertRun(t=0, replan_every=16, stall_steps=100)
    _drive(env, run, restart=True)
    assert run.outcome == "stuck" and run.n == 100


def test_full_run_timeout_counts_steps_before_the_branch():
    env, run = FakeEnv([10_000]), pb.ExpertRun(t=1290, stall_steps=10_000)
    _drive(env, run)
    assert run.outcome == "timeout" and run.n == 11


def test_timeout_is_checked_before_the_env_stop_flag():
    env, run = FakeEnv([11]), pb.ExpertRun(t=1290)
    _drive(env, run)
    assert run.outcome == "timeout" and run.n == 11


def test_failure_and_stalled_solves():
    env, run = FakeEnv([50], fail_at=7), pb.ExpertRun(t=0)
    _drive(env, run)
    assert run.outcome == "fail" and run.n == 7
    env, run = FakeEnv([1000]), pb.ExpertRun(t=0)
    _drive(env, run, length=5)
    assert run.outcome == "stuck" and run.n == 15 and run.solve_calls == 3


def test_replan_interrupted_solves_do_not_count_as_stalled():
    env, run = FakeEnv([1000]), pb.ExpertRun(t=0, replan_every=16, stall_steps=200)
    _drive(env, run, length=30)
    assert run.outcome == "stuck" and run.n == 200 and run.replans == 12


def test_exception_and_swallowed_signal():
    run = pb.ExpertRun(t=0)

    def broken(k):
        raise ValueError("no grasp")
    run.run(lambda: 0, broken, lambda: None)
    assert run.outcome == "exception" and "no grasp" in run.error

    env, run = FakeEnv([5]), pb.ExpertRun(t=0)

    def swallowing(k):
        try:
            for _ in range(10):
                env.step(run, np.zeros(8))
        except:  # noqa: E722 - mirrors the bare excepts in subgoal_planner_func
            pass
    run.run(lambda: env.subgoal, swallowing, lambda: None)
    assert run.outcome == "success" and run.n == 5


class FakeScrewFailure(RuntimeError):
    pass


class FakePlanner:
    OPEN, CLOSED = 1, -1

    def __init__(self, screw, rrt):
        self.screw_results, self.rrt_results = list(screw), list(rrt)
        self.gripper_state = self.OPEN

    def move_to_pose_with_screw(self, pose):
        r = self.screw_results.pop(0)
        if r == "raise":
            raise FakeScrewFailure()
        return r

    def move_to_pose_with_RRTStar(self, pose):
        r = self.rrt_results.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r

    def open_gripper(self):
        return "opened"


class FakeWrapper:
    _demo_screw_max_attempts = 1
    _demo_rrt_max_attempts = 3


def test_instrumented_retry_counts_and_exhaustion():
    run = pb.ExpertRun(t=0)
    planner = pb.instrument_planner(FakePlanner(["raise"], [-1, RuntimeError("ik"), -1]), FakeWrapper(), run,
                                    FakeScrewFailure)
    assert planner.move_to_pose_with_screw(None) == -1
    assert run.stats == {"moves": 1, "screw_failures": 1, "rrt_attempts": 3, "rrt_failures": 3, "exhausted": [0],
                         "degenerate_moves": 0}
    planner = pb.instrument_planner(FakePlanner(["raise"], [-1, "path"]), FakeWrapper(), run, FakeScrewFailure)
    assert planner.move_to_pose_with_screw(None) == "path" and run.stats["rrt_attempts"] == 5
    assert planner.open_gripper() == "opened" and run.source == "other"


def test_degenerate_screw_plan_is_a_completed_move_only_for_the_phase_expert():
    class Degenerate(FakePlanner):
        def move_to_pose_with_screw(self, pose):
            raise RuntimeError(pb.DEGENERATE)

    run = pb.ExpertRun(t=0)
    planner = pb.instrument_planner(Degenerate([], []), FakeWrapper(), run, FakeScrewFailure, degenerate_ok=True)
    assert planner.move_to_pose_with_screw(None) is None and run.stats["degenerate_moves"] == 1
    planner = pb.instrument_planner(Degenerate([], []), FakeWrapper(), run, FakeScrewFailure)
    with pytest.raises(RuntimeError, match=pb.DEGENERATE):
        planner.move_to_pose_with_screw(None)


def test_stop_signal_passes_through_rrt_retry():
    run = pb.ExpertRun(t=0)
    planner = pb.instrument_planner(FakePlanner(["raise"], [pb.Stop("success")]), FakeWrapper(), run,
                                    FakeScrewFailure)
    with pytest.raises(pb.Stop):
        planner.move_to_pose_with_screw(None)
    assert run.stats["rrt_failures"] == 0 and run.source == "other"


def test_record_and_chunk_availability():
    env, run = FakeEnv([10, 25]), pb.ExpertRun(t=0)
    _drive(env, run)
    rec = run.record()
    assert len(rec["chunk"]) == 20 and rec["steps"] == 35 and pb.chunk_available(rec)
    assert rec["actions_sha256"] == sb.array_digest(np.asarray(run.actions))
    assert not pb.chunk_available({**rec, "stats": {**rec["stats"], "exhausted": [19]}})
    assert pb.chunk_available({**rec, "stats": {**rec["stats"], "exhausted": [20]}})
    short = {**rec, "chunk": rec["chunk"][:5]}
    assert pb.chunk_available({**short, "outcome": "success"}) and not pb.chunk_available({**short, "outcome": "exception"})
    assert not pb.chunk_available(None)


def test_chunk_metrics():
    q = np.zeros(7)
    policy = np.zeros((20, 8))
    policy[:, :7] = np.linspace(0.02, 0.4, 20)[:, None]
    policy[:, 7] = -1.003
    m = pb.chunk_metrics(policy.copy(), policy, q, sources=["screw"] * 20)
    assert m["l2_expert_policy"] == 0 and m["direction_cosine"] == pytest.approx(1)
    assert m["start_speed_ratio"] == pytest.approx(1) and m["gripper_agreement"] == 1 and m["hold_steps"] == 0
    hold = np.zeros((20, 8))
    hold[:, 7] = 1
    m = pb.chunk_metrics(hold, policy, q, sources=["gripper"] * 6 + ["screw"] * 14)
    assert m["direction_cosine"] is None and m["start_speed_ratio"] == 0 and m["hold_steps"] == 6
    assert m["gripper_agreement"] == 0 and m["l2_hold_expert"] == 0
    assert pb.chunk_metrics(policy, policy, q, gripper=False)["gripper_agreement"] is None


def _reference(task_indices):
    frames = [{"frame_id": i, "control_step": i, "gt": {"current_task_index": k}} for i, k in enumerate(task_indices)]
    return sb.Reference(Path("."), "BinFill", 0, 7, 0, 0, 0, "success", np.zeros((len(frames) - 1, 8)), frames, [])


def test_fresh_points():
    ref = _reference([0] * 20 + [1] * 25 + [2] * 20)
    assert ref.decision_points() == [0, 16, 32, 48]
    assert pb.fresh_points(ref) == [0, 32, 48]


# --------------------------------------------------------------------------- evaluation

def _expert(outcome="success", digest="d", rrt=False, exhausted=()):
    return {"outcome": outcome, "error": None, "steps": 30, "actions_sha256": digest, "chunk": [[0.0] * 8] * 20,
            "chunk_sources": ["screw"] * 20, "rrt_used": rrt, "rrt_in_chunk": False,
            "stats": {"moves": 2, "screw_failures": 0, "rrt_attempts": int(rrt), "rrt_failures": 0,
                      "exhausted": list(exhausted)},
            "gripper_inits": 1, "gripper_mismatches": 0, "seconds": 1.0, "plan_seconds": 0.1}


def _branch(t, mode, **kwargs):
    rec = {"t": t, "mode": mode, "start_ok": True, "targets": [], "expert": _expert(**kwargs)}
    if mode == "full":
        rec["metrics"] = {k: 0.5 for k in pb.METRICS}
    return rec


def _episode(key, outcome, points, rh="success", full="success"):
    jobs = {"main_00": {"kind": "points", "points": points, "modes": list(pb.MODES), "repeat": False},
            "repeat": {"kind": "points", "points": [points[0], points[-1]], "modes": list(pb.MODES), "repeat": True}}
    info = {"outcome": outcome, "executed_steps": points[-1] + 10, "points": points, "fresh_points": [0], "jobs": jobs}

    def result(spec):
        return {"completed": True, "points": {str(t): {"full": _branch(t, "full", outcome=full),
                                                       "rh": _branch(t, "rh", outcome=rh)} for t in spec["points"]}}
    return {"key": key, "info": info, "outputs": {name: result(spec) for name, spec in jobs.items()}}


def _passing():
    return [_episode("a", "success", [0, 16, 32]), _episode("b", "fail", [0, 16], full="fail", rh="fail")]


def test_evaluate_task_passing():
    r = pb.evaluate_task(_passing(), PARAMS)
    assert all(r["gates"].values()) and r["usable"]
    assert r["counts"]["decisions"] == 5 and r["counts"]["expected_branches"] == r["counts"]["branches"] == 18
    assert r["diagnostics"]["recoverability"]["b"]["full_recoverable"] == 0
    assert set(r["diagnostics"]["chunk_metrics"]) == {"success/fresh", "success/mid", "fail/fresh", "fail/mid"}


def test_evaluate_task_failures():
    eps = _passing()
    del eps[0]["outputs"]["repeat"]
    r = pb.evaluate_task(eps, PARAMS)
    assert not r["gates"]["P0_branch_start"] and r["counts"]["missing_jobs"] == ["a/repeat"]

    eps = _passing()
    eps[0]["outputs"]["main_00"]["points"]["16"]["rh"]["start_ok"] = False
    assert not pb.evaluate_task(eps, PARAMS)["gates"]["P0_branch_start"]

    eps = _passing()
    eps[0]["outputs"]["main_00"]["points"]["16"]["rh"]["expert"]["outcome"] = "stuck"
    r = pb.evaluate_task(eps, PARAMS)
    assert r["gates"]["P2_full_competent"] and not r["gates"]["P3_rh_competent"] and not r["usable"]

    eps = _passing()
    eps[1]["outputs"]["main_00"]["points"]["0"]["full"]["expert"]["stats"]["exhausted"] = [3]
    r = pb.evaluate_task(eps, PARAMS)
    assert r["rates"]["P1_chunk_available"] == 0.8 and not r["gates"]["P1_chunk_available"]


def test_evaluate_task_determinism():
    eps = _passing()
    eps[0]["outputs"]["repeat"]["points"]["32"]["full"]["expert"]["actions_sha256"] = "other"
    r = pb.evaluate_task(eps, PARAMS)
    assert not r["gates"]["P4_deterministic"]
    assert r["diagnostics"]["determinism"]["mismatched_pairs"] == [{"episode": "a", "t": 32, "mode": "full"}]

    eps = _passing()
    eps[0]["outputs"]["repeat"]["points"]["32"]["full"]["expert"].update(rrt_used=True, actions_sha256="other")
    r = pb.evaluate_task(eps, PARAMS)
    assert r["gates"]["P4_deterministic"] and r["diagnostics"]["determinism"]["rrt_pairs"] == 1

    eps = copy.deepcopy(_passing())
    for ep in eps:
        for o in ep["outputs"].values():
            for modes in o["points"].values():
                for rec in modes.values():
                    rec["expert"]["rrt_used"] = True
    assert not pb.evaluate_task(eps, PARAMS)["gates"]["P4_deterministic"]


def _script():
    sys.path.insert(0, str(ECM / "scripts"))
    spec = importlib.util.spec_from_file_location("run_planner_preflight", ECM / "scripts/run_planner_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_episode_selection_and_jobs():
    module = _script()
    rows = [{"condition": "D0a", "task": "T", "model_seed": "7", "episode_id": str(e),
             "outcome": "success" if e in (2, 5, 6, 9, 11) else "fail"} for e in range(12)]
    rows.append({"condition": "D0b", "task": "T", "model_seed": "7", "episode_id": "1", "outcome": "success"})
    assert module.select_episodes(rows, "T", 7, 4) == ([2, 5, 6, 9], [0, 1, 3, 4])
    with pytest.raises(SystemExit):
        module.select_episodes(rows, "T", 7, 6)
    decisions = list(range(0, 16 * 13, 16))
    jobs = module.plan_jobs(decisions, module.PARAMS)
    assert [jobs[f"main_{j:02d}"]["points"] for j in range(3)] == [decisions[:6], decisions[6:12], decisions[12:]]
    assert jobs["repeat"] == {"kind": "points", "points": [0, 96, 192], "modes": ["full", "rh"], "repeat": True}
    assert module.repeat_points([0, 16]) == [0, 16]


@pytest.mark.skipif(not (VA_EPISODES / "BinFill/ep05_seed7").exists(), reason="V-A reference not available")
def test_real_reference_policy_chunk_and_fresh_points():
    ref = sb.load_reference(VA_EPISODES / "BinFill/ep05_seed7")
    chunk = pb.policy_chunk(ref, 32)
    assert chunk.shape == (20, 8) and np.array_equal(chunk[:16], ref.actions[32:48])
    fresh = pb.fresh_points(ref)
    assert fresh[0] == 0 and 112 in fresh and 176 in fresh and 128 not in fresh
