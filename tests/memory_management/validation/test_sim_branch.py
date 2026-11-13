import collections
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import pytest

from src.memory_management.validation import sim_branch as sb

ECM = Path(__file__).resolve().parents[3]
VA_EPISODES = ECM / "results/V_A_formal_20261023/episodes/D0a"


def _frame(i, exec_start, task_index):
    return {"schema": "acm-m3-v1", "frame_id": i, "control_step": i - exec_start, "is_demo": i < exec_start,
            "exec_start_idx": exec_start, "front_sha256": f"f{i}", "wrist_sha256": f"w{i}",
            "state_sha256": f"s{i}", "timestamp_ns": i,
            "gt": {"current_task_index": task_index, "elapsed_steps": i, "front_visible_pixels": {"cube": i},
                   "info": {"simple_subgoal_online": "pick", "grounded_subgoal_online": f"pick at <{i}, 1>"}}}


def _write_reference(directory, n=20, exec_start=2):
    rng = np.random.default_rng(0)
    chunks = [rng.normal(size=(20, 8)) for _ in range(math.ceil(n / 16))]
    (directory / "actions").mkdir(parents=True)
    for k, chunk in enumerate(chunks):
        np.savez(directory / "actions" / f"decision_{k:04d}.npz", actions=chunk)
    executed = np.concatenate([c[:16] for c in chunks])[:n]
    controls = [{"control_step": i + 1, "action": a.tolist(), "stop": i == n - 1,
                 "outcome": "success" if i == n - 1 else "ongoing"} for i, a in enumerate(executed)]
    frames = [_frame(i, exec_start, 0 if i < exec_start + 10 else 1) for i in range(exec_start + n + 1)]
    (directory / "control_actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in controls))
    (directory / "raw_frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in frames))
    (directory / "result.json").write_text(json.dumps({
        "task": "BinFill", "episode_id": 3, "model_seed": 7, "server": 1, "environment_seed": 123,
        "outcome": "success", "executed_steps": n, "raw_frames": len(frames), "exec_start_idx": exec_start}))
    return executed


def test_load_reference_reconstructs_executed_actions(tmp_path):
    executed = _write_reference(tmp_path)
    ref = sb.load_reference(tmp_path)
    assert ref.actions.dtype == np.float64 and np.array_equal(ref.actions, executed)
    assert ref.decision_points() == [0, 16]
    assert ref.frame_at(0)["control_step"] == 0 and ref.frame_at(20)["control_step"] == 20
    assert ref.subgoal_transitions() == [10]


def test_load_reference_rejects_action_drift(tmp_path):
    _write_reference(tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "control_actions.jsonl").read_text().splitlines()]
    rows[5]["action"][0] += 1e-12
    (tmp_path / "control_actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(sb.ReferenceMismatch):
        sb.load_reference(tmp_path)


@pytest.mark.skipif(not VA_EPISODES.exists(), reason="V-A formal run not available")
def test_real_reference_and_branch_points():
    ref = sb.load_reference(VA_EPISODES / "BinFill/ep05_seed7")
    assert ref.n_steps == 701 and len(ref.decision_points()) == 44
    assert all(ref.frame_at(t)["control_step"] == t for t in ref.decision_points())
    assert ref.subgoal_transitions()[:6] == [107, 170, 282, 332, 550, 601]
    points = sb.select_branch_points(ref.decision_points(), ref.subgoal_transitions())
    assert len(points) == 8 and points == sorted(set(points))
    assert {0, 688, 112, 176, 288} <= set(points)
    vus = sb.load_reference(VA_EPISODES / "VideoUnmaskSwap/ep16_seed7")
    assert vus.exec_start == 114 and vus.frame_at(0)["is_demo"] is False and vus.frames[113]["is_demo"] is True


def test_select_branch_points_is_bounded_and_deterministic():
    decisions = list(range(0, 300, 16))
    a = sb.select_branch_points(decisions, [20, 21, 100, 200, 250], max_points=6)
    assert a == sb.select_branch_points(decisions, [20, 21, 100, 200, 250], max_points=6)
    assert len(a) == 6 and a[0] == 0 and a[-1] == 288 and {32, 112, 208} <= set(a)
    assert sb.select_branch_points([0], [], max_points=8) == [0]


def test_compare_rows_separates_render_derived_fields():
    ref = _frame(5, 0, 0)
    render_only = copy.deepcopy(ref)
    render_only.update(front_sha256="x", timestamp_ns=99)
    render_only["gt"]["front_visible_pixels"] = {"cube": 0}
    render_only["gt"]["info"]["grounded_subgoal_online"] = "pick at <0, 0>"
    diff = sb.compare_rows(ref, render_only)
    assert diff["physics"] == [] and sorted(diff["render"]) == [
        "front_sha256", "gt.front_visible_pixels", "gt.info.grounded_subgoal_online"]
    physics = copy.deepcopy(ref)
    physics["state_sha256"] = "y"
    physics["gt"]["current_task_index"] = 3
    assert sb.compare_rows(ref, physics) == {"physics": ["gt.current_task_index", "state_sha256"], "render": []}
    assert sb.row_digest(ref)[0] == sb.row_digest(render_only)[0] != sb.row_digest(physics)[0]


def test_compare_frames_and_controls_report_first_mismatch():
    ref = [_frame(i, 0, 0) for i in range(5)]
    rows = copy.deepcopy(ref)
    rows[3]["state_sha256"] = "bad"
    rows[4]["wrist_sha256"] = "bad"
    out = sb.compare_frames(ref, rows + [_frame(5, 0, 0)])
    assert out["physics_mismatches"] == 2 and out["render_mismatches"] == 1
    assert out["first_physics_mismatch"] == {"frame_id": 3, "control_step": 3, "keys": ["state_sha256"]}
    controls = [{"control_step": 1, "stop": False, "outcome": "ongoing"}, {"control_step": 2, "stop": True, "outcome": "fail"}]
    got = copy.deepcopy(controls)
    got[1]["outcome"] = "success"
    assert sb.compare_controls(controls, controls)["mismatches"] == 0
    assert sb.compare_controls(controls, got)["first_mismatch"]["control_step"] == 2


def test_state_digests_are_bitwise_and_order_free():
    state = {"actors": {"cube": np.ones((1, 13), np.float32)}, "articulations": {"arm": np.zeros((1, 31), np.float32)}}
    base = sb.state_digests(state)
    assert set(base) == {"actors/cube", "articulations/arm"}
    reordered = {"articulations": dict(state["articulations"]), "actors": dict(state["actors"])}
    assert sb.combined_digest(sb.state_digests(reordered)) == sb.combined_digest(base)
    for change in (lambda s: s["actors"].__setitem__("cube", s["actors"]["cube"].astype(np.float64)),
                   lambda s: s["actors"].__setitem__("cube", s["actors"]["cube"].reshape(13, 1)),
                   lambda s: s["actors"]["cube"].__setitem__((0, 0), np.nextafter(np.float32(1), np.float32(2)))):
        other = copy.deepcopy(state)
        change(other)
        assert sb.combined_digest(sb.state_digests(other)) != sb.combined_digest(base)


def test_load_actions():
    actions = np.arange(40 * 8, dtype=np.float64).reshape(40, 8)
    ref = sb.Reference(Path("."), "BinFill", 0, 7, 1, 0, 0, "fail", actions, [], [])
    assert sb.load_actions(ref, 16, "L0").shape == (0, 8)
    l1 = sb.load_actions(ref, 16, "L1", steps=32, delta=0.05)
    assert l1.shape == (32, 8)
    assert np.array_equal(l1[:, :7], actions[np.minimum(np.arange(16, 48), 39), :7] + 0.05)
    assert np.array_equal(l1[:, 7], actions[np.minimum(np.arange(16, 48), 39), 7])
    assert np.array_equal(actions[16], np.arange(16 * 8, 17 * 8))


class _Opaque:
    pass


class _Owner:
    pass


def test_attribute_snapshot_restores_in_place():
    o, shared, opaque = _Owner(), [1, 2], _Opaque()
    o.items, o.obj, o.count = shared, opaque, 5
    o.table = {"a": np.zeros(3), "b": [1]}
    o.arr = np.arange(4.0)
    o.rng, o.gen, o.pyrng = np.random.RandomState(1), np.random.default_rng(2), random.Random(3)
    o.mt = np.random.Generator(np.random.MT19937(4))
    o.tup, o.queue, o.tags = (1, np.ones(2)), collections.deque([1], maxlen=3), {"x"}
    before = sb.attribute_digests(o)
    snap = sb.capture_attributes(o)
    shared.append(3)
    o.table["a"][0] = 9
    o.table["c"] = 1
    o.arr = np.zeros(4)
    o.rng.rand(), o.gen.random(), o.pyrng.random(), o.mt.random()
    o.count += 1
    o.new_attr = 1
    o.queue.append(2)
    o.tags.add("y")
    o.obj = _Opaque()
    after_load = sb.attribute_digests(o)
    assert {k for k in after_load if after_load[k] != before.get(k)} == set(before) - {"tup"} | {"new_attr"}
    sb.restore_attributes(o, snap)
    assert sb.attribute_digests(o) == before
    assert o.items is shared and shared == [1, 2] and o.obj is opaque and not hasattr(o, "new_attr")
    assert o.rng.rand() == np.random.RandomState(1).rand()
    assert o.mt.random() == np.random.Generator(np.random.MT19937(4)).random()
    assert o.queue.maxlen == 3 and list(o.queue) == [1]


class BatchedRNG(np.random.RandomState):
    """Stand-in with ManiSkill's batching __getattribute__ (get_state() of it raises)."""

    def __init__(self, rngs):
        self.rngs = rngs

    def __getattribute__(self, item):
        if item in ("rngs", "__class__", "__dict__", "__getattribute__"):
            return object.__getattribute__(self, item)
        rngs = object.__getattribute__(self, "rngs")
        return lambda *a, **k: np.array([getattr(r, item)(*a, **k) for r in rngs])


def test_attribute_snapshot_handles_batched_rng():
    o = _Owner()
    o.rng = BatchedRNG([np.random.RandomState(1), np.random.RandomState(2)])
    with pytest.raises(ValueError):
        o.rng.get_state()
    before = sb.attribute_digests(o)
    assert not before["rng"].startswith("error:")
    snap = sb.capture_attributes(o)
    o.rng.rand()
    assert sb.attribute_digests(o) != before
    sb.restore_attributes(o, snap)
    assert sb.attribute_digests(o) == before
    assert np.array_equal(o.rng.rand(), [np.random.RandomState(1).rand(), np.random.RandomState(2).rand()])


def test_attribute_digest_marks_unfingerprintable_values():
    class Broken(dict):
        def items(self):
            raise RuntimeError("no")

    o = _Owner()
    o.bad, o.good = Broken(a=1), 3
    digests = sb.attribute_digests(o)
    assert digests["bad"] == "error:test_attribute_digest_marks_unfingerprintable_values.<locals>.Broken:RuntimeError"
    assert not digests["good"].startswith("error:")


def test_attribute_snapshot_handles_torch():
    torch = pytest.importorskip("torch")
    o = _Owner()
    o.steps, o.gen = torch.zeros(2), torch.Generator().manual_seed(5)
    alias = o.steps
    before = sb.attribute_digests(o)
    snap = sb.capture_attributes(o)
    o.steps += 3
    torch.rand(1, generator=o.gen)
    assert sb.attribute_digests(o) != before
    sb.restore_attributes(o, snap)
    assert sb.attribute_digests(o) == before and o.steps is alias and float(alias.sum()) == 0
    assert torch.equal(torch.rand(1, generator=o.gen), torch.rand(1, generator=torch.Generator().manual_seed(5)))


def test_run_forked_isolates_parent_and_reports_failures(tmp_path):
    state = [0]

    def child():
        state.append(1)
        return {"seen": list(state)}

    ok = sb.run_forked(child, tmp_path / "ok.json", 30)
    assert ok["status"] == "ok" and ok["result"] == {"seen": [0, 1]} and state == [0]
    err = sb.run_forked(lambda: 1 / 0, tmp_path / "err.json", 30)
    assert err["status"] == "error" and "ZeroDivisionError" in err["error"]
    slow = sb.run_forked(lambda: time.sleep(10), tmp_path / "slow.json", 0.3)
    assert slow["status"] == "timeout"


# --------------------------------------------------------------------------- gate evaluation

FRAMES = {"frames": 10, "physics_mismatches": 0, "render_mismatches": 0, "first_physics_mismatch": None,
          "first_render_mismatch": None}
BAD_FRAMES = {**FRAMES, "physics_mismatches": 3,
              "first_physics_mismatch": {"frame_id": 20, "control_step": 20, "keys": ["state_sha256"]}}
CONTROLS = {"steps": 5, "mismatches": 0, "first_mismatch": None}


def _trace(t):
    return [{"sim": f"load{t}", "physics": "p", "render": "r", "stop": False, "outcome": "ongoing"}]


def _passing_outputs():
    g0 = {"completed": True, "complete": True, "frames": FRAMES, "controls": CONTROLS, "row_digests": [["p", "r"]],
          "state_digests": {"0": {"combined": "h0", "keys": {}}, "16": {"combined": "h16", "keys": {}}}}
    run = lambda t: {"reached": True, "start": {"t": t, "sim": f"h{t}", "frames": FRAMES, "controls": CONTROLS},
                     "load": _trace(t)}
    fork = lambda t, k: {"t": t, "k": k, "status": "ok", "seconds": 0.1, "pre": f"h{t}", "post": f"h{t}",
                         "start": f"h{t}", "load": _trace(t), "error": None}
    event = {"t": 16, "load_steps": 32, "sim_equal": True, "sim_diff": [], "python_equal": True, "python_diff": [],
             "rng_equal": True}
    s = {"completed": True, "complete": True, "frames": FRAMES, "controls": CONTROLS, "events": [event], "error": None}
    return {
        "g0_A": g0, "g0_B": copy.deepcopy(g0),
        "r": {"completed": True, "points": {"0": [run(0), run(0)], "16": [run(16), run(16)]},
              "off_by_one": {"t": 16, "sim": "other", "row_physics_mismatch": True, "keys": ["gt.elapsed_steps"]}},
        "r_proc_t0000": {"completed": True, "t": 0, **run(0)}, "r_proc_t0016": {"completed": True, "t": 16, **run(16)},
        "f": {"completed": True, "complete": True, "frames": FRAMES, "controls": CONTROLS,
              "forks": [fork(0, 0), fork(0, 1), fork(16, 0), fork(16, 1)]},
        "f_l0": {"completed": True, "complete": True, "frames": FRAMES, "controls": CONTROLS,
                 "forks": [{**fork(0, 0), "load": []}, {**fork(16, 0), "load": []}]},
        "f0": {"completed": True, "ok": False, "error": "KeyError: 'rgb'"},
        "s_L0": s, "s_L1": copy.deepcopy(s),
        "s_nopy": {**s, "complete": False, "frames": BAD_FRAMES, "events": [{**event, "python_equal": False}]},
        "perturb": {"completed": True, "t": 16, "detected": True, "frames": BAD_FRAMES},
    }


def test_evaluate_episode_passing_set():
    g = sb.evaluate_episode(_passing_outputs())
    assert all(g[k] for k in ("G0a", "G0b", "G0c", "R_pass", "F_pass", "S_pass", "F_agrees_with_R",
                              "F_noop_fork_safe"))
    assert not g["F0_headless_ok"] and g["missing_jobs"] == []
    overall = sb.evaluate_overall({"a": g})
    assert overall["step1_passed"] and overall["recommended_method"] == "F" and overall["stop_reason"] is None


def test_evaluate_episode_failure_modes():
    out = _passing_outputs()
    out["r"]["points"]["16"][1]["start"]["sim"] = "drift"
    g = sb.evaluate_episode(out)
    assert not g["R_start_equal"] and not g["R_pass"] and g["F_pass"]

    out = _passing_outputs()
    out["perturb"]["detected"] = False
    g = sb.evaluate_episode(out)
    assert not (g["R_pass"] or g["F_pass"] or g["S_pass"])
    assert not sb.evaluate_overall({"a": g})["step1_passed"]

    out = _passing_outputs()
    out["g0_A"]["frames"] = {**FRAMES, "render_mismatches": 2}
    g = sb.evaluate_episode(out)
    assert g["G0a"] and not g["G0b"] and g["R_pass"] and g["F_pass"]

    out = _passing_outputs()
    out["s_nopy"] = {**out["s_L1"]}
    assert not sb.evaluate_episode(out)["S_pass"]

    out = _passing_outputs()
    out["f"]["forks"][2].update(status="timeout", start=None, load=None)
    g = sb.evaluate_episode(out)
    assert not g["F_pass"] and g["F_fork_failures"] == ["timeout"] and g["R_pass"]
    assert sb.evaluate_overall({"a": g, "b": sb.evaluate_episode(_passing_outputs())})["recommended_method"] == "R"

    out = _passing_outputs()
    out["g0_B"]["state_digests"]["16"]["combined"] = "other"
    g = sb.evaluate_episode(out)
    assert not g["G0c"] and not g["R_pass"]
    assert sb.evaluate_overall({"a": g})["stop_reason"].startswith("reference replay")

    out = _passing_outputs()
    out["f"] = {"completed": False, "exit": 1}
    g = sb.evaluate_episode(out)
    assert g["missing_jobs"] == ["f"] and not g["F_pass"]

    out = _passing_outputs()
    out["f_l0"]["forks"][1]["post"] = "changed"
    g = sb.evaluate_episode(out)
    assert not g["F_noop_fork_safe"] and g["F_pass"]


@pytest.mark.skipif(not VA_EPISODES.exists(), reason="V-A formal run not available")
def test_preflight_plan_jobs():
    sys.path.insert(0, str(ECM / "scripts"))
    spec = importlib.util.spec_from_file_location("run_branch_preflight", ECM / "scripts/run_branch_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref = module.sb.load_reference(VA_EPISODES / "BinFill/ep05_seed7")
    points, jobs = module.plan_jobs(ref, module.PARAMS, smoke=False)
    assert len(points) == 8 and jobs["r"]["points"] == points == jobs["f"]["double_points"] == jobs["s_L1"]["points"]
    assert jobs["r"]["off_by_one"] >= 16 and jobs["s_nopy"]["points"] == [jobs["r"]["off_by_one"]]
    assert sorted(n for n in jobs if n.startswith("r_proc_")) == [f"r_proc_t{points[0]:04d}", f"r_proc_t{points[4]:04d}"]
    assert jobs["perturb"]["t"] == points[4]
    smoke_points, _ = module.plan_jobs(ref, module.PARAMS, smoke=True)
    assert len(smoke_points) == 3
