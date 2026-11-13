import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.memory_management import event_keyframes as ek
from src.memory_management.validation import event_memory as em
from src.memory_management.validation import oracle_branch as ob
from src.memory_management.validation import planner_branch as pb
from src.memory_management.validation import teacher_oracle as to
from src.memory_management.validation.cached_flow_loss import common_random_conditions
from src.memory_management.validation.partition import partition
from src.memory_management.validation.teacher_probe import stable_seed

ECM = Path(__file__).resolve().parents[3]
PARAMS = {"pool_size": 8, "mc_samples": 16, "batch_size": 16, "se_mult": 2.0, "rel_err_max": 1e-3}


# --------------------------------------------------------------------------- candidates and rules

def test_candidate_swaps_change_exactly_one_history_frame():
    step = 300
    p, pool, swaps = to.candidate_swaps("PatternLock", 3, 7, 2, step, 8)
    assert len(pool) == 8 and pool == sorted(set(pool))
    assert len(swaps) == 8 * len(p.history) == 8 * 27
    uniform = set(p.uniform)
    for s in swaps:
        frames = s["frames"]
        assert len(frames) == 32 and frames == sorted(set(frames))
        assert set(frames) - uniform == {s["frame"]} and uniform - set(frames) == {s["replaced"]}
        assert s["replaced"] == p.history[s["slot"]]
        assert 1 <= s["frame"] < p.recent[0] and s["frame"] not in uniform
        assert set(p.anchor) | set(p.recent) <= set(frames)
    again = to.candidate_swaps("PatternLock", 3, 7, 2, step, 8)[1]
    assert again == pool
    assert to.candidate_swaps("PatternLock", 3, 7, 3, step, 8)[1] != pool


def test_candidate_swaps_without_dropped_frames():
    p, pool, swaps = to.candidate_swaps("PatternLock", 0, 7, 0, 20, 8)
    assert not p.full and pool == [] and swaps == []


def test_small_allowed_set_uses_every_allowed_frame():
    step = 32
    p, pool, swaps = to.candidate_swaps("PatternLock", 0, 7, 0, step, 8)
    allowed = [f for f in range(1, p.recent[0]) if f not in set(p.uniform)]
    assert pool == allowed and len(swaps) == len(allowed) * len(p.history)


def test_paired_gains_and_extremes():
    base = np.linspace(1.0, 1.15, 16)
    noisy = base - 0.1 + np.where(np.arange(16) % 2, 0.5, -0.5)
    losses = np.vstack([base, base, base - 0.05, base + 0.04, noisy])
    mean, se = to.paired_gains(losses)
    np.testing.assert_allclose(mean, [0, 0.05, -0.04, 0.1], atol=1e-12)
    # The rule takes the extreme mean first, then tests it: a noisy best candidate means no swap.
    assert to.extremes(mean, se, 2.0) == {"TO": (3, False), "AO": (2, True)}
    assert to.extremes(mean[:3], se[:3], 2.0) == {"TO": (1, True), "AO": (2, True)}
    flat = to.extremes(np.array([0.01, -0.01]), np.array([0.01, 0.01]), 2.0)
    assert flat == {"TO": (0, False), "AO": (1, False)}


def test_random_swap_is_deterministic_and_in_range():
    picks = [to.random_swap("PatternLock", 1, 8, d, 216) for d in range(50)]
    assert picks == [to.random_swap("PatternLock", 1, 8, d, 216) for d in range(50)]
    assert all(0 <= v < 216 for v in picks) and len(set(picks)) > 10


def test_common_conditions_match_v_b_time_grid():
    noise, tau = to.common_conditions("PatternLock", 2, 9, 4, 16, 20, 32)
    _, tau_vb = common_random_conditions("PatternLock", 2, 64, 16, 20, 32)
    assert noise.shape == (16, 20, 32) and noise.dtype == np.float32
    np.testing.assert_array_equal(tau, tau_vb)
    again, _ = to.common_conditions("PatternLock", 2, 9, 4, 16, 20, 32)
    np.testing.assert_array_equal(noise, again)
    other, _ = to.common_conditions("PatternLock", 2, 8, 4, 16, 20, 32)
    assert not np.array_equal(noise, other)


def test_expert_target_pads_the_unscored_gripper_column():
    state = np.arange(8, dtype=np.float32)
    seven = np.ones((20, 7))
    out = to.expert_target(seven, state)
    assert out.shape == (20, 8) and out.dtype == np.float32 and (out[:, 7] == 7).all()
    eight = np.ones((20, 8), np.float32)
    np.testing.assert_array_equal(to.expert_target(eight, state), eight)
    with pytest.raises(ValueError):
        to.expert_target(np.ones((20, 6)), state)
    assert to.loss_dims("PatternLock") == 7 and to.loss_dims("BinFill") == 8


def test_rs_schedule():
    rows = [{"decision": 0, "swapped": False}, {"decision": 1, "swapped": True}, {"decision": 3, "swapped": True}]
    assert to.rs_schedule(rows) == [1, 3]


# --------------------------------------------------------------------------- server-side logger

class FakeBuffer:
    def __init__(self):
        self._history_feats = {}

    def get_frame_sampling_indices(self, step_idx, token_budget, token_per_image):
        return partition(step_idx).uniform


class FakePolicy:
    def __init__(self, seed=7):
        self._seed, self.step_idx, self.exec_start_idx = seed, -1, 0
        self.config = SimpleNamespace(budget=512, token_per_image=16)
        self.mem_buffer = FakeBuffer()

    def reset(self):
        self.mem_buffer, self.step_idx = FakeBuffer(), -1

    def add_buffer(self, obs):
        self.exec_start_idx = obs["exec_start_idx"]
        self.step_idx += len(obs["images"])

    def _prepare_history(self, obs):
        return obs

    def infer(self, obs):
        return {"actions": np.zeros((20, 8))}


class FakeScorer:
    horizon, action_dim = 20, 32

    def __init__(self, best=None, worst=None, audit=True):
        self.best, self.worst, self.audit, self.calls = best, worst, audit, []
        self.policy = SimpleNamespace(step_idx=-1)

    def reset(self):
        self.policy.step_idx = -1

    def add_buffer(self, obs):
        self.policy.step_idx += len(obs["images"])

    def score(self, raw, memories, target, noise, tau, dims):
        self.calls.append({"memories": memories, "target": target, "dims": dims})
        losses = np.ones((len(memories), len(tau))) + 1e-4 * np.arange(len(tau))
        if self.best is not None:
            losses[1 + self.best] -= 0.1
        if self.worst is not None:
            losses[1 + self.worst] += 0.1
        return losses, {"passed": self.audit, "cached_vs_full_rel_err": 0.0}, {"score_s": 1.0}


def make_logger(tmp_path, condition, scorer=None, schedule=None, step=300, params=None):
    context = {"run_id": "r", "task": "PatternLock", "episode_id": 4, "variant": condition, "condition": condition,
               "directory": str(tmp_path), "rs_schedule": schedule}
    path = tmp_path / "context.json"
    path.write_text(json.dumps(context))
    policy = FakePolicy()
    logger = to.OracleLogger(policy, context_path=path, scorer=scorer or FakeScorer(), params=params or PARAMS)
    policy.reset()
    policy.add_buffer({"images": np.zeros((step + 1, 1, 2, 2, 3), np.uint8), "state": np.zeros((step + 1, 8)),
                       "exec_start_idx": 100})
    return logger, policy


def obs():
    return {"observation/image": np.zeros((4, 4, 3), np.uint8), "observation/wrist_image": np.zeros((4, 4, 3), np.uint8),
            "observation/state": np.arange(8, dtype=np.float32), "prompt": "p"}


def teacher(decision=0, status="ok"):
    return {"teacher_decision": decision, "teacher_status": status, "teacher_actions": np.ones((20, 7), np.float32),
            "teacher_valid_steps": 20 if status == "ok" else 0}


def decide(logger, policy, payload):
    logger.decide(obs(), payload)
    return logger.select(policy.step_idx, 512, 16)


def oracle_rows(tmp_path):
    return [json.loads(x) for x in (tmp_path / "oracle_decisions.jsonl").read_text().splitlines()]


def test_u0_and_s0_keep_the_official_selection(tmp_path):
    for condition in ("U0", "S0"):
        d = tmp_path / condition
        d.mkdir()
        scorer = FakeScorer(best=5)
        logger, policy = make_logger(d, condition, scorer)
        logger.reset()
        policy.add_buffer({"images": np.zeros((301, 1, 2, 2, 3)), "state": np.zeros((301, 8)), "exec_start_idx": 100})
        chosen = decide(logger, policy, teacher() if condition == "S0" else {})
        assert chosen == list(partition(policy.step_idx).uniform)
        row = oracle_rows(d)[0]
        assert row["swapped"] is False and row["n_candidates"] == 216
        if condition == "S0":
            assert len(scorer.calls) == 1 and len(scorer.calls[0]["memories"]) == 217
            assert scorer.calls[0]["dims"] == 7 and scorer.calls[0]["target"].shape == (20, 8)
            assert row["extremes"]["TO"]["index"] == 5 and row["extremes"]["TO"]["accepted"]
            assert (d / "oracle" / "decision_0000.npz").exists()
        else:
            assert scorer.calls == []


def run_one(tmp_path, condition, scorer=None, payload=None, schedule=None):
    logger, policy = make_logger(tmp_path, condition, scorer, schedule)
    logger.reset()
    policy.add_buffer({"images": np.zeros((301, 1, 2, 2, 3)), "state": np.zeros((301, 8)), "exec_start_idx": 100})
    chosen = decide(logger, policy, teacher() if payload is None else payload)
    _, _, swaps = to.candidate_swaps("PatternLock", 4, 7, 0, policy.step_idx, 8)
    return chosen, swaps, oracle_rows(tmp_path)[0], partition(policy.step_idx).uniform


def test_capacity_16_candidates_swap_one_of_eleven_history_slots():
    p, pool, swaps = to.candidate_swaps("PatternLock", 3, 7, 2, 300, 8, capacity=16)
    assert p.uniform == tuple(partition(300, 16).uniform) and len(p.history) == 11 and len(swaps) == 8 * 11
    assert len(pool) == 8 and set(pool).isdisjoint(p.uniform)
    for s in swaps:
        assert len(s["frames"]) == 16 and set(p.uniform) - set(s["frames"]) == {s["replaced"]}
        assert s["frame"] not in p.uniform and 1 <= s["frame"] < p.recent[0]
    assert not to.candidate_swaps("PatternLock", 3, 7, 2, 15, 8, capacity=16)[0].full


def test_capacity_16_conditions_start_from_the_16_frame_selection(tmp_path):
    params = {**PARAMS, "capacity": 16}
    u16 = list(partition(300, 16).uniform)
    swaps = to.candidate_swaps("PatternLock", 4, 7, 0, 300, 8, capacity=16)[2]
    for condition, scorer, payload, expected in (("U0", None, {}, u16), ("S0", FakeScorer(best=5), None, u16),
                                                 ("TO", FakeScorer(best=5), None, swaps[5]["frames"]),
                                                 ("AO", FakeScorer(worst=7), None, swaps[7]["frames"])):
        d = tmp_path / condition
        d.mkdir()
        logger, policy = make_logger(d, condition, scorer, params=params)
        logger.reset()
        policy.add_buffer({"images": np.zeros((301, 1, 2, 2, 3)), "state": np.zeros((301, 8)), "exec_start_idx": 100})
        assert decide(logger, policy, teacher() if payload is None else payload) == expected
        row = oracle_rows(d)[0]
        assert row["uniform"] == u16 and row["capacity"] == 16 and row["n_candidates"] == 88
        if scorer:
            assert [len(m) for m in scorer.calls[0]["memories"]] == [16] * 89
    d = tmp_path / "RS"
    d.mkdir()
    logger, policy = make_logger(d, "RS", schedule=[0], params=params)
    logger.reset()
    policy.add_buffer({"images": np.zeros((301, 1, 2, 2, 3)), "state": np.zeros((301, 8)), "exec_start_idx": 100})
    assert decide(logger, policy, {}) == swaps[to.random_swap("PatternLock", 4, 7, 0, 88)]["frames"]


EVENT = {"alpha": 0.5, "tau": 0.05, "d0": 0.0, "alphas": [0.25, 0.5, 1.0], "pool": "random",
         "signal": {"kind": "ground_truth"}}


def write_event_episode(directory, n=301, center=150):
    """raw_frames.jsonl of a PatternLock episode whose TCP is next to a target button near `center`."""
    with (directory / "raw_frames.jsonl").open("w") as f:
        for i in range(n):
            x = 0.0 if abs(i - center) <= 10 else 0.4
            gt = {"tcp_position": [[x, 0.0, 0.07]], "selected_buttons": [{"name": "target_1", "position": [[0.0, 0.0, 0.07]]}]}
            f.write(json.dumps({"frame_id": i, "exec_start_idx": 100, "gt": gt}) + "\n")
    distances = np.where(np.abs(np.arange(n) - center) <= 10, 0.0, 0.4)
    return distances


def run_event(tmp_path, condition, scorer=None, payload=None, event=None, extra=None):
    tmp_path.mkdir(exist_ok=True)
    distances = write_event_episode(tmp_path)
    params = {**PARAMS, "event": event or EVENT, **(extra or {})}
    logger, policy = make_logger(tmp_path, condition, scorer, params=params)
    logger.reset()
    policy.add_buffer({"images": np.zeros((301, 1, 2, 2, 3)), "state": np.zeros((301, 8)), "exec_start_idx": 100})
    chosen = decide(logger, policy, payload if payload is not None else teacher())
    cfg = em.EventMemoryConfig.from_params(event or EVENT).for_task("PatternLock")
    memories = em.strategy_memories(partition(300), distances, cfg, stable_seed("PatternLock", 4, 7, 0, "evr"))
    return chosen, memories, oracle_rows(tmp_path)[0], logger


def test_event_conditions_feed_the_event_memories(tmp_path):
    for condition in ("EV", "EVU", "EVR", "EVA"):
        chosen, memories, row, _ = run_event(tmp_path / condition, condition, payload={})
        assert chosen == memories[condition] != list(partition(300).uniform)
        assert row["event"]["changed"] > 0 and row["pool_kind"] == "random"
        signal = oracle_rows_named(tmp_path / condition, "event_signal.jsonl")
        assert signal[0]["first_frame"] == 0 and len(signal[0]["distance"]) == 301
    ev = run_event(tmp_path / "EV2", "EV", payload={})[2]["event"]
    assert ev["near"] > ev["near_uniform"]
    with pytest.raises(ValueError):
        make_logger(tmp_path / "EV", "EV")[0].reset()


def oracle_rows_named(directory, name):
    return [json.loads(x) for x in (directory / name).read_text().splitlines()]


def test_es_scores_every_distinct_event_memory_and_te_takes_the_best(tmp_path):
    scorer = FakeScorer(best=1)
    chosen, memories, row, _ = run_event(tmp_path / "ES", "ES", scorer)
    groups = to.event_groups(memories, em.EventMemoryConfig.from_params(EVENT).candidates(), list(partition(300).uniform))
    assert chosen == list(partition(300).uniform) and row["n_candidates"] == len(groups) >= 3
    assert scorer.calls[0]["memories"] == [list(partition(300).uniform)] + [list(k) for k in groups]
    assert set(row["gains"]) == {n for names in groups.values() for n in names}
    assert row["gains"]["EV"] == row["gains"]["EV@0.5"]
    chosen, _, row, _ = run_event(tmp_path / "TE", "TE", FakeScorer(best=1))
    best = list(groups)[1]
    assert chosen == list(best) and row["swapped"] and row["choice"]["name"] == groups[best][0]
    chosen, _, row, _ = run_event(tmp_path / "TE0", "TE", FakeScorer())
    assert chosen == list(partition(300).uniform) and row["reason"] == "below_threshold"
    chosen, _, row, _ = run_event(tmp_path / "TE1", "TE", FakeScorer(best=1), teacher(status="unavailable:stuck"))
    assert row["reason"] == "no_expert:unavailable:stuck" and row["n_candidates"] == len(groups)


def test_event_pool_draws_the_single_swap_candidates(tmp_path):
    event = {**EVENT, "pool": "event"}
    scorer = FakeScorer(best=0)
    chosen, _, row, _ = run_event(tmp_path / "TO", "TO", scorer, event=event)
    pool = em.event_pool(partition(300), write_event_episode(tmp_path), em.EventMemoryConfig.from_params(event), 8)
    assert row["pool"] == pool and row["pool_kind"] == "event" and sum(140 <= f <= 160 for f in pool) >= 4
    assert row["swapped"] and row["choice"]["frame"] in pool


PLUGIN = '''
class Plugin:
    def __init__(self):
        self.seen = []
    def reset(self, context):
        self.seen = []
    def distance(self, frame):
        self.seen.append((frame.frame_id, frame.is_demo, frame.image.shape, frame.state.shape))
        return 0.0 if 140 <= frame.frame_id <= 160 else 0.4
'''


def test_external_signal_sees_each_ingested_frame_once(tmp_path):
    (tmp_path / "plugin_mod.py").write_text(PLUGIN)
    event = {**EVENT, "signal": {"kind": "external", "factory": "plugin_mod:Plugin", "path": str(tmp_path)}}
    chosen, memories, row, logger = run_event(tmp_path / "EV", "EV", payload={}, event=event)
    seen = logger.signal.module.seen
    assert [s[0] for s in seen] == list(range(301)) and seen[99][1] and not seen[100][1]
    assert seen[0][2:] == ((2, 2, 3), (8,))
    assert chosen == memories["EV"]


def test_selection_audit_recomputes_the_event_memories(runner, tmp_path):
    records = []
    for condition, scorer in (("EV", None), ("EVU", None), ("EVR", None), ("EVA", None), ("TE", FakeScorer(best=1)),
                              ("ES", FakeScorer(best=1)), ("U0", None)):
        d = tmp_path / condition
        chosen, _, row, _ = run_event(d, condition, scorer, payload=None if scorer else {})
        (d / "policy_decisions.jsonl").write_text(json.dumps(
            {"decision_idx": 0, "available_max_frame_id": 300, "selected_frame_ids": chosen,
             "valid_tokens": 16 * len(chosen)}) + "\n")
        records.append({"block": condition, "condition": condition, "attempt_directory": str(d), "task": "PatternLock",
                        "episode_id": 4, "model_seed": 7, "oracle": oracle_rows(d)})
    assert runner.selection_audit(records, 32, EVENT) == {"decisions": 7, "mismatches": 0, "examples": []}
    tampered = tmp_path / "EV" / "event_signal.jsonl"
    rows = oracle_rows_named(tmp_path / "EV", "event_signal.jsonl")
    rows[0]["distance"] = [0.4] * 301
    tampered.write_text(json.dumps(rows[0]) + "\n")
    assert runner.selection_audit(records, 32, EVENT)["mismatches"] == 1


KEYFRAMES = {f"K{k}": {"frames": 16, "events": k, "radius": 0.03, "tol": 0.005} for k in (0, 4, 8, 12)}
FINETUNE = {"capacity": 16, "keyframes": KEYFRAMES}


def test_keyframe_conditions_feed_the_event_keyframe_memory(tmp_path):
    distances = write_event_episode(tmp_path)
    for condition in ("K0", "K4", "K12"):
        chosen, _, row, _ = run_event(tmp_path / condition, condition, payload={}, extra=FINETUNE)
        expected, info = ek.select(distances, 300, ek.KeyframeConfig(**KEYFRAMES[condition]))
        assert chosen == expected and row["keyframes"] == info and row["uniform"] == list(partition(300, 16).uniform)
    assert run_event(tmp_path / "K0b", "K0", payload={}, extra=FINETUNE)[0] == list(partition(300, 16).uniform)
    k4 = run_event(tmp_path / "K4b", "K4", payload={}, extra=FINETUNE)[2]
    assert k4["keyframes"] == {"events": 1, "keyframes": [140, 161]} and k4["event"]["changed"] > 0
    with pytest.raises(ValueError):
        run_event(tmp_path / "K8", "K8", payload={}, extra={"capacity": 16})


def test_selection_audit_recomputes_the_keyframe_memories(runner, tmp_path):
    records = []
    for condition in ("K0", "K4", "K8", "K12", "B16"):
        d = tmp_path / condition
        chosen, _, row, _ = run_event(d, condition, payload={}, extra=FINETUNE)
        (d / "policy_decisions.jsonl").write_text(json.dumps(
            {"decision_idx": 0, "available_max_frame_id": 300, "selected_frame_ids": chosen,
             "valid_tokens": 16 * len(chosen)}) + "\n")
        records.append({"block": condition, "condition": condition, "attempt_directory": str(d), "task": "PatternLock",
                        "episode_id": 4, "model_seed": 7, "oracle": oracle_rows(d)})
    assert runner.selection_audit(records, 16, EVENT, KEYFRAMES) == {"decisions": 5, "mismatches": 0, "examples": []}
    rows = oracle_rows_named(tmp_path / "K4", "event_signal.jsonl")
    rows[0]["distance"] = [0.4] * 301
    (tmp_path / "K4" / "event_signal.jsonl").write_text(json.dumps(rows[0]) + "\n")
    assert runner.selection_audit(records, 16, EVENT, KEYFRAMES)["mismatches"] == 1


def test_condition_router_serves_each_condition_from_its_checkpoint(tmp_path):
    sys.path.insert(0, str(ECM / "scripts"))
    import serve_teacher_policy as stp
    calls = []

    class Named(FakePolicy):
        def __init__(self, name):
            super().__init__()
            self.name, self.metadata = name, {"name": name}

        def reset(self):
            calls.append(("reset", self.name))

        def infer(self, obs):
            calls.append(("infer", self.name))

    context = tmp_path / "context.json"
    router = stp.ConditionRouter({"a": Named("a"), "b": Named("b")}, {"K0": "a", "K4": "b", "B16": "a"}, context)
    assert router.metadata == {"name": "a"}
    for condition, name in (("K4", "b"), ("B16", "a")):
        context.write_text(json.dumps({"condition": condition}))
        router.reset()
        router.infer({})
        assert calls[-2:] == [("reset", name), ("infer", name)]
    context.write_text(json.dumps({"condition": "U0"}))
    with pytest.raises(ValueError):
        router.reset()


def test_ablations_keep_only_anchor_and_recent_or_the_current_frame(tmp_path):
    p = partition(300)
    assert to.ablate("NH", p) == [0, *p.recent] and len(to.ablate("NH", p)) == 5
    assert to.ablate("NM", p) == [300]
    early = partition(2)
    assert to.ablate("NH", early) == [0, 1, 2] and to.ablate("NM", early) == [2]
    with pytest.raises(ValueError):
        to.ablate("U0", p)
    for condition in ("NH", "NM"):
        (tmp_path / condition).mkdir()
        scorer = FakeScorer(best=5)
        chosen, _, row, uniform = run_one(tmp_path / condition, condition, scorer, payload={})
        assert chosen == to.ablate(condition, partition(300)) == row["ablated"]
        assert row["swapped"] is False and scorer.calls == []


def test_budget_conditions_use_the_official_sampling_with_fewer_frames(tmp_path):
    p = partition(300)
    assert to.ablate("B16", p) == [int(f) for f in np.linspace(0, 300, 16, dtype=np.int32)]
    assert to.ablate("B8", p) == [int(f) for f in np.linspace(0, 300, 8, dtype=np.int32)]
    assert to.ablate("B8", p)[0] == 0 and to.ablate("B8", p)[-1] == 300
    assert to.ablate("B8", partition(5)) == [0, 1, 2, 3, 4, 5]
    (tmp_path / "b8").mkdir()
    chosen, _, row, _ = run_one(tmp_path / "b8", "B8", payload={})
    assert chosen == to.ablate("B8", partition(300)) == row["ablated"]


def test_to_and_ao_take_the_extreme_candidate(tmp_path):
    (tmp_path / "to").mkdir()
    chosen, swaps, row, _ = run_one(tmp_path / "to", "TO", FakeScorer(best=17, worst=40))
    assert chosen == swaps[17]["frames"] and row["swapped"] and row["choice"]["index"] == 17
    (tmp_path / "ao").mkdir()
    chosen, swaps, row, _ = run_one(tmp_path / "ao", "AO", FakeScorer(best=17, worst=40))
    assert chosen == swaps[40]["frames"] and row["swapped"]


def test_to_keeps_u_below_threshold_or_on_failed_audit_or_without_expert(tmp_path):
    cases = {"flat": (FakeScorer(), None, "below_threshold"),
             "audit": (FakeScorer(best=3, audit=False), None, "audit_failed"),
             "expert": (FakeScorer(best=3), teacher(status="start_mismatch"), "no_expert:start_mismatch")}
    for name, (scorer, payload, reason) in cases.items():
        (tmp_path / name).mkdir()
        chosen, _, row, uniform = run_one(tmp_path / name, "TO", scorer, payload)
        assert chosen == list(uniform) and row["swapped"] is False and row["reason"] == reason


def test_rs_swaps_only_on_the_scheduled_decisions(tmp_path):
    (tmp_path / "on").mkdir()
    chosen, swaps, row, _ = run_one(tmp_path / "on", "RS", payload={}, schedule=[0])
    v = to.random_swap("PatternLock", 4, 7, 0, len(swaps))
    assert chosen == swaps[v]["frames"] and row["swapped"] and row["index"] == v
    (tmp_path / "off").mkdir()
    chosen, _, row, uniform = run_one(tmp_path / "off", "RS", payload={}, schedule=[1])
    assert chosen == list(uniform) and not row["swapped"]


def test_logger_rejects_mismatched_payload_and_unprepared_selection(tmp_path):
    logger, policy = make_logger(tmp_path, "TO", FakeScorer(best=1))
    logger.reset()
    policy.add_buffer({"images": np.zeros((301, 1, 2, 2, 3)), "state": np.zeros((301, 8)), "exec_start_idx": 100})
    with pytest.raises(RuntimeError):
        logger.decide(obs(), teacher(decision=3))
    with pytest.raises(RuntimeError):
        logger.select(policy.step_idx, 512, 16)
    bad = tmp_path / "context.json"
    bad.write_text(json.dumps({**json.loads(bad.read_text()), "condition": "D0a"}))
    with pytest.raises(ValueError):
        logger.reset()


def test_infer_strips_teacher_fields_before_the_policy(tmp_path):
    seen = {}
    logger, policy = make_logger(tmp_path, "U0")
    logger.reset()
    policy.add_buffer({"images": np.zeros((301, 1, 2, 2, 3)), "state": np.zeros((301, 8)), "exec_start_idx": 100})

    def infer(o):
        seen.update(o)
        logger.select(policy.step_idx, 512, 16)
        return {"actions": np.zeros((20, 8)), "infer_time_ms": 1.0}

    logger.original["infer"] = infer
    logger.infer({**obs(), **teacher(status="not_requested")})
    assert set(seen) == set(obs())


# --------------------------------------------------------------------------- simulator side

def write_live(directory, t, exec_start=5, tamper=False):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "actions").mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    chunks = [rng.normal(size=(20, 8)) for _ in range(-(-t // 16))]
    for d, c in enumerate(chunks):
        np.savez(directory / "actions" / f"decision_{d:04d}.npz", actions=c)
    actions = np.concatenate([c[:16] for c in chunks])[:t] if chunks else np.zeros((0, 8))
    with (directory / "control_actions.jsonl").open("w") as f:
        for i, a in enumerate(actions):
            a = a + (1e-9 if tamper and i == t - 1 else 0)
            f.write(json.dumps({"control_step": i + 1, "action": a.tolist(), "stop": False, "outcome": "ongoing"}) + "\n")
    with (directory / "raw_frames.jsonl").open("w") as f:
        for i in range(exec_start + t + 1):
            f.write(json.dumps({"frame_id": i, "control_step": i - exec_start, "exec_start_idx": exec_start, "gt": {}}) + "\n")
    return actions


def test_live_reference(tmp_path):
    actions = write_live(tmp_path / "a", 32)
    ref = ob.live_reference(tmp_path / "a", "PatternLock", 2, 7, 123, 32)
    np.testing.assert_array_equal(ref.actions, actions)
    assert ref.exec_start == 5 and len(ref.frames) == 38 and ref.environment_seed == 123
    write_live(tmp_path / "b", 0)
    (tmp_path / "b" / "control_actions.jsonl").unlink()
    assert ob.live_reference(tmp_path / "b", "PatternLock", 2, 7, 123, 0).actions.shape == (0, 8)
    write_live(tmp_path / "c", 16, tamper=True)
    with pytest.raises(ValueError):
        ob.live_reference(tmp_path / "c", "PatternLock", 2, 7, 123, 16)
    write_live(tmp_path / "d", 16)
    with pytest.raises(ValueError):
        ob.live_reference(tmp_path / "d", "PatternLock", 2, 7, 123, 32)


def test_expert_budget_and_payload():
    params = {"chunk_len": 20, "max_steps": 1300}
    assert ob.expert_params(params, 48)["max_steps"] == 67
    assert ob.expert_params(params, 1290)["max_steps"] == 1300
    run = pb.ExpertRun(48, None, ob.expert_params(params, 48)["max_steps"])
    with pytest.raises(pb.Stop) as stop:
        for _ in range(25):
            run.after_step(np.zeros(7), False, False, "ongoing", 0)
    assert run.n == 20 and stop.value.reason == "timeout"
    full = {"start_ok": True, "expert": {"outcome": "timeout", "chunk": np.ones((20, 7)).tolist(), "stats": {"exhausted": []}}}
    p = ob.payload(full, 20)
    assert p["teacher_status"] == "ok" and p["teacher_actions"].shape == (20, 7) and p["teacher_valid_steps"] == 20
    short = {"start_ok": True, "expert": {"outcome": "success", "chunk": np.arange(12 * 7).reshape(12, 7).tolist(),
                                          "stats": {"exhausted": []}}}
    p = ob.payload(short, 20)
    assert p["teacher_valid_steps"] == 12 and (p["teacher_actions"][12:] == p["teacher_actions"][11]).all()
    stuck = {"start_ok": True, "expert": {"outcome": "stuck", "chunk": np.ones((5, 7)).tolist(), "stats": {"exhausted": []}}}
    assert ob.payload(stuck, 20)["teacher_status"] == "unavailable:stuck"
    assert ob.payload({"start_ok": False}, 20)["teacher_status"] == "start_mismatch"
    assert ob.payload({"start_ok": False, "error": "x"}, 20)["teacher_status"] == "error"


WORKER = """
import json, os, sys, time
fd = int(sys.argv[sys.argv.index('--reply-fd') + 1])
out = os.fdopen(fd, 'w')
for line in sys.stdin:
    t = json.loads(line)['t']
    if t == 99:
        time.sleep(5)
    print('noise on stdout', flush=True)
    out.write(json.dumps({'t': t, 'start_ok': True}) + '\\n')
    out.flush()
"""


def test_branch_client_query_timeout_and_restart(tmp_path):
    client = ob.BranchClient([sys.executable, "-c", WORKER], None, tmp_path / "w.log", timeout_s=1.0)
    client.start()
    assert client.query(16) == {"t": 16, "start_ok": True}
    assert client.query(99)["error"] == "branch worker timeout" and client.proc is None
    assert client.query(32) == {"t": 32, "start_ok": True}
    client.stop()
    assert "noise on stdout" in (tmp_path / "w.log").read_text()


# --------------------------------------------------------------------------- controller and analysis

@pytest.fixture(scope="module")
def runner():
    sys.path.insert(0, str(ECM / "scripts"))
    spec = importlib.util.spec_from_file_location("run_teacher_oracle", ECM / "scripts/run_teacher_oracle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_blocks(runner):
    params = runner.stage_params("pilot", False)
    plan = runner.build_plan(params)
    assert len(plan) == 60 and len({b["key"] for b in plan}) == 60
    for b in plan:
        assert b["order"][-1] == "RS" and sorted(b["order"][:4]) == sorted(["U0", "S0", "TO", "AO"])
    assert plan == runner.build_plan(params)
    assert len({tuple(b["order"]) for b in plan}) > 1


def test_ablation_stage(runner):
    params = runner.stage_params("ablation", False)
    assert params["conditions"] == ["U0", "NH", "NM"] and params["experts"] == {}
    assert params["stats"]["scope"] == "per_task" and set(params["stats"]["contrasts"]) == {"A1", "A2"}
    plan = runner.build_plan(params)
    assert len(plan) == 180 and all(sorted(b["order"]) == ["NH", "NM", "U0"] for b in plan)
    assert len({tuple(b["order"]) for b in plan}) > 1
    assert runner.stage_params("pilot", False)["conditions"] == ["U0", "S0", "TO", "AO", "RS"]


def test_budget_stage_and_route(runner):
    params = runner.stage_params("budget", False)
    assert params["episodes"] == {"PatternLock": list(range(20, 50))} and params["conditions"] == ["U0", "B16", "B8"]
    assert len(runner.build_plan(params)) == 90
    assert runner.stage_params("budget", True)["episodes"] == {"PatternLock": [20]}
    rule = params["stats"]["route"]
    pc = lambda u, b16, b8: {"PatternLock": {"U0": {"success_rate": u}, "B16": {"success_rate": b16},  # noqa: E731
                                             "B8": {"success_rate": b8}}}
    ok = {"G3": True}
    assert runner.budget_route(ok, pc(0.6, 0.45, 0.3), "PatternLock", rule).startswith("选定 16 帧")
    assert runner.budget_route(ok, pc(0.6, 0.55, 0.3), "PatternLock", rule).startswith("选定 8 帧")
    assert runner.budget_route(ok, pc(0.6, 0.55, 0.1), "PatternLock", rule).startswith("两个预算都不满足")
    assert runner.budget_route(ok, pc(0.6, 0.1, 0.05), "PatternLock", rule).startswith("两个预算都不满足")
    assert runner.budget_route({"G3": False}, pc(0.6, 0.45, 0.3), "PatternLock", rule).startswith("工程无效")


def test_budget16_stage_is_the_pilot_with_a_16_frame_memory(runner):
    params, pilot = runner.stage_params("budget16", False), runner.stage_params("pilot", False)
    assert params["oracle"] == {**pilot["oracle"], "capacity": 16} and "capacity" not in pilot["oracle"]
    assert {k: v for k, v in params.items() if k not in ("stage", "oracle")} == \
        {k: v for k, v in pilot.items() if k not in ("stage", "oracle")}
    assert runner.build_plan(params) == runner.build_plan(pilot)
    assert runner.stage_params("budget16", True)["episodes"] == {"PatternLock": [0]}


def test_selection_audit_applies_the_logged_swap(runner, tmp_path):
    u16 = list(partition(300, 16).uniform)
    swapped = sorted((set(u16) - {u16[5]}) | {7})

    def record(name, condition, sent, oracle):
        d = tmp_path / name
        d.mkdir()
        (d / "policy_decisions.jsonl").write_text(json.dumps(
            {"decision_idx": 0, "available_max_frame_id": 300, "selected_frame_ids": sent,
             "valid_tokens": 16 * len(sent)}) + "\n")
        return {"block": name, "condition": condition, "attempt_directory": str(d), "oracle": oracle}

    swap_row = {"decision": 0, "uniform": u16, "swapped": True, "choice": {"frame": 7, "replaced": u16[5], "slot": 4}}
    good = [record("u0", "U0", u16, []), record("to", "TO", swapped, [swap_row]),
            record("s0", "S0", u16, [{"decision": 0, "uniform": u16, "swapped": False}]),
            record("b8", "B8", to.ablate("B8", partition(300, 16)), [])]
    assert runner.selection_audit(good, 16) == {"decisions": 4, "mismatches": 0, "examples": []}
    bad = [record("rs", "RS", u16, [swap_row]), record("s0b", "S0", u16, []),
           record("u0b", "U0", list(partition(300).uniform), [])]
    assert runner.selection_audit(bad, 16)["mismatches"] == 3


def test_event_stage_and_route(runner):
    params = runner.stage_params("event", False)
    assert params["conditions"] == ["U0", "EV", "EVU", "EVR", "EVA"] and params["experts"] == {}
    assert params["stats"]["contrasts"] == {"E1": ["EV", "U0"], "E2": ["EV", "EVU"]}
    assert params["stats"]["secondary"] == {"E3": ["EV", "EVR"], "E4": ["EVA", "U0"]}
    assert params["oracle"]["event"]["signal"] == {"kind": "ground_truth"} and "event" not in PARAMS
    cfg = em.EventMemoryConfig.from_params(params["oracle"]["event"])
    assert [cfg.for_task(t).tau for t in params["tasks"]] == [0.01, 0.05, 0.05]
    assert all(sorted(b["order"]) == sorted(params["conditions"]) for b in runner.build_plan(params))
    assert len(runner.build_plan(params)) == 180 and not set(params["conditions"]) & to.SCORED
    assert "event" not in runner.stage_params("pilot", False)["oracle"]
    assert all("secondary" not in runner.stage_params(s, False)["stats"] for s in runner.STAGES
               if s not in ("event", "finetune", "fruit_dart"))
    ok, tasks = {"G3": True}, ["BinFill", "PatternLock"]
    states = {"E1/BinFill": "improved", "E2/BinFill": "improved", "E1/PatternLock": "improved", "E2/PatternLock": "bounded"}
    assert runner.event_route(ok, states, tasks) == ("BinFill：事件记忆优于官方选择，也优于同一采样器的无事件版本；"
                                                     "PatternLock：优于官方选择，但不优于无事件版本，收益不能归因于事件信号")
    assert runner.event_route({"G3": False}, states, tasks).startswith("工程无效")


def test_finetune_stage_and_route(runner):
    params = runner.stage_params("finetune", False)
    assert params["conditions"] == ["K0", "K4", "K8", "K12", "B16"] and params["experts"] == {}
    assert params["episodes"]["BinFill"] == list(range(20, 50)) and params["runtime"]["servers_per_seed"] == 1
    assert params["oracle"]["capacity"] == 16 and params["oracle"]["keyframes"] == KEYFRAMES
    assert params["checkpoints"]["B16"] == str(runner.CHECKPOINT)
    assert params["checkpoints"]["K8"].endswith("runs/finetune_kf_20261025/ckpt/mme_vla_suite/k8/5999")
    assert params["stats"]["contrasts"] == {"F1": ["K4", "K0"], "F2": ["K8", "K0"], "F3": ["K12", "K0"]}
    assert params["stats"]["secondary"] == {"F4": ["K0", "B16"]} and not set(params["conditions"]) & to.SCORED
    assert len(runner.build_plan(params)) == 270
    assert "checkpoints" not in runner.stage_params("event", False)
    contrasts = params["stats"]["contrasts"]
    states = {"F1/BinFill": "improved", "F2/BinFill": "bounded", "F3/BinFill": "degraded",
              "F1/PatternLock": "bounded", "F2/PatternLock": "bounded", "F3/PatternLock": "bounded",
              "F1/VideoUnmaskSwap": "degraded", "F2/VideoUnmaskSwap": "undetermined", "F3/VideoUnmaskSwap": "bounded"}
    assert runner.finetune_route({"G3": True}, states, ["BinFill", "PatternLock", "VideoUnmaskSwap"], contrasts) == (
        "BinFill：K4 优于同样微调的均匀记忆 K0，K12 差于 K0；PatternLock：三个 k 与 K0 的差都小于 δ，事件关键帧没有可见作用；"
        "VideoUnmaskSwap：K4 差于 K0，没有 k 优于 K0")
    assert runner.finetune_route({"G3": False}, states, ["BinFill"], contrasts).startswith("工程无效")


def test_fruit_stage_and_route(runner):
    params = runner.stage_params("fruit", False)
    assert params["tasks"] == ["FruitCount", "FruitOccludedRipe"] and params["conditions"] == ["K0", "K4", "K8"]
    assert params["episodes"]["FruitOccludedRipe"] == list(range(50)) and params["client"] == "eval_fruit_episode.py"
    assert params["fruit"]["root"].endswith("data/fruit_picking/rm75b_eg2/val") and params["experts"] == {}
    assert params["oracle"]["capacity"] == 16
    assert params["oracle"]["keyframes"] == {c: KEYFRAMES[c] for c in ("K0", "K4", "K8")}
    assert params["checkpoints"] == {f"K{k}": str(ECM / f"runs/fruit_finetune_kf_20261025/ckpt/mme_vla_suite/k{k}/11999")
                                     for k in (0, 4, 8)}
    assert params["stats"]["contrasts"] == {"G1": ["K4", "K0"], "G2": ["K8", "K0"]} and "secondary" not in params["stats"]
    assert params["stats"]["route"] == {"kind": "finetune", "floor": 0.05}
    plan = runner.build_plan(params)
    assert len(plan) == 300 and all(sorted(b["order"]) == ["K0", "K4", "K8"] for b in plan)
    assert runner.stage_params("fruit", True)["episodes"] == {"FruitCount": [0], "FruitOccludedRipe": [0]}
    finetune = runner.stage_params("finetune", False)
    assert "fruit" not in finetune and "client" not in finetune and runner.code_files(finetune) == runner.CODE
    assert set(runner.FRUIT_CODE) <= set(runner.code_files(params))
    signal = runner.em.make_signal(runner.em.EventMemoryConfig.from_params(params["oracle"]["event"]).signal)
    assert signal.needs_frames and type(signal.module).__name__ == "RecordedDistance"
    deps = runner.dependencies(params)
    assert sum(p.name.endswith("_manifest.json") for p in deps) == 100 and ECM / "fruit_picking/env.py" in deps
    assert ECM / "fruit_picking/assets/rm75b_eg2/rm75b_eg2.urdf" in deps
    assert not any("env_metadata" in str(p) for p in deps) and set(runner.va.dependencies([])) <= set(deps)

    contrasts = params["stats"]["contrasts"]
    rates = lambda *r: {c: {"success_rate": x} for c, x in zip(params["conditions"], r)}  # noqa: E731
    per_condition = {"FruitCount": rates(0.0, 0.05, 0.02), "FruitOccludedRipe": rates(0.3, 0.5, 0.4)}
    states = {"G1/FruitCount": "bounded", "G2/FruitCount": "bounded",
              "G1/FruitOccludedRipe": "improved", "G2/FruitOccludedRipe": "bounded"}
    route = lambda: runner.finetune_route({"G3": True}, states, params["tasks"], contrasts, per_condition, 0.05)  # noqa: E731
    assert route() == "FruitCount：地板效应，所有条件成功率都 ≤ 5%，无法比较；FruitOccludedRipe：K4 优于同样微调的均匀记忆 K0"
    per_condition["FruitCount"] = rates(0.0, 0.06, 0.02)
    assert route().startswith("FruitCount：两个 k 与 K0 的差都小于 δ，事件关键帧没有可见作用；")
    assert runner.finetune_route({"G3": True}, states, params["tasks"], contrasts).startswith("FruitCount：两个 k")


def test_fruit_dart_stage_splits_the_reference_onto_its_own_server(runner):
    params = runner.stage_params("fruit_dart", False)
    assert params["conditions"] == ["K0", "K4", "K8", "C0"] and params["episodes"]["FruitCount"] == list(range(50, 100))
    assert params["checkpoints"]["C0"] == str(ECM / "runs/fruit_finetune_kf_20261025/ckpt/mme_vla_suite/k0/11999")
    assert params["checkpoints"]["K8"] == str(ECM / "runs/fruit_dart_kf_20261027/ckpt/mme_vla_suite/k8/11999")
    assert params["stats"]["contrasts"] == {"G1": ["K4", "K0"], "G2": ["K8", "K0"]}
    assert params["stats"]["secondary"] == {"D1": ["K0", "C0"]}
    plan = runner.build_plan(params)
    assert len(plan) == 300 and all(sorted(b["order"]) == ["C0", "K0", "K4", "K8"] for b in plan)
    configs = runner.server_configs(params)
    assert [(c["model_seed"], c["conditions"]) for c in configs] == [
        (s, g) for s in (7, 8, 9) for g in (["K0", "K4", "K8"], ["C0"])]
    assert [c["policy_gpu"] for c in configs] == [0, 1, 2, 3, 4, 5] and {c["sim_gpu"] for c in configs} == {6, 7}
    assert all(set(c["checkpoints"]) == set(c["conditions"]) for c in configs)
    assert len({c["port"] for c in configs}) == 6
    smoke = runner.server_configs(runner.stage_params("fruit_dart", True))
    assert [c["conditions"] for c in smoke] == [["K0", "K4", "K8"], ["C0"]]
    fruit = runner.server_configs(runner.stage_params("fruit", False))
    assert all(c["conditions"] == ["K0", "K4", "K8"] and c["checkpoints"] == params_checkpoints(runner, "fruit")
               for c in fruit)


def params_checkpoints(runner, stage):
    return runner.stage_params(stage, False)["checkpoints"]


def test_block_completion_with_condition_groups(runner):
    protocol = {"params": {"conditions": ["K0", "K4", "K8", "C0"],
                           "runtime": {"condition_groups": [["K0", "K4", "K8"], ["C0"]]}}}
    rows = {**{c: {"server_session": "1@a"} for c in ("K0", "K4", "K8")}, "C0": {"server_session": "2@b"}}
    assert runner.block_complete(protocol, rows)
    assert not runner.block_complete(protocol, {**rows, "K4": {"server_session": "3@c"}})
    assert not runner.block_complete(protocol, {**rows, "C0": {"server_session": None}})


def test_per_task_family_is_its_own_holm_family(runner):
    rng = np.random.default_rng(0)
    outcomes = {}
    for t, lift in (("BinFill", 0.6), ("PatternLock", 0.0)):
        for i in range(20):
            for s in (7, 8, 9):
                base = rng.random() < 0.3
                outcomes[(t, i, s, "U0")] = "success" if base else "fail"
                outcomes[(t, i, s, "EV")] = "success" if base or rng.random() < lift else "fail"
                outcomes[(t, i, s, "EVU")] = outcomes[(t, i, s, "U0")]
    boot = runner.ava.PairedBootstrap(outcomes, ["BinFill", "PatternLock"], 2000, seed=1)
    primary = runner.per_task_family(boot, ["BinFill", "PatternLock"], {"E1": ["EV", "U0"]}, 0.12, 0.05)
    both = runner.per_task_family(boot, ["BinFill", "PatternLock"], {"E1": ["EV", "U0"], "E2": ["EV", "EVU"]}, 0.12, 0.05)
    assert list(primary) == ["E1/BinFill", "E1/PatternLock"] and len(both) == 4
    assert primary["E1/BinFill"]["state"] == "improved" and primary["E1/PatternLock"]["state"] == "bounded"
    assert primary["E1/BinFill"]["p_holm"] <= both["E1/BinFill"]["p_holm"]
    assert {k: primary["E1/BinFill"][k] for k in ("delta", "ci95")} == {k: both["E1/BinFill"][k] for k in ("delta", "ci95")}
    assert runner.per_task_family(boot, ["BinFill"], {}, 0.12, 0.05) == {}


def test_ablation_route(runner):
    ok, tasks = {"G3": True}, ["PatternLock", "BinFill", "VideoUnmaskSwap"]
    states = {"A2/PatternLock": "degraded", "A1/BinFill": "bounded", "A2/BinFill": "bounded",
              "A1/VideoUnmaskSwap": "bounded", "A2/VideoUnmaskSwap": "degraded"}
    verdict = runner.ablation_route(ok, states, tasks, "PatternLock")
    assert verdict == "BinFill：看不出策略使用记忆；VideoUnmaskSwap：只用近期和锚点帧，长历史无关"
    assert runner.ablation_route(ok, {**states, "A1/BinFill": "degraded"}, tasks, "PatternLock").startswith(
        "BinFill：长历史被使用")
    assert "未定" in runner.ablation_route(ok, {**states, "A2/BinFill": "undetermined"}, tasks, "PatternLock")
    assert runner.ablation_route(ok, {**states, "A2/PatternLock": "bounded"}, tasks, "PatternLock").startswith("诊断无效")
    assert runner.ablation_route({"G3": False}, states, tasks, "PatternLock").startswith("工程无效")


def test_extension_stage(runner):
    params = runner.stage_params("extension", False)
    assert params["tasks"] == ["BinFill", "VideoUnmaskSwap"] and set(params["experts"].values()) == {"phase"}
    assert params["runtime"]["episode_wall_timeout_s"] == 14400
    plan = runner.build_plan(params)
    assert len(plan) == 120 and {b["task"] for b in plan} == set(params["tasks"])
    assert sum(b["task"] == "BinFill" for b in plan) == 60
    pilot = runner.stage_params("pilot", False)
    assert {k: v for k, v in pilot.items() if k not in ("stage", "tasks", "experts", "episodes", "runtime")} == \
        {k: v for k, v in params.items() if k not in ("stage", "tasks", "experts", "episodes", "runtime")}


def test_block_completion_requires_one_server_process(runner):
    protocol = {"params": {"conditions": list(to.CONDITIONS)}}
    rows = {c: {"server_session": "1@a"} for c in to.CONDITIONS}
    assert runner.block_complete(protocol, rows)
    assert not runner.block_complete(protocol, {**rows, "RS": {"server_session": "2@b"}})
    assert not runner.block_complete(protocol, {c: r for c, r in rows.items() if c != "RS"})


def test_judge_route_and_divergence(runner):
    res = lambda lo95, hi95, lo90, hi90, p: {"ci95": [lo95, hi95], "ci90": [lo90, hi90], "p_holm": p}  # noqa: E731
    assert runner.judge(res(0.02, 0.3, 0.04, 0.25, 0.01), 0.12, 0.05) == "improved"
    assert runner.judge(res(-0.3, -0.02, -0.25, -0.04, 0.01), 0.12, 0.05) == "degraded"
    assert runner.judge(res(-0.1, 0.1, -0.08, 0.08, 0.6), 0.12, 0.05) == "bounded"
    assert runner.judge(res(-0.05, 0.2, -0.03, 0.15, 0.2), 0.12, 0.05) == "undetermined"
    ok = {"G0": True}
    assert runner.route(ok, {"C1": "improved", "C2": "improved"}) == "有闭环空间"
    assert runner.route(ok, {"C1": "improved", "C2": "bounded"}) == "只是扰动效应"
    assert runner.route(ok, {"C1": "bounded", "C2": "bounded"}) == "没有可见空间"
    assert runner.route({"G0": False}, {"C1": "improved", "C2": "improved"}).startswith("工程无效")
    assert runner.first_divergence(["a", "b"], ["a", "b"]) is None
    assert runner.first_divergence(["a", "b"], ["a", "c"]) == 1
    assert runner.first_divergence(["a"], ["a", "b"]) == 1
