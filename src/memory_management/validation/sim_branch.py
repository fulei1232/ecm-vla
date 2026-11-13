"""Simulator branching for the closed-loop Teacher Oracle, step 1 (branch replay preflight).

Protocol: reports/branch_replay_preflight_protocol_20261023.md. A recorded V-A rollout is the
reference. Three ways of branching off a decision point are compared:
  S  in-process snapshot/restore of sim state, Python task/wrapper/controller state and RNGs;
  R  a fresh env instance reset and replayed to the branch point;
  F  a shadow env forked at the branch point (the parent never runs the branch).
All comparisons are bitwise; there are no numeric tolerances.

Only numpy is imported at module level so that the pure helpers are testable without the
simulator; sim-dependent code imports lazily and expects the V-A sys.path to be set.
"""
from __future__ import annotations

import collections
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np

from .acm_copy.diagnostic_logging import RawRecorder, digest

EXEC_HORIZON = 16
ARM_JOINTS = 7
MAX_DEPTH = 12
ROW_KEYS = ("frame_id", "control_step", "is_demo", "exec_start_idx", "front_sha256", "wrist_sha256",
            "state_sha256", "gt")
RENDER_ROW_KEYS = ("front_sha256", "wrist_sha256")
RENDER_GT_KEYS = ("front_visible_pixels",)
RENDER_INFO_KEYS = ("grounded_subgoal_online",)
_SCALARS = (type(None), bool, int, float, complex, str, bytes)


class ReferenceMismatch(ValueError):
    pass


def _read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _canonical(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- reference

@dataclass
class Reference:
    directory: Path
    task: str
    episode_id: int
    model_seed: int
    server: int
    environment_seed: int
    exec_start: int
    outcome: str
    actions: np.ndarray
    frames: list
    controls: list

    @property
    def n_steps(self):
        return len(self.actions)

    def decision_points(self):
        return list(range(0, self.n_steps, EXEC_HORIZON))

    def frame_at(self, t):
        """Recorded frame after t executed actions (t=0 is the last initial frame)."""
        return self.frames[self.exec_start + t]

    def subgoal_transitions(self):
        return [row["control_step"] for prev, row in zip(self.frames, self.frames[1:])
                if row["control_step"] >= 1
                and row["gt"].get("current_task_index") != prev["gt"].get("current_task_index")]


def load_reference(directory) -> Reference:
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    controls = _read_jsonl(directory / "control_actions.jsonl")
    frames = _read_jsonl(directory / "raw_frames.jsonl")
    chunks = [np.load(p)["actions"] for p in sorted((directory / "actions").glob("decision_*.npz"))]
    n = int(result["executed_steps"])
    if len(controls) != n:
        raise ReferenceMismatch(f"{len(controls)} control rows for {n} executed steps")
    if len(chunks) != math.ceil(n / EXEC_HORIZON):
        raise ReferenceMismatch(f"{len(chunks)} action chunks for {n} executed steps")
    actions = np.concatenate([c[:EXEC_HORIZON] for c in chunks])[:n]
    for i, row in enumerate(controls):
        recorded = np.asarray(row["action"], dtype=actions.dtype)
        if row["control_step"] != i + 1 or not np.array_equal(recorded, actions[i]):
            raise ReferenceMismatch(f"control step {i + 1} differs from the saved action chunk")
    exec_start = int(result["exec_start_idx"])
    if len(frames) != int(result["raw_frames"]) or len(frames) != exec_start + n + 1:
        raise ReferenceMismatch(f"{len(frames)} frames for exec_start={exec_start}, steps={n}")
    for i, row in enumerate(frames):
        if row["frame_id"] != i or row["control_step"] != i - exec_start:
            raise ReferenceMismatch(f"frame {i} is out of order")
    return Reference(directory, result["task"], int(result["episode_id"]), int(result["model_seed"]),
                     int(result["server"]), int(result["environment_seed"]), exec_start, result["outcome"],
                     actions, frames, controls)


def select_branch_points(decisions, transitions, max_points=8, max_transition_points=3):
    """First and last decision, the first decision after up to three subgoal switches, then uniform."""
    decisions = sorted(decisions)
    chosen = {decisions[0], decisions[-1]}
    after, limit = [], min(max_transition_points, max(max_points - len(chosen), 0))
    for step in transitions:
        if len(after) == limit:
            break
        d = next((d for d in decisions if d >= step), None)
        if d is not None and d not in chosen and d not in after:
            after.append(d)
    chosen.update(after)
    rest = [d for d in decisions if d not in chosen]
    need = max_points - len(chosen)
    if need > 0 and rest:
        idx = np.unique(np.round(np.linspace(0, len(rest) - 1, min(need, len(rest)))).astype(int))
        chosen.update(rest[i] for i in idx)
    return sorted(chosen)


def load_actions(ref, t, load, steps=32, delta=0.05):
    """L0: nothing. L1: recorded actions from t, all arm joint targets shifted by delta."""
    if load == "L0":
        return ref.actions[:0].copy()
    if load != "L1":
        raise ValueError(load)
    idx = np.minimum(np.arange(t, t + steps), ref.n_steps - 1)
    actions = ref.actions[idx].copy()
    actions[:, :ARM_JOINTS] += delta
    return actions


# --------------------------------------------------------------------------- frame comparison

def split_row(row):
    """Return (physics, render) parts; render holds fields derived from camera images."""
    physics = _canonical({k: row.get(k) for k in ROW_KEYS})
    render = {k: physics.pop(k) for k in RENDER_ROW_KEYS}
    gt = physics.get("gt") or {}
    for k in RENDER_GT_KEYS:
        if k in gt:
            render[f"gt.{k}"] = gt.pop(k)
    info = gt.get("info") or {}
    for k in RENDER_INFO_KEYS:
        if k in info:
            render[f"gt.info.{k}"] = info.pop(k)
    return physics, render


def _diff_keys(a, b):
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if x == y and (k in a) == (k in b):
            continue
        if k == "gt" and isinstance(x, dict) and isinstance(y, dict):
            out += [f"gt.{g}" for g in sorted(set(x) | set(y)) if x.get(g) != y.get(g) or (g in x) != (g in y)]
        else:
            out.append(k)
    return out


def compare_rows(reference_row, row):
    ref_physics, ref_render = split_row(reference_row)
    physics, render = split_row(row)
    return {"physics": _diff_keys(ref_physics, physics), "render": _diff_keys(ref_render, render)}


def row_digest(row):
    physics, render = split_row(row)
    return [_sha(json.dumps(physics, sort_keys=True)), _sha(json.dumps(render, sort_keys=True))]


def compare_frames(reference_frames, rows):
    first, counts = {"physics": None, "render": None}, {"physics": 0, "render": 0}
    for row in rows:
        idx = row["frame_id"]
        if idx >= len(reference_frames):
            diff = {"physics": ["extra_frame"], "render": []}
        else:
            diff = compare_rows(reference_frames[idx], row)
        for kind in counts:
            if diff[kind]:
                counts[kind] += 1
                if first[kind] is None:
                    first[kind] = {"frame_id": idx, "control_step": row.get("control_step"), "keys": diff[kind]}
    return {"frames": len(rows), "physics_mismatches": counts["physics"], "render_mismatches": counts["render"],
            "first_physics_mismatch": first["physics"], "first_render_mismatch": first["render"]}


def compare_controls(reference_controls, controls):
    mismatches, first = 0, None
    for i, row in enumerate(controls):
        ref = reference_controls[i] if i < len(reference_controls) else None
        if ref is None or (ref["stop"], ref["outcome"]) != (row["stop"], row["outcome"]):
            mismatches += 1
            if first is None:
                first = {"control_step": i + 1, "reference": None if ref is None else [ref["stop"], ref["outcome"]],
                         "replay": [row["stop"], row["outcome"]]}
    return {"steps": len(controls), "mismatches": mismatches, "first_mismatch": first}


# --------------------------------------------------------------------------- state digests

def _as_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value))


def array_digest(value):
    a = _as_array(value)
    h = hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def flatten_state(state, prefix=""):
    out = {}
    for k in sorted(state):
        name = f"{prefix}{k}"
        if isinstance(state[k], dict):
            out.update(flatten_state(state[k], name + "/"))
        else:
            out[name] = state[k]
    return out


def state_digests(state):
    return {k: array_digest(v) for k, v in flatten_state(state).items()}


def combined_digest(digests):
    return _sha(json.dumps(sorted(digests.items())))


# --------------------------------------------------------------------------- Python state (method S)

def _torch():
    return sys.modules.get("torch")


def _batched_rngs(value):
    """ManiSkill's BatchedRNG subclasses RandomState but batches every method; its state is .rngs."""
    if type(value).__name__ == "BatchedRNG":
        return object.__getattribute__(value, "rngs")
    return None


def _fingerprint(value, h, depth=0):
    torch = _torch()
    if isinstance(value, _SCALARS):
        h.update(f"{type(value).__name__}:{value!r};".encode())
    elif isinstance(value, (np.ndarray, np.generic)) or (torch is not None and isinstance(value, torch.Tensor)):
        h.update(f"arr:{array_digest(value)};".encode())
    elif depth >= MAX_DEPTH:
        h.update(f"deep:{type(value).__qualname__}@{id(value)};".encode())
    elif _batched_rngs(value) is not None:
        _fingerprint(list(_batched_rngs(value)), h, depth + 1)
    elif isinstance(value, np.random.RandomState):
        _fingerprint(value.get_state(), h, depth + 1)
    elif isinstance(value, np.random.Generator):
        _fingerprint(value.bit_generator.state, h, depth + 1)
    elif torch is not None and isinstance(value, torch.Generator):
        _fingerprint(value.get_state(), h, depth + 1)
    elif isinstance(value, random.Random):
        _fingerprint(value.getstate(), h, depth + 1)
    elif isinstance(value, (list, tuple, collections.deque)):
        h.update(f"{type(value).__name__}[{len(value)}];".encode())
        for v in value:
            _fingerprint(v, h, depth + 1)
    elif isinstance(value, dict):
        h.update(f"{type(value).__name__}{{{len(value)}}};".encode())
        for k, v in value.items():
            _fingerprint(k, h, depth + 1)
            _fingerprint(v, h, depth + 1)
    elif isinstance(value, (set, frozenset)):
        h.update(f"{type(value).__name__}<{len(value)}>;".encode())
        for v in sorted(value, key=repr):
            _fingerprint(v, h, depth + 1)
    else:
        # Opaque objects (sapien actors, closures, owners) compare by identity within one process.
        h.update(f"obj:{type(value).__module__}.{type(value).__qualname__}@{id(value)};".encode())


def value_digest(value):
    h = hashlib.sha256()
    _fingerprint(value, h)
    return h.hexdigest()


def attribute_digests(owner):
    """Per-attribute digests; attributes that cannot be fingerprinted get a stable 'error:' marker."""
    out = {}
    for k, v in vars(owner).items():
        try:
            out[k] = value_digest(v)
        except Exception as exc:
            out[k] = f"error:{type(v).__qualname__}:{type(exc).__name__}"
    return out


def _capture(value, depth=0):
    torch = _torch()
    if isinstance(value, _SCALARS) or depth >= MAX_DEPTH:
        return ("ref", value, None)
    if isinstance(value, np.ndarray):
        return ("ndarray", value, value.copy())
    if torch is not None and isinstance(value, torch.Tensor):
        return ("tensor", value, value.detach().clone())
    if _batched_rngs(value) is not None:
        return ("batched", value, _capture(_batched_rngs(value), depth + 1))
    if isinstance(value, np.random.RandomState):
        return ("rs", value, value.get_state())
    if isinstance(value, np.random.Generator):
        return ("gen", value, copy.deepcopy(value.bit_generator.state))
    if torch is not None and isinstance(value, torch.Generator):
        return ("tgen", value, value.get_state())
    if isinstance(value, random.Random):
        return ("pyrng", value, value.getstate())
    if isinstance(value, list):
        return ("list", value, [_capture(v, depth + 1) for v in value])
    if isinstance(value, collections.deque):
        return ("deque", value, [_capture(v, depth + 1) for v in value])
    if isinstance(value, dict):
        return ("dict", value, [(k, _capture(v, depth + 1)) for k, v in value.items()])
    if isinstance(value, set):
        return ("set", value, list(value))
    if isinstance(value, tuple):
        return ("tuple", value, [_capture(v, depth + 1) for v in value])
    return ("ref", value, None)


def _restore(node):
    """Restore mutable objects in place so that aliases held elsewhere see the restored value."""
    kind, obj, data = node
    if kind == "ref":
        return obj
    if kind == "ndarray":
        if obj.flags.writeable and obj.shape == data.shape and obj.dtype == data.dtype:
            obj[...] = data
            return obj
        return data.copy()
    if kind == "tensor":
        torch = _torch()
        if obj.shape == data.shape and obj.dtype == data.dtype:
            with torch.no_grad():
                obj.copy_(data)
            return obj
        return data.clone()
    if kind == "batched":
        _restore(data)
    elif kind == "rs":
        obj.set_state(data)
    elif kind == "gen":
        obj.bit_generator.state = data
    elif kind == "tgen":
        obj.set_state(data)
    elif kind == "pyrng":
        obj.setstate(data)
    elif kind == "list":
        obj[:] = [_restore(n) for n in data]
    elif kind == "deque":
        items = [_restore(n) for n in data]
        obj.clear()
        obj.extend(items)
    elif kind == "dict":
        items = [(k, _restore(n)) for k, n in data]
        obj.clear()
        obj.update(items)
    elif kind == "set":
        obj.clear()
        obj.update(data)
    elif kind == "tuple":
        items = [_restore(n) for n in data]
        return tuple(items) if type(obj) is tuple else type(obj)(*items)
    return obj


def capture_attributes(owner, uncaptured=None):
    """Snapshot of vars(owner); attributes that fail to copy are kept by reference and listed."""
    out = {}
    for k, v in vars(owner).items():
        try:
            out[k] = _capture(v)
        except Exception as exc:
            out[k] = ("ref", v, None)
            if uncaptured is not None:
                uncaptured.append(f"{k}:{type(exc).__name__}")
    return out


def restore_attributes(owner, captured):
    attrs = vars(owner)
    for k in [k for k in attrs if k not in captured]:
        del attrs[k]
    for k, node in captured.items():
        attrs[k] = _restore(node)


def _global_rng_state():
    torch = _torch()
    state = {"python": random.getstate(), "numpy": np.random.get_state()}
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_global_rng_state(state):
    torch = _torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def state_owners(runner):
    """Objects whose __dict__ holds episode state: runner, wrappers, task env, agent, controllers."""
    base = runner.env.unwrapped
    owners, env, i = [("runner", runner)], runner.env, 0
    while env is not None and env is not base:
        owners.append((f"wrapper{i}:{type(env).__name__}", env))
        env, i = getattr(env, "env", None), i + 1
    owners.append(("env", base))
    agent = getattr(base, "agent", None)
    if agent is not None:
        owners.append(("agent", agent))
        controller = getattr(agent, "controller", None)
        if controller is not None:
            owners.append(("controller", controller))
            for uid, sub in (getattr(controller, "controllers", None) or {}).items():
                owners.append((f"controller.{uid}", sub))
    return [(label, owner) for label, owner in owners if hasattr(owner, "__dict__")]


def python_digests(runner):
    return {f"{label}.{k}": v for label, owner in state_owners(runner) for k, v in attribute_digests(owner).items()}


def rng_digest():
    return value_digest(_global_rng_state())


def _clone_tree(value):
    if isinstance(value, dict):
        return {k: _clone_tree(v) for k, v in value.items()}
    return value.clone() if hasattr(value, "clone") else np.array(value, copy=True)


class InProcessSnapshot:
    """Method S. include_python=False keeps only the ManiSkill state dict (sensitivity control)."""

    def __init__(self, runner, include_python=True):
        base = runner.env.unwrapped
        self.runner, self.include_python = runner, include_python
        self.sim = base.get_state_dict()
        self.controller = _clone_tree(base.agent.get_controller_state())
        self.uncaptured = []
        self.python = ([(owner, capture_attributes(owner, self.uncaptured)) for _, owner in state_owners(runner)]
                       if include_python else [])
        self.rng = _global_rng_state() if include_python else None

    def restore(self):
        base = self.runner.env.unwrapped
        for owner, captured in self.python:
            restore_attributes(owner, captured)
        if self.rng is not None:
            _set_global_rng_state(self.rng)
        base.set_state_dict(self.sim)
        if self.controller:
            base.agent.set_controller_state(self.controller)


# --------------------------------------------------------------------------- forked branches (method F)

def run_forked(fn, out_path, timeout_s):
    """Run fn() in a forked child; the parent's memory is never touched by the child."""
    out_path = Path(out_path)
    out_path.unlink(missing_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    start = time.monotonic()
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            payload = {"ok": True, "result": fn()}
        except BaseException:
            payload, code = {"ok": False, "error": traceback.format_exc()}, 1
        try:
            tmp = out_path.with_name(out_path.name + ".tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, out_path)
        finally:
            os._exit(code)
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        if time.monotonic() - start > timeout_s:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            return {"status": "timeout", "seconds": round(time.monotonic() - start, 3)}
        time.sleep(0.005)
    seconds = round(time.monotonic() - start, 3)
    if os.WIFSIGNALED(status):
        return {"status": f"signal_{os.WTERMSIG(status)}", "seconds": seconds}
    if not out_path.exists():
        return {"status": "no_output", "exit_code": os.WEXITSTATUS(status), "seconds": seconds}
    payload = json.loads(out_path.read_text())
    if not payload["ok"]:
        return {"status": "error", "error": payload["error"], "seconds": seconds}
    return {"status": "ok", "result": payload["result"], "seconds": seconds}


# --------------------------------------------------------------------------- replay engine

class MemoryRecorder(RawRecorder):
    """RawRecorder with the identical hooks, hashing and GT snapshot; frames stay in memory."""

    def add(self, front, wrist, state, details, idx, control, demo):
        self.rows.append({"frame_id": idx, "control_step": control, "is_demo": demo,
                          "exec_start_idx": self.exec_start, **details,
                          "wrist_sha256": digest(wrist), "state_sha256": digest(state)})

    def flush(self):
        pass


class Replayer:
    """One env instance driven through the V-A path: EnvRunner(val) + RawRecorder hooks."""

    def __init__(self, ref, scratch, max_steps=1300, record=True):
        from env_runner import EnvRunner
        from robomme.env_record_wrapper import BenchmarkEnvBuilder

        self.ref, self.scratch = ref, Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.runner = EnvRunner(ref.task, str(self.scratch), max_steps=max_steps)
        self.runner.env_builder = BenchmarkEnvBuilder(env_id=ref.task, dataset="val", action_space="joint_angle",
                                                      gui_render=False, max_steps=max_steps)
        seed = int(self.runner.env_builder.metadata_index[(ref.task, ref.episode_id)]["seed"])
        if seed != ref.environment_seed:
            raise ReferenceMismatch(f"val metadata seed {seed} != recorded {ref.environment_seed}")
        self.runner.make_env(ref.episode_id)
        self.recorder = None
        if record:
            self.recorder = MemoryRecorder(self.scratch, ref.task)
            self.recorder.install(self.runner)
        self.t, self.controls = None, []

    @property
    def base(self):
        return self.runner.env.unwrapped

    @property
    def rows(self):
        return self.recorder.rows

    def reset(self):
        self.runner.get_init_obs()
        if self.recorder is not None and self.recorder.exec_start != self.ref.exec_start:
            raise ReferenceMismatch(f"exec_start {self.recorder.exec_start} != {self.ref.exec_start}")
        self.t = 0
        return self

    def step(self, action):
        _, stop, outcome = self.runner.step(action)
        self.t += 1
        self.controls.append({"control_step": self.t, "stop": bool(stop), "outcome": outcome})
        return bool(stop)

    def advance_to(self, t, actions=None):
        """Step recorded actions until t; returns False if the episode stopped earlier."""
        actions = self.ref.actions if actions is None else actions
        while self.t < t:
            if self.step(actions[self.t]) and self.t < t:
                return False
        return True

    def sim_digests(self):
        return state_digests(self.base.get_state_dict())

    def start_summary(self):
        return {"t": self.t, "sim": combined_digest(self.sim_digests()),
                "frames": compare_frames(self.ref.frames, self.rows),
                "controls": compare_controls(self.ref.controls, self.controls)}

    def mark(self):
        return self.t, len(self.controls), len(self.rows), self.recorder.control_step

    def rollback(self, mark):
        self.t, n_controls, n_rows, control_step = mark
        del self.controls[n_controls:]
        del self.recorder.rows[n_rows:]
        self.recorder.control_step = control_step
        self.recorder.pending.clear()

    def backend(self):
        backend = getattr(self.base, "backend", None)
        return {"sim_backend": getattr(backend, "sim_backend", None),
                "render_backend": getattr(backend, "render_backend", None),
                "gpu_sim_enabled": bool(getattr(self.base, "gpu_sim_enabled", False))}

    def close(self):
        self.runner.close_env()


def run_load(replayer, actions):
    trace = []
    for action in actions:
        stop = replayer.step(action)
        physics, render = row_digest(replayer.rows[-1]) if replayer.rows else (None, None)
        trace.append({"sim": combined_digest(replayer.sim_digests()), "physics": physics, "render": render,
                      "stop": stop, "outcome": replayer.controls[-1]["outcome"]})
        if stop:
            break
    return trace


def full_digests(replayer):
    return {"sim": replayer.sim_digests(), "python": python_digests(replayer.runner), "rng": rng_digest()}


def _changed(a, b, limit=20):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))[:limit]


# --------------------------------------------------------------------------- jobs

def job_g0(ref, spec, scratch, params, dump_path=None):
    rp = Replayer(ref, scratch).reset()
    points, states, arrays = set(ref.decision_points()), {}, {}
    for t in range(ref.n_steps + 1):
        if t in points:
            sd = rp.base.get_state_dict()
            keys = state_digests(sd)
            states[str(t)] = {"combined": combined_digest(keys), "keys": keys}
            if dump_path is not None:
                arrays.update({f"t{t:04d}/{k}": _as_array(v) for k, v in flatten_state(sd).items()})
        if t == ref.n_steps or (rp.step(ref.actions[t]) and t + 1 < ref.n_steps):
            break
    if dump_path is not None:
        Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dump_path, **arrays)
    out = {"complete": len(rp.rows) == len(ref.frames), "frames": compare_frames(ref.frames, rp.rows),
           "controls": compare_controls(ref.controls, rp.controls),
           "row_digests": [row_digest(r) for r in rp.rows], "state_digests": states, "backend": rp.backend()}
    rp.close()
    return out


def _branch_by_replay(ref, t, load, scratch, params):
    rp = Replayer(ref, scratch).reset()
    reached = rp.advance_to(t)
    start = rp.start_summary()
    trace = run_load(rp, load_actions(ref, t, load, params["load_steps"], params["load_delta"])) if reached else []
    rp.close()
    return {"reached": reached, "start": start, "load": trace}


def job_r(ref, spec, scratch, params, dump_path=None):
    points = {}
    for t in spec["points"]:
        points[str(t)] = [_branch_by_replay(ref, t, spec["load"], Path(scratch) / f"t{t:04d}_{k}", params)
                          for k in range(spec["repeats"])]
    t = spec["off_by_one"]
    rp = Replayer(ref, Path(scratch) / "off_by_one").reset()
    rp.advance_to(t - 1)
    unlabeled = lambda row: {k: v for k, v in row.items() if k not in ("frame_id", "control_step")}
    diff = compare_rows(unlabeled(ref.frame_at(t)), unlabeled(rp.rows[-1]))
    off_by_one = {"t": t, "sim": combined_digest(rp.sim_digests()), "row_physics_mismatch": bool(diff["physics"]),
                  "keys": diff["physics"]}
    rp.close()
    return {"points": points, "off_by_one": off_by_one}


def job_r_point(ref, spec, scratch, params, dump_path=None):
    return {"t": spec["t"], **_branch_by_replay(ref, spec["t"], spec["load"], scratch, params)}


def job_f(ref, spec, scratch, params, dump_path=None):
    scratch = Path(scratch)
    rp = Replayer(ref, scratch / "parent").reset()
    double, forks, error = set(spec["double_points"]), [], None
    try:
        for t in ref.decision_points():
            if not rp.advance_to(t):
                break
            pre = combined_digest(rp.sim_digests())
            for k in range(2 if t in double else 1):
                child_dir = scratch / f"child_t{t:04d}_{k}"

                def child(t=t, child_dir=child_dir):
                    child_dir.mkdir(parents=True, exist_ok=True)
                    rp.recorder.directory = child_dir
                    start = combined_digest(rp.sim_digests())
                    actions = load_actions(ref, t, spec["load"], params["load_steps"], params["load_delta"])
                    return {"start": start, "load": run_load(rp, actions)}

                res = run_forked(child, scratch / f"fork_t{t:04d}_{k}.json", params["fork_timeout_s"])
                post = combined_digest(rp.sim_digests())
                forks.append({"t": t, "k": k, "status": res["status"], "seconds": res["seconds"], "pre": pre,
                              "post": post, "start": (res.get("result") or {}).get("start"),
                              "load": (res.get("result") or {}).get("load"), "error": res.get("error")})
        rp.advance_to(ref.n_steps)
    except Exception:
        # A renderer broken by the fork makes the parent's next step fail; that is the finding.
        error = traceback.format_exc()[-3000:]
    out = {"load": spec["load"], "complete": error is None and len(rp.rows) == len(ref.frames),
           "frames": compare_frames(ref.frames, rp.rows), "controls": compare_controls(ref.controls, rp.controls),
           "forks": forks, "error": error, "backend": rp.backend(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    if error is None:
        rp.close()
    return out


def job_f0(ref, spec, scratch, params, dump_path=None):
    """Probe whether the unpatched wrapper stack runs without rendering (obs_mode='state')."""
    import gymnasium as gym

    original = gym.make

    def make(env_id, **kwargs):
        kwargs["obs_mode"] = "state"
        return original(env_id, **kwargs)

    gym.make = make
    try:
        rp = Replayer(ref, scratch, record=False)
        rp.runner.env.reset()
        start = time.monotonic()
        for action in ref.actions[:spec["steps"]]:
            rp.runner.env.step(action)
        out = {"ok": True, "error": None, "seconds_per_step": (time.monotonic() - start) / spec["steps"]}
        rp.close()
    except Exception as exc:
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}
    finally:
        gym.make = original
    return out


def job_s(ref, spec, scratch, params, dump_path=None):
    rp = Replayer(ref, scratch).reset()
    events, error = [], None
    try:
        for t in sorted(spec["points"]):
            if not rp.advance_to(t):
                break
            before = full_digests(rp)
            mark = rp.mark()
            snap = InProcessSnapshot(rp.runner, include_python=spec["python"])
            trace = run_load(rp, load_actions(ref, t, spec["load"], params["load_steps"], params["load_delta"]))
            snap.restore()
            rp.rollback(mark)
            after = full_digests(rp)
            events.append({"t": t, "load_steps": len(trace),
                           "sim_equal": before["sim"] == after["sim"],
                           "sim_diff": _changed(before["sim"], after["sim"]),
                           "python_equal": before["python"] == after["python"],
                           "python_diff": _changed(before["python"], after["python"]),
                           "rng_equal": before["rng"] == after["rng"], "uncaptured": snap.uncaptured[:20],
                           "unfingerprinted": sorted(k for k, v in before["python"].items()
                                                     if v.startswith("error:"))[:20]})
        rp.advance_to(ref.n_steps)
    except Exception:
        # A restored env that diverges may fail later; the divergence itself is the result.
        error = traceback.format_exc()[-3000:]
    out = {"python": spec["python"], "load": spec["load"], "events": events, "error": error,
           "complete": error is None and len(rp.rows) == len(ref.frames),
           "frames": compare_frames(ref.frames, rp.rows), "controls": compare_controls(ref.controls, rp.controls)}
    rp.close()
    return out


def job_perturb(ref, spec, scratch, params, dump_path=None):
    t = spec["t"]
    actions = ref.actions.copy()
    actions[t, spec["joint"]] += spec["delta"]
    rp = Replayer(ref, scratch).reset()
    error = None
    try:
        rp.advance_to(ref.n_steps, actions)
    except Exception:
        error = traceback.format_exc()[-3000:]
    frames = compare_frames(ref.frames, rp.rows)
    first = frames["first_physics_mismatch"]
    rp.close()
    return {"t": t, "frames": frames, "error": error,
            "detected": bool(first and first["control_step"] is not None and first["control_step"] > t)}


JOBS = {"g0": job_g0, "r": job_r, "r_point": job_r_point, "f": job_f, "f0": job_f0, "s": job_s,
        "perturb": job_perturb}


def run_job(ref, spec, scratch, params, dump_path=None):
    return JOBS[spec["kind"]](ref, spec, scratch, params, dump_path)


# --------------------------------------------------------------------------- evaluation

def _replay_ok(o, render):
    f = o["frames"]
    return bool(o.get("complete") and f["physics_mismatches"] == 0 and o["controls"]["mismatches"] == 0
                and (not render or f["render_mismatches"] == 0))


def _start_ok(start, reference_digest, render):
    f = start["frames"]
    return bool(reference_digest is not None and start["sim"] == reference_digest and f["physics_mismatches"] == 0
                and start["controls"]["mismatches"] == 0 and (not render or f["render_mismatches"] == 0))


def traces_equal(a, b, render):
    keys = ("sim", "physics", "stop", "outcome") + (("render",) if render else ())
    return bool(a) and b is not None and len(a) == len(b) and all(x[k] == y[k] for x, y in zip(a, b) for k in keys)


def evaluate_episode(outputs):
    """Gate verdicts for one episode from its job outputs (None or completed=False means missing)."""
    get = lambda name: outputs.get(name) if (outputs.get(name) or {}).get("completed") else None
    A, B = get("g0_A"), get("g0_B")
    missing = sorted(n for n in outputs if get(n) is None)
    g = {"missing_jobs": missing}
    g["G0a"] = bool(A and B and _replay_ok(A, False) and _replay_ok(B, False))
    g["G0b"] = bool(A and B and _replay_ok(A, True) and _replay_ok(B, True))
    g["G0c"] = bool(A and B and {t: v["combined"] for t, v in A["state_digests"].items()}
                    == {t: v["combined"] for t, v in B["state_digests"].items()}
                    and [d[0] for d in A["row_digests"]] == [d[0] for d in B["row_digests"]])
    g["G0c_render"] = bool(A and B and [d[1] for d in A["row_digests"]] == [d[1] for d in B["row_digests"]])
    render = g["G0b"]
    ref = {int(t): v["combined"] for t, v in A["state_digests"].items()} if A else {}

    p = get("perturb")
    g["control_perturb_detected"] = bool(p and p["detected"])

    r = get("r")
    obo = (r or {}).get("off_by_one")
    g["control_off_by_one_detected"] = bool(obo and obo["row_physics_mismatch"] and obo["sim"] != ref.get(obo["t"]))
    runs = [(int(t), x) for t, xs in (r or {}).get("points", {}).items() for x in xs]
    g["R_start_equal"] = bool(r and runs and all(x["reached"] and _start_ok(x["start"], ref.get(t), render)
                                                 for t, x in runs))
    g["R_load_deterministic"] = bool(r and all(traces_equal(xs[0]["load"], xs[1]["load"], render)
                                               for xs in r["points"].values()))
    procs = [get(n) for n in outputs if n.startswith("r_proc_")]
    g["R_fresh_process_equal"] = bool(r and procs and all(
        x and _start_ok(x["start"], ref.get(x["t"]), render)
        and traces_equal(x["load"], r["points"][str(x["t"])][0]["load"], render) for x in procs))
    g["R_pass"] = all(g[k] for k in ("G0a", "G0c", "R_start_equal", "R_load_deterministic",
                                     "R_fresh_process_equal", "control_off_by_one_detected",
                                     "control_perturb_detected"))

    f = get("f")
    forks = (f or {}).get("forks", [])
    g["F_parent_unchanged"] = bool(f and _replay_ok(f, render) and forks
                                   and all(x["post"] == x["pre"] for x in forks))
    g["F_child_start_equal"] = bool(f and forks and all(x["status"] == "ok" and x["start"] == x["pre"] == ref.get(x["t"])
                                                        for x in forks))
    by_t = collections.defaultdict(list)
    for x in forks:
        by_t[x["t"]].append(x)
    doubles = [xs for xs in by_t.values() if len(xs) > 1]
    g["F_load_deterministic"] = bool(doubles and all(traces_equal(xs[0]["load"], xs[1]["load"], render)
                                                     for xs in doubles))
    g["F_fork_failures"] = sorted({x["status"] for x in forks if x["status"] != "ok"})
    g["F_agrees_with_R"] = bool(r and forks and all(
        traces_equal(by_t[int(t)][0]["load"], xs[0]["load"], render) for t, xs in r["points"].items() if int(t) in by_t))
    g["F_pass"] = all(g[k] for k in ("G0a", "G0c", "F_parent_unchanged", "F_child_start_equal",
                                     "F_load_deterministic", "control_off_by_one_detected",
                                     "control_perturb_detected"))
    f_l0 = get("f_l0")
    l0_forks = (f_l0 or {}).get("forks", [])
    g["F_noop_fork_safe"] = bool(f_l0 and _replay_ok(f_l0, render) and l0_forks and all(
        x["status"] == "ok" and x["start"] == x["pre"] == x["post"] for x in l0_forks))
    f0 = get("f0")
    g["F0_headless_ok"] = bool(f0 and f0["ok"])

    for name in ("s_L0", "s_L1"):
        s = get(name)
        g[f"S_{name[2:]}_restored"] = bool(s and s["events"] and all(
            e["sim_equal"] and e["python_equal"] and e["rng_equal"] for e in s["events"]))
        g[f"S_{name[2:]}_replay_equal"] = bool(s and _replay_ok(s, render))
    s_nopy = get("s_nopy")
    g["control_s_without_python_detected"] = bool(s_nopy and s_nopy["events"] and not _replay_ok(s_nopy, False))
    g["S_pass"] = all(g[k] for k in ("G0a", "G0c", "S_L0_restored", "S_L0_replay_equal", "S_L1_restored",
                                     "S_L1_replay_equal", "control_s_without_python_detected",
                                     "control_perturb_detected"))
    return g


def evaluate_overall(per_episode):
    every = lambda key: bool(per_episode) and all(e[key] for e in per_episode.values())
    out = {k: every(k) for k in ("G0a", "G0b", "G0c", "R_pass", "F_pass", "S_pass")}
    out["step1_passed"] = out["G0a"] and out["G0c"] and (out["R_pass"] or out["F_pass"])
    out["recommended_method"] = ("F" if out["F_pass"] else "R" if out["R_pass"] else None) if out["step1_passed"] else None
    if not out["G0a"] or not out["G0c"]:
        out["stop_reason"] = "reference replay is not reproducible (G0a/G0c)"
    elif not out["step1_passed"]:
        out["stop_reason"] = "neither R nor F passed on every episode"
    else:
        out["stop_reason"] = None
    return out
