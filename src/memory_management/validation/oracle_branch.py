"""Closed-loop Teacher Oracle, step 3: the expert chunk at a live decision (simulator side).

Protocol: reports/teacher_oracle_pilot_protocol_20261024.md. A persistent branch worker process
answers one query per decision: it builds a Reference from what the live episode has written so
far (raw_frames.jsonl, control_actions.jsonl and the policy's saved action chunks), replays a fresh
env instance to the decision (method R, step 1), checks the start frames bitwise, and runs the
step-2 expert (planner_branch.run_expert, mode "full") for at most `chunk_len` steps.

The live episode's client attaches the reply to the WebSocket observation as teacher_* fields.
Only numpy is imported at module level; sim-dependent code imports lazily.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import time
import traceback

import numpy as np

from . import planner_branch as pb
from . import sim_branch as sb


def live_reference(directory, task, episode_id, model_seed, environment_seed, t):
    """The live episode after t executed steps, cross-checked like sim_branch.load_reference."""
    directory = Path(directory)
    frames = sb._read_jsonl(directory / "raw_frames.jsonl")
    # The recorder creates control_actions.jsonl at the first executed step.
    controls = sb._read_jsonl(directory / "control_actions.jsonl") if t else []
    if not frames:
        raise sb.ReferenceMismatch("no recorded frames")
    exec_start = int(frames[0]["exec_start_idx"])
    if len(controls) != t:
        raise sb.ReferenceMismatch(f"{len(controls)} control rows at t={t}")
    chunks = [np.load(directory / "actions" / f"decision_{d:04d}.npz")["actions"]
              for d in range(math.ceil(t / sb.EXEC_HORIZON))]
    actions = np.concatenate([c[:sb.EXEC_HORIZON] for c in chunks])[:t] if chunks else np.zeros((0, 8))
    for i, row in enumerate(controls):
        if row["control_step"] != i + 1 or not np.array_equal(np.asarray(row["action"], dtype=actions.dtype), actions[i]):
            raise sb.ReferenceMismatch(f"control step {i + 1} differs from the saved action chunk")
    if len(frames) != exec_start + t + 1:
        raise sb.ReferenceMismatch(f"{len(frames)} frames for exec_start={exec_start}, t={t}")
    for i, row in enumerate(frames):
        if row["frame_id"] != i or row["control_step"] != i - exec_start:
            raise sb.ReferenceMismatch(f"frame {i} is out of order")
    return sb.Reference(directory, task, int(episode_id), int(model_seed), -1, int(environment_seed), exec_start,
                        "live", actions, frames, controls)


def expert_params(params, t):
    """Step-2 expert parameters; the step budget ends the run after chunk_len steps."""
    return {**params, "max_steps": min(params["max_steps"], t + params["chunk_len"] - 1)}


def payload(rec, chunk_len):
    """teacher_* fields for a branch result; actions are padded to chunk_len with the last action."""
    expert = rec.get("expert")
    if not rec.get("start_ok"):
        status = "error" if rec.get("error") else "start_mismatch"
    elif not pb.chunk_available(expert, chunk_len) or not expert["chunk"]:
        status = f"unavailable:{(expert or {}).get('outcome')}"
    else:
        # demo_source.action_chunk's padding rule; the stick expert has 7 dims, so not that function.
        valid = np.asarray(expert["chunk"], dtype=np.float32)[:chunk_len]
        actions = np.concatenate([valid, np.repeat(valid[-1:], chunk_len - len(valid), axis=0)])
        return {"teacher_status": "ok", "teacher_actions": actions, "teacher_valid_steps": len(valid)}
    return {"teacher_status": status, "teacher_actions": np.zeros((chunk_len, 8), np.float32), "teacher_valid_steps": 0}


def branch(info, t, scratch, params):
    out = {"t": t, "start_ok": False}
    rp = None
    try:
        start = time.monotonic()
        ref = live_reference(info["directory"], info["task"], info["episode_id"], info["model_seed"],
                             info["environment_seed"], t)
        rp = sb.Replayer(ref, scratch).reset()
        reached = rp.advance_to(t)
        frames = sb.compare_frames(ref.frames, rp.rows)
        expected = ref.exec_start + t + 1
        out["start"] = {"reached": reached, "rows": len(rp.rows), "expected_rows": expected, **frames,
                        "replay_seconds": round(time.monotonic() - start, 3)}
        out["start_ok"] = bool(reached and len(rp.rows) == expected and frames["physics_mismatches"] == 0
                               and frames["render_mismatches"] == 0)
        if out["start_ok"]:
            run = pb.run_expert(rp, ref, t, "full", expert_params(params, t))
            out["expert"] = run.record(params["chunk_len"])
    except Exception:
        out["error"] = traceback.format_exc()[-3000:]
    finally:
        if rp is not None:
            try:
                rp.close()
            except Exception:
                out["close_error"] = traceback.format_exc()[-1000:]
    return out


def serve(info, params, scratch_root, requests, replies):
    """Worker loop: one JSON request {"t": t} per line; one JSON reply line per request."""
    for line in requests:
        if not line.strip():
            continue
        t = int(json.loads(line)["t"])
        start = time.monotonic()
        scratch = Path(scratch_root) / f"t{t:04d}"
        rec = branch(info, t, scratch, params)
        shutil.rmtree(scratch, ignore_errors=True)
        rec["seconds"] = round(time.monotonic() - start, 3)
        replies.write(json.dumps(rec) + "\n")
        replies.flush()


class BranchClient:
    """The live episode's handle on its branch worker; a hung worker is killed and restarted."""

    def __init__(self, command, env, log_path, timeout_s):
        self.command, self.env, self.log_path, self.timeout_s = command, env, Path(log_path), timeout_s
        self.proc = self.reply = None

    def start(self):
        read, write = os.pipe()
        log = self.log_path.open("a", buffering=1)
        self.proc = subprocess.Popen(self.command + ["--reply-fd", str(write)], stdin=subprocess.PIPE, stdout=log,
                                     stderr=subprocess.STDOUT, env=self.env, pass_fds=(write,), text=True,
                                     start_new_session=True)
        os.close(write)
        log.close()
        self.reply = os.fdopen(read, "r")

    def query(self, t):
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        try:
            self.proc.stdin.write(json.dumps({"t": t}) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            self.stop()
            return {"t": t, "start_ok": False, "error": "branch worker exited"}
        ready, _, _ = select.select([self.reply], [], [], self.timeout_s)
        line = self.reply.readline() if ready else ""
        if not line:
            self.stop()
            return {"t": t, "start_ok": False, "error": "branch worker timeout" if not ready else "branch worker exited"}
        return json.loads(line)

    def stop(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, 9)
            except ProcessLookupError:
                pass
            self.proc.wait()
        if self.reply is not None:
            self.reply.close()
        self.proc = self.reply = None


def install_oracle_client(official_eval, recorder, client, log_path, chunk_len):
    """Wrap the (already logging) client so every infer carries the decision's teacher_* fields.

    client is a BranchClient, or None for conditions without an expert (status "not_requested").
    """
    base = official_eval._websocket_client_policy.MMEVLAWebsocketClientPolicy

    class OracleClient(base):
        def infer(self, obs):
            t = recorder.control_step
            decision = self.decision_idx
            if t != decision * sb.EXEC_HORIZON:
                raise RuntimeError(f"decision {decision} at t={t}")
            if client is None:
                fields = {"teacher_status": "not_requested", "teacher_actions": np.zeros((chunk_len, 8), np.float32),
                          "teacher_valid_steps": 0}
                rec = None
            else:
                rec = client.query(t)
                fields = payload(rec, chunk_len)
            if rec is not None:
                with Path(log_path).open("a") as f:
                    f.write(json.dumps({"decision": decision, "t": t, "status": fields["teacher_status"],
                                        "valid_steps": fields["teacher_valid_steps"], **rec}) + "\n")
            return super().infer({**obs, "teacher_decision": decision, **fields})

    official_eval._websocket_client_policy.MMEVLAWebsocketClientPolicy = OracleClient
