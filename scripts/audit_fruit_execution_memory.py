#!/usr/bin/env python3
"""Reuse the formal selection audit and check every ingested frame at execution cadence N."""
from __future__ import annotations

import argparse
from pathlib import Path

from run_teacher_oracle import selection_audit
from v_a_common import ECM, atomic_json, read_json, read_jsonl


def audit(directory, horizon, oracle, allow_control_abort=False):
    result = read_json(directory / "result.json")
    result.update(attempt_directory=str(directory), model_seed=7,
                  oracle=read_jsonl(directory / "oracle_decisions.jsonl"),
                  block=f"{result['task']}/ep{result['episode_id']}")
    formal = selection_audit([result], oracle["capacity"], oracle.get("event"), oracle["keyframes"])
    raw = read_jsonl(directory / "raw_frames.jsonl")
    client = read_jsonl(directory / "client_ingest.jsonl")
    server = read_jsonl(directory / "policy_ingest.jsonl")
    decisions = read_jsonl(directory / "policy_decisions.jsonl")
    sent = read_jsonl(directory / "client_decisions.jsonl")
    received = read_jsonl(directory / "policy_actions.jsonl")
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    engineering_error = result.get("outcome") not in ("success", "fail", "timeout") or bool(result.get("error"))
    expected_abort = (allow_control_abort and result.get("joint_guard_infeasible", False)
                      and "joint guard has no feasible envelope/step intersection" in result.get("error", ""))
    check(not engineering_error or expected_abort, "episode_engineering_error")
    check(result.get("obs_horizon") == 16 and result.get("exec_horizon") == horizon, "observation_or_execution_horizon")
    check(bool(decisions) and len(client) == len(server) == len(decisions) == len(sent) == len(received), "decision_record_counts")
    cursor = 0
    for index, (c, s, d, tx, rx) in enumerate(zip(client, server, decisions, sent, received)):
        maximum = index * horizon
        check(c["first_frame_id"] == s["first_frame_id"] == cursor, f"ingest_start/{index}")
        check(c["last_frame_id"] == s["last_frame_id"] == maximum == d["available_max_frame_id"], f"ingest_end/{index}")
        check(c["samples"] == s["samples"] == (1 if index == 0 else horizon), f"ingest_samples/{index}")
        check(d["history_count"] == maximum+1 and d["control_step"] == maximum, f"history_frame_clock/{index}")
        check(tx["input_sha256"] == rx["input_sha256"] and tx["actions_sha256"] == rx["actions_sha256"], f"websocket_hashes/{index}")
        for frame in s["frames"]:
            i = frame["frame_id"]
            check(i < len(raw) and frame["front_sha256"] == raw[i]["front_sha256"]
                  and frame["state_sha256"] == raw[i]["state_sha256"], f"raw_ingest_hash/{i}")
        selected = d["selected_frame_ids"]
        check(d["source_position_frame_ids"] == selected, f"source_positions/{index}")
        check(d["selected_front_sha256"] == [raw[i]["front_sha256"] for i in selected], f"selected_source_hash/{index}")
        check(d["mask"] == [True] * d["valid_tokens"] + [False] * d["padding_tokens"], f"padding_mask/{index}")
        cursor = maximum+1
    check(formal["decisions"] > 0 and formal["mismatches"] == 0, "formal_selection_audit")
    return {"directory": str(directory), "exec_horizon": horizon, "formal_selection_audit": formal,
            "ingested_frames": cursor, "decisions": len(decisions), "mismatches": len(errors),
            "errors": errors, "passed": not errors, "episode_engineering_error": bool(engineering_error),
            "expected_control_abort": bool(expected_abort)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--exec-horizon", type=int, required=True)
    a = p.parse_args()
    oracle = read_json(ECM / "results/fruit_dart_kf_eval_20261027/protocol.json")["params"]["oracle"]
    result = audit(a.directory, a.exec_horizon, oracle)
    atomic_json(a.directory / "memory_audit.json", result)
    print(result)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
