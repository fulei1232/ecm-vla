#!/usr/bin/env python3
"""Replay a failed first pick exactly, then hand the held fruit to the expert.

This is a simulator counterfactual. The cutoff is a control step from the saved
trajectory; replay alignment is checked before the new carry plan is executed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from v_a_common import ECM, POLICY, SIM, atomic_json


class EpisodeStopped(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cutoff", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path[:0] = [str(POLICY / "examples/robomme"), str(POLICY / "packages/openpi-client/src"),
                    str(SIM / "src"), str(ECM), str(ECM / "src")]
    from fruit_picking.solver import CLOSED, OPEN, FruitPickSolver
    from memory_management.fruit_eval import FruitEnvRunner, snapshot

    a.out.mkdir(parents=True, exist_ok=True)
    source = json.loads((a.source / "result.json").read_text())
    actions = [json.loads(s)["action"] for s in (a.source / "control_actions.jsonl").read_text().splitlines()]
    references = {r["control_step"]: r["gt"] for r in map(json.loads,
                    (a.source / "raw_frames.jsonl").read_text().splitlines())}
    row = {"task": source["task"], "episode_id": source["episode_id"],
           "target_index": source["target_index"], "cutoff": a.cutoff,
           "source": str(a.source.resolve()), "outcome": "error", "error": ""}
    runner = None
    try:
        runner = FruitEnvRunner(source["task"], a.root, max_steps=1300)
        runner.make_env(source["episode_id"])
        runner.get_init_obs()
        count = 0

        def step(action, labels):
            nonlocal count
            _, stop, outcome = runner.step(action)
            count += 1
            if stop:
                raise EpisodeStopped(f"{labels['phase']} stopped at {count}: {outcome}")
            return None, None

        solver = FruitPickSolver(runner.env, step)
        for action in actions[:a.cutoff]:
            step(action, {"phase": "replay"})
        gt = snapshot(runner.env)
        ref = references[a.cutoff]
        i = source["target_index"]
        tcp_err = float(np.linalg.norm(np.asarray(gt["tcp_position"]) - np.asarray(ref["tcp_position"])))
        fruit_err = float(np.linalg.norm(np.asarray(gt["fruit_positions"][i]) -
                                         np.asarray(ref["fruit_positions"][i])))
        row.update(replay_tcp_error_m=tcp_err, replay_fruit_error_m=fruit_err,
                   separation_at_cutoff_m=float(np.linalg.norm(np.asarray(gt["tcp_position"]) -
                                                              np.asarray(gt["fruit_positions"][i]))))
        if max(tcp_err, fruit_err) > 0.001:
            raise ValueError("replay diverged from recorded state by more than 1 mm")
        if not gt["detached"][i]:
            raise ValueError("target is still attached to tree at cutoff")
        solver.gripper = CLOSED
        solver.refresh_obstacles(skip=(i,))
        solver.planner.update_attached_sphere(runner.env.unwrapped.layout.fruits[i]["radius"],
                                               [0, 0, 0, 1.0, 0, 0, 0])
        try:
            solver.move(solver.home, {"subgoal": "carry recovery to basket", "phase": "carry_repair"}, attach=True)
            solver.gripper = OPEN
            solver.hold(8, {"subgoal": "release in basket", "phase": "release"})
            solver.hold(12, {"subgoal": "settle", "phase": "settle"})
            row["outcome"] = "first_pick_complete" if runner.env.unwrapped.in_basket()[i] else "missed_basket"
        except EpisodeStopped as exc:
            row.update(outcome="stopped", reason=str(exc))
        end = snapshot(runner.env)
        row.update(first_pick_success=bool(end["in_basket"][i]),
                   target_detached=bool(end["detached"][i]),
                   final_fruit_position=end["fruit_positions"][i],
                   final_tcp_position=end["tcp_position"],
                   executed_steps=count)
    except BaseException:
        row["error"] = traceback.format_exc()
    finally:
        if runner is not None:
            runner.close_env()
        atomic_json(a.out / "result.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0 if not row["error"] else 1


if __name__ == "__main__":
    sys.exit(main())
