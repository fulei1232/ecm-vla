#!/usr/bin/env python3
"""First-pick phase substitution on a development episode.

The scripted expert handles every phase except the selected one, which uses
the DART-trained K0 policy with its ordinary task prompt and 16-step chunks.
This is a diagnostic intervention, not a deployable policy or a formal result.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from v_a_common import ECM, POLICY, SIM, atomic_json

PHASES = ("reach", "approach", "grasp", "pull", "carry", "release")
MODES = ("expert", *PHASES, "all", "reach_long", "carry_long", "reach_repair",
         "carry_guard", "carry_repair", "reach_carry_repair", "reach_carry_safe",
         "reach_carry_grasp_safe", "reach_collect", "carry_collect", "carry_collect_late",
         "reach_collect_auto", "carry_collect_auto", "grasp_collect_auto", "full_collect")


class EpisodeStopped(Exception):
    def __init__(self, outcome: str):
        self.outcome = outcome


class SafetyFallback(Exception):
    pass


class AutoCorrection(Exception):
    def __init__(self, reason: str):
        self.reason = reason


def use_policy(mode: str, phase: str) -> bool:
    return phase in PHASES and (mode == phase or mode == f"{phase}_long" or mode == "all" or
                                (mode == "reach_repair" and phase == "reach") or
                                (mode in ("carry_guard", "carry_repair") and phase == "carry") or
                                (mode in ("reach_carry_repair", "reach_carry_safe", "reach_carry_grasp_safe") and
                                 phase in ("reach", "carry")) or
                                (mode in ("reach_collect", "reach_collect_auto", "full_collect") and phase == "reach") or
                                (mode in ("carry_collect", "carry_collect_late", "carry_collect_auto", "full_collect") and phase == "carry") or
                                (mode in ("grasp_collect_auto", "full_collect") and phase == "grasp"))


def run(args):
    sys.path[:0] = [str(POLICY / "examples/robomme"), str(POLICY / "packages/openpi-client/src"),
                    str(SIM / "src"), str(ECM), str(ECM / "src")]
    import eval as official_eval
    from fruit_picking.solver import FruitPickSolver, PlanningFailure
    from memory_management.fruit_eval import FruitEnvRunner, FruitRawRecorder, snapshot

    args.attempt_dir.mkdir(parents=True, exist_ok=True)
    row = {"task": args.task, "episode_id": args.episode_id, "mode": args.mode,
           "condition": "K0", "outcome": "setup_error", "error": ""}
    runner = recorder = None
    action_trace = []
    start = time.monotonic()
    try:
        max_steps = 2000 if args.mode == "full_collect" else 1300
        runner = FruitEnvRunner(args.task, args.root, max_steps=max_steps)
        man = runner.manifest(args.episode_id)
        row.update(environment_seed=man["seed"], n_pick=man["n_pick"], demo_steps=man["steps"])
        runner.make_env(args.episode_id)
        recorder = FruitRawRecorder(args.attempt_dir, args.task)
        recorder.install(runner)
        recorder.install_client(official_eval)

        initial = runner.get_init_obs()
        state = official_eval.EpisodeState()
        state.image_buffer.extend(initial["images"])
        state.wrist_image_buffer.extend(initial["wrist_images"])
        state.state_buffer.extend(initial["states"])
        state.exec_start_idx = len(state.image_buffer) - 1
        current = (initial["images"][-1], initial["wrist_images"][-1], initial["states"][-1])

        context = {"run_id": args.attempt_dir.parent.parent.parent.name,
                   "task": args.task, "episode_id": args.episode_id,
                   "variant": "K0", "condition": "K0", "directory": str(args.attempt_dir),
                   "rs_schedule": None}
        args.context.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.context, context)
        client = None
        evaluator = None
        if args.mode != "expert":
            evaluator = official_eval.EpisodeEvaluator(
                official_eval.Args(host="127.0.0.1", port=args.port, model_seed=7,
                                   model_ckpt_id=79999, policy_name="perceptual-framesamp-modul"),
                args.attempt_dir)
            client = official_eval._websocket_client_policy.MMEVLAWebsocketClientPolicy("127.0.0.1", args.port)
            response = client.reset()
            while not response.get("reset_finished", False):
                time.sleep(0.1)

        target = runner.env.unwrapped.layout.targets[0]
        row["target_index"] = int(target)
        pending = deque()
        previous_phase = None
        counts = Counter()
        first = {"near5": None, "near3": None, "detach": None, "basket": None, "drop": None}
        min_distance = float("inf")
        phase_best = {"reach": float("inf"), "carry": float("inf")}
        pick_counts = Counter()
        last_outcome = "ongoing"

        def step(expert_action, labels):
            nonlocal current, previous_phase, min_distance, last_outcome
            phase = labels["phase"]
            if phase != previous_phase:
                pending.clear()
                previous_phase = phase
            source = "policy" if use_policy(args.mode, phase) else "expert"
            if source == "policy":
                if not pending:
                    chunk = evaluator.get_action_chunk(client, state, *current,
                                                       initial["task_goal"], None, exec_horizon=16)
                    pending.extend(np.asarray(a) for a in chunk)
                    state.clear_buffers()
                    counts["policy_queries"] += 1
                action = pending.popleft()
            else:
                action = expert_action
            proposed = np.asarray(action).copy()
            if args.mode in ("reach_collect_auto", "carry_collect_auto", "grasp_collect_auto", "full_collect") and source == "policy":
                gt_before = snapshot(runner.env)
                policy_steps = pick_counts[f"{phase}_policy_steps"]
                if phase == "reach":
                    distance = float(np.linalg.norm(np.asarray(gt_before["tcp_position"]) -
                                                    np.asarray(gt_before["fruit_positions"][target])))
                    phase_best[phase] = min(phase_best[phase], distance)
                    reason = ("reach_diverging" if policy_steps >= 16 and
                              distance > phase_best[phase] + 0.10 else
                              "reach_budget" if policy_steps >= (64 if args.mode == "full_collect" else 96)
                              and distance > 0.14 else None)
                elif phase == "carry":
                    distance = float(np.linalg.norm(np.asarray(gt_before["tcp_position"]) - solver.home.p))
                    phase_best[phase] = min(phase_best[phase], distance)
                    fruit_separation = float(np.linalg.norm(np.asarray(gt_before["tcp_position"]) -
                                                               np.asarray(gt_before["fruit_positions"][target])))
                    reason = ("premature_open" if proposed[-1] > 0 and distance > 0.05 else
                              "fruit_separating" if fruit_separation > 0.015 else
                              "carry_diverging" if policy_steps >= 16 and distance > phase_best[phase] + 0.10 else
                              "carry_budget" if policy_steps >= (64 if args.mode == "full_collect" else 96)
                              and distance > 0.05 else None)
                else:
                    distance = float(np.linalg.norm(np.asarray(gt_before["tcp_position"]) -
                                                    np.asarray(gt_before["fruit_positions"][target])))
                    reason = "grasp_not_closing" if proposed[-1] > 0 else None
                if reason:
                    row["auto_correction_trigger"] = {"control_step": recorder.control_step + 1,
                                                       "phase": phase, "reason": reason,
                                                       "goal_distance_m": distance,
                                                       "phase_policy_steps": policy_steps}
                    raise AutoCorrection(reason)
            if args.mode in ("carry_guard", "carry_repair", "reach_carry_repair",
                             "reach_carry_safe", "reach_carry_grasp_safe",
                             "carry_collect", "carry_collect_late", "carry_collect_auto", "full_collect") and phase == "carry" and action[-1] > 0:
                action = action.copy()
                action[-1] = -1.0
                counts["carry_open_suppressed"] += 1
            if args.mode in ("reach_carry_safe", "reach_carry_grasp_safe") and phase == "carry" and source == "policy":
                if args.mode == "reach_carry_grasp_safe":
                    gt_before = snapshot(runner.env)
                    separation = float(np.linalg.norm(np.asarray(gt_before["tcp_position"]) -
                                                      np.asarray(gt_before["fruit_positions"][target])))
                    if separation > 0.015:
                        row["carry_safety_trigger"] = {"control_step": recorder.control_step + 1,
                                                        "reason": "fruit_tcp_separation",
                                                        "separation_m": separation}
                        raise SafetyFallback("fruit separating from gripper")
                target_qpos = solver.qpos().copy()
                target_qpos[:solver.n_arm] = action[:solver.n_arm]
                commanded_z = float(solver.fk(target_qpos).p[2])
                if commanded_z < 0.20:
                    row["carry_safety_trigger"] = {"control_step": recorder.control_step + 1,
                                                    "reason": "low_commanded_tcp",
                                                    "commanded_tcp_z_m": commanded_z}
                    raise SafetyFallback("carry target below 20 cm")
            current, stop, last_outcome = runner.step(action)
            counts[f"{phase}_{source}_steps"] += 1
            pick_counts[f"{phase}_{source}_steps"] += 1
            t = recorder.control_step
            action_trace.append({"control_step": t, "pre_frame_id": t - 1, "phase": phase,
                                 "source": source, "labels": labels,
                                 "executed_action": np.asarray(action).tolist(),
                                 "policy_action": proposed.tolist() if source == "policy" else None})
            gt = snapshot(runner.env)
            d = gt["fruit_distance"]
            min_distance = min(min_distance, d)
            for key, event in (("near5", d <= 0.05), ("near3", d <= 0.03),
                               ("detach", gt["detached"][target]),
                               ("basket", gt["in_basket"][target]),
                               ("drop", gt["detached"][target] and not gt["in_basket"][target]
                                and gt["fruit_positions"][target][2] < 0.06)):
                if event and first[key] is None:
                    first[key] = t
            if current[0] is not None:
                state.add_observation(*current)
            state.count += 1
            if stop or t > max_steps:
                raise EpisodeStopped(last_outcome if stop else "timeout")
            return current, None

        solver = FruitPickSolver(runner.env, step)
        if args.mode in ("grasp_collect_auto", "full_collect"):
            original_hold = solver.hold

            def grasp_hold(n, labels):
                if labels["phase"] != "grasp":
                    return original_hold(n, labels)
                try:
                    original_hold(n, labels)
                except AutoCorrection:
                    original_hold(10, {**labels, "phase": "grasp_repair"})
                    return
                if not runner.env.unwrapped.attached[target]:
                    row["auto_correction_trigger"] = {"control_step": recorder.control_step + 1,
                                                       "phase": "grasp", "reason": "grasp_failed_after_10_steps",
                                                       "phase_policy_steps": counts["grasp_policy_steps"]}
                    original_hold(10, {**labels, "phase": "grasp_repair"})

            solver.hold = grasp_hold
        extended_phases = {"reach_long": ("reach",), "reach_repair": ("reach",),
                           "reach_collect": ("reach",), "reach_collect_auto": ("reach",),
                           "carry_collect": ("carry",), "carry_collect_auto": ("carry",),
                           "full_collect": ("reach", "carry"),
                           "carry_collect_late": ("carry",),
                           "carry_long": ("carry",), "carry_guard": ("carry",),
                           "carry_repair": ("carry",),
                           "reach_carry_repair": ("reach", "carry"),
                           "reach_carry_safe": ("reach", "carry"),
                           "reach_carry_grasp_safe": ("reach", "carry")}.get(args.mode, ())
        if extended_phases:
            original_move = solver.move

            def extended_move(pose, labels, screw=False, attach=False):
                phase = labels["phase"]
                if phase not in extended_phases:
                    return original_move(pose, labels, screw=screw, attach=attach)
                if phase == "reach":
                    def goal_distance():
                        gt = snapshot(runner.env)
                        return float(np.linalg.norm(np.asarray(gt["tcp_position"]) -
                                                    np.asarray(gt["fruit_positions"][target])))
                    tolerance, budget = 0.14, (64 if args.mode == "full_collect" else
                                               96 if args.mode in ("reach_collect", "reach_collect_auto") else 240)
                else:
                    def goal_distance():
                        return float(np.linalg.norm(np.asarray(snapshot(runner.env)["tcp_position"]) - solver.home.p))
                    tolerance, budget = 0.05, (64 if args.mode == "full_collect" else
                                               96 if args.mode in ("carry_collect_late", "carry_collect_auto") else
                                               48 if args.mode == "carry_collect" else 240)
                try:
                    original_move(pose, labels, screw=screw, attach=attach)
                    while (args.mode in ("reach_collect", "carry_collect", "carry_collect_late") or
                           goal_distance() > tolerance) and \
                            pick_counts[f"{phase}_policy_steps"] < budget:
                        step(solver.action(solver.qpos()), labels)
                except AutoCorrection:
                    original_move(pose, {**labels, "phase": f"{phase}_repair"},
                                  screw=screw, attach=attach)
                    row[f"{phase}_repair_distance_m"] = goal_distance()
                    return
                except SafetyFallback:
                    if phase != "carry" or args.mode not in ("reach_carry_safe", "reach_carry_grasp_safe"):
                        raise
                    original_move(pose, {**labels, "phase": "carry_repair"},
                                  screw=screw, attach=attach)
                    row["carry_repair_distance_m"] = goal_distance()
                    return
                row[f"{phase}_goal_distance_m"] = goal_distance()
                row[f"{phase}_goal_reached"] = goal_distance() <= tolerance
                if args.mode in ("reach_collect_auto", "carry_collect_auto", "full_collect") and not row[f"{phase}_goal_reached"]:
                    row["auto_correction_trigger"] = {"control_step": recorder.control_step + 1,
                                                       "phase": phase, "reason": f"{phase}_budget",
                                                       "goal_distance_m": goal_distance(),
                                                       "phase_policy_steps": pick_counts[f"{phase}_policy_steps"]}
                if phase == "reach" and (args.mode in ("reach_repair", "reach_collect", "reach_carry_repair",
                                                        "reach_carry_safe", "reach_carry_grasp_safe") or
                                         (args.mode in ("reach_collect_auto", "full_collect") and
                                          not row[f"{phase}_goal_reached"])):
                    original_move(pose, {**labels, "phase": "reach_repair"}, screw=screw, attach=attach)
                    row["reach_repair_distance_m"] = goal_distance()
                if phase == "carry" and (not row["carry_goal_reached"] or
                                         args.mode in ("carry_collect", "carry_collect_late")):
                    if args.mode == "carry_guard":
                        raise PlanningFailure("carry goal not reached before release")
                    if args.mode in ("carry_repair", "carry_collect", "carry_collect_late", "carry_collect_auto",
                                     "full_collect", "reach_carry_repair",
                                     "reach_carry_safe", "reach_carry_grasp_safe"):
                        original_move(pose, {**labels, "phase": "carry_repair"},
                                      screw=screw, attach=attach)
                        row["carry_repair_distance_m"] = goal_distance()

            solver.move = extended_move
        row["outcome"] = "incomplete"
        try:
            solver.hold(runner.env.unwrapped.layout.observe_steps,
                        {"subgoal": "observe the tree", "target": None, "phase": "observe"})
            if args.mode == "full_collect":
                targets = list(runner.env.unwrapped.layout.targets)
                for pick_index, target in enumerate(targets):
                    pick_counts = Counter()
                    phase_best = {"reach": float("inf"), "carry": float("inf")}
                    solver.pick(target, pick_index, len(targets))
                row["outcome"] = "success" if bool(runner.env.unwrapped.evaluate()["success"][0]) else "incomplete"
            else:
                solver.pick(target, 0, len(runner.env.unwrapped.layout.targets))
                row["outcome"] = "first_pick_complete"
        except PlanningFailure as exc:
            row.update(outcome="planning_failure", reason=str(exc))
        except EpisodeStopped as exc:
            row.update(outcome=exc.outcome, reason="environment stopped")
        gt = snapshot(runner.env)
        targets = list(runner.env.unwrapped.layout.targets)
        row.update(first=first, phase_steps=dict(counts), min_distance_m=min_distance,
                   first_pick_success=bool(gt["in_basket"][targets[0]]),
                   task_success=bool(runner.env.unwrapped.evaluate()["success"][0]),
                   target_baskets=sum(bool(gt["in_basket"][i]) for i in targets),
                   wrong_baskets=sum(bool(v) for i, v in enumerate(gt["in_basket"]) if i not in targets),
                   target_detached=bool(gt["detached"][target]),
                   target_in_basket=bool(gt["in_basket"][target]),
                   final_detached=sum(gt["detached"]), final_in_basket=gt["n_in_basket"],
                   executed_steps=recorder.control_step)
    except BaseException:
        row.update(outcome="error", error=traceback.format_exc())
    finally:
        if recorder is not None:
            recorder.flush()
        if runner is not None:
            try:
                runner.close_env()
            except BaseException:
                row["error"] += "\nCleanup error:\n" + traceback.format_exc()
                row["outcome"] = "error"
        atomic_json(args.attempt_dir / "phase_trace.json", action_trace)
        row["elapsed_seconds"] = round(time.monotonic() - start, 3)
        atomic_json(args.attempt_dir / "result.json", row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--task", choices=("FruitCount", "FruitOccludedRipe"), required=True)
    p.add_argument("--episode-id", type=int, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--attempt-dir", type=Path, required=True)
    args = p.parse_args()
    row = run(args)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0 if row["outcome"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
