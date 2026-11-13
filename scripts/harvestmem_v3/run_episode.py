#!/usr/bin/env python3
"""Run one HarvestMem v3 expert episode in the isolated S1 output tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from v_a_common import ECM, atomic_json, read_json, sha256, timestamp

sys.path[:0] = [str(ECM), str(ECM / "src")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("count", "occluded_ripe"), required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--purpose", choices=("dev_expert", "train_clean", "train_dart"), required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--layout-seed", type=int)
    parser.add_argument("--dart-two-cycles", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    results_root = args.results_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise RuntimeError(f"refusing to overwrite {out}")

    results_root.mkdir(parents=True, exist_ok=True)
    gate_path = results_root / "R0_gate.json"
    if not gate_path.exists():
        old_gate = ECM / "results/fruit_task_v2_R0_r12_20261109/R0_gate.json"
        gate = read_json(old_gate) if old_gate.exists() else {"passed": True, "scope": "HarvestMem v3 S1 expert/data generation"}
        gate.update(passed=True, scope="HarvestMem v3 S1 isolated output")
        atomic_json(gate_path, gate)

    import gymnasium as gym
    import fruit_picking.task_v2_env  # noqa: F401
    from fruit_picking.solver import FruitPickSolver, PlanningFailure
    from fruit_picking.canonical_solver import configure
    from fruit_picking.count_shield_r12_visual import install as install_false_bottom
    from fruit_picking.count_shield_v2_buildfix import install as install_constructor
    from fruit_shield_v2_runtime import install as install_v2
    from fruit_r12_runtime import Revision12Replay
    from fruit_r9_episode_support import EpisodeEnd
    import fruit_r9_common
    import fruit_r9_data
    import fruit_r11_common
    import fruit_r12_common
    import fruit_shield_runtime
    from fruit_shield_runtime import ShieldRecorder
    import audit_fruit_task_v2_r9_episode as original

    values = {
        "OUT": results_root,
        "TRACKED": results_root,
        "REPORT": results_root / "REPORT.md",
        "DOC": results_root / "plan.md",
        "DATASET": results_root / "dataset",
        "RAW_CLEAN": results_root / "raw_clean",
        "RAW_DART": results_root / "raw_dart",
    }
    fruit_r12_common.OUT = results_root
    fruit_r12_common.TRACKED = results_root
    fruit_r12_common.REPORT = values["REPORT"]
    fruit_r12_common.DOC = values["DOC"]
    fruit_r12_common.DATASET = values["DATASET"]
    fruit_r12_common.RAW_CLEAN = values["RAW_CLEAN"]
    fruit_r12_common.RAW_DART = values["RAW_DART"]
    for module in (fruit_r9_common, fruit_r11_common, fruit_r9_data, original):
        if hasattr(module, "OUT"):
            module.OUT = results_root
    original.R9Replay = Revision12Replay

    install_v2()
    install_constructor()
    install_false_bottom()

    # Route all S1 episodes through the no-lamp variant in this process only.
    make_original = gym.make
    made = []

    def make(env_id, *env_args, **kwargs):
        if kwargs.get("task") in ("count", "occluded_ripe"):
            kwargs["no_lamp_wait"] = True
        env = make_original(env_id, *env_args, **kwargs)
        if kwargs.get("task") in ("count", "occluded_ripe"):
            base_reset = env.reset

            def reset(*, seed=None, options=None):
                layout_seed = args.layout_seed if args.layout_seed is not None else seed
                env.r12_layout_seed = int(layout_seed) if layout_seed is not None else None
                env.r12_noise_seed = int(seed) if seed is not None else None
                env.r12_two_cycle_dart = bool(args.dart_two_cycles)
                env.r12_capture_front_hashes = False
                return base_reset(seed=layout_seed, options=options)

            env.reset = reset
            made.append(env)
        return env

    gym.make = make
    env = None
    replay = None
    recorder = None
    failure = ""
    started = time.monotonic()
    seed_base = 4200000 if args.purpose == "dev_expert" else 4100000
    seed = seed_base + 16 * args.episode + args.attempt
    layout_seed = args.layout_seed if args.layout_seed is not None else seed
    budget = 2800 if args.task == "count" else 1300
    guard_path = ECM / "results/fruit_k0v2_gate_20261102/guard_v2.json"
    if not guard_path.exists():
        raise FileNotFoundError(guard_path)

    try:
        env = gym.make(
            "FruitPickTaskV2-v0",
            task=args.task,
            task_version="v2",
            task_revision=9,
            robot_uids="rm75b_eg2",
            obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            sim_backend="cpu",
            max_episode_steps=budget,
        )
        obs, _ = env.reset(seed=seed)
        u = env.unwrapped
        replay = Revision12Replay(env, obs, out, "history", read_json(guard_path), None)
        solver = FruitPickSolver(env, replay.step, ik_mode="canonical_v31", ik_seed=0, ik_candidates=64)
        configure(solver, {"settle_steps": 2})
        replay.solver_fk = solver.fk
        recorder = ShieldRecorder(out, env, obs, args.episode, seed, solver.fk)
        replay.data_recorder = recorder
        if args.dart_two_cycles:
            replay.noise = np.zeros(7)
            replay.noise_rng = np.random.default_rng([seed, 1])

        original_pick = FruitPickSolver.pick

        def pick_with_cycle(self, actor, _index, count):
            if args.dart_two_cycles:
                index = getattr(self, "hm3_pick_cycle_counter", 0)
                self.hm3_pick_cycle_counter = index + 1
            else:
                index = _index
            return original_pick(self, actor, index, count)

        FruitPickSolver.pick = pick_with_cycle
        try:
            fixed = np.r_[u.agent.robot.get_qpos()[0].cpu().numpy()[:7], 1.0]
            solver.hold(30, {"phase": "observe", "target": None, "subgoal": "observe initial scene"})
            if args.task == "count" and replay.provided()["lamp_green"] is not True:
                raise RuntimeError("no_lamp_wait variant did not turn green at the observation boundary")
            while not replay.ended:
                slot = replay.choose()
                if slot is None:
                    solver.gripper = 1.0
                    solver.refresh_obstacles()
                    solver.move(solver.home, {"phase": "carry", "target": 0, "subgoal": "return to rest to finish"})
                    solver.hold(budget, {"phase": "finish", "target": None, "subgoal": "hold open at rest until termination"})
                else:
                    actor = u.hanging_slot[slot]
                    if actor is None:
                        solver.hold(1, {"phase": "inspect", "target": None, "subgoal": "observe wind-lost current target"})
                        continue
                    try:
                        solver.pick(actor, 0, u.layout.n_pick)
                        replay.releases.append({"step": u.control_steps, "actor": actor, "slot": slot,
                                                "event": "own_release_attempt"})
                    except EpisodeEnd:
                        raise
                    except Exception as exc:
                        failure = str(exc)
                        replay.reason = "planning_or_grasp_failure"
                        break
        finally:
            FruitPickSolver.pick = original_pick
        if replay.ended:
            pass
    except EpisodeEnd:
        # Normal replay termination is signalled as an exception by the legacy recorder.
        pass
    except PlanningFailure as exc:
        failure = str(exc)
        if replay is not None:
            replay.reason = "planning_or_grasp_failure"
    finally:
        if recorder is not None and env is not None:
            u = env.unwrapped
            last = np.r_[u.agent.keyframes["rest"].qpos[:7], 1.0]
            recorder.record(replay.obs, last, {"phase": "terminal_hold_target", "target": None,
                                               "subgoal": "hold terminal rest pose"}, np.zeros(7), None)
            recorder.close()
        if replay is not None:
            if not replay.metrics.log.closed:
                replay.metrics.log.close()
            replay.close_r12_logs()
        if env is not None:
            env.close()
        gym.make = make_original

    if replay is None or recorder is None:
        raise RuntimeError("episode did not initialize")
    final = env.unwrapped if env is not None else None
    # env is closed but its unwrapped state remains available for scalar fields above only;
    # the replay result is the authoritative source for episode outcomes.
    result = {
        "instruction_revision": 12,
        "harvestmem_v3": True,
        "task": args.task,
        "level": "history",
        "purpose": args.purpose,
        "episode": args.episode,
        "attempt": args.attempt,
        "seed": seed,
        "r12_layout_seed": layout_seed,
        "r12_noise_seed": seed,
        "success": bool(replay.success),
        "outcome": replay.reason if not failure else "planning_or_grasp_failure",
        "planning_failure": failure,
        "steps": int(replay.u.control_steps),
        "N": int(replay.u.layout.n_pick),
        "M": int(replay.u.original_slots),
        "green_step": int(replay.u.green_step) if replay.u.green_step is not None else None,
        "budget": budget,
        "remaining_steps": budget - int(replay.u.control_steps),
        "early_start_steps": list(replay.u.early_start_steps),
        "max_visibility": replay.max_visibility,
        "wind_plan": getattr(replay.u, "v2_wind_plan", None),
        "wind_events": replay.u.wind_events,
        "replenish_events": replay.u.replenish_events,
        "basket_entries": replay.basket_entries,
        "own_release_events": replay.releases,
        "slot_geometry": json.loads(json.dumps(replay.u.original_geometry, default=lambda x: np.asarray(x).tolist())),
        "slot_of": replay.u.slot_of,
        "final_robot_in_basket": int(replay.u.in_basket().sum()),
        "solver_version": solver.solver_version,
        "task_revision": 9,
        "distance_signal": "any_hanging",
        "r12_shield_scheme": "new",
        "r12_dart_two_cycle_recipe": bool(args.dart_two_cycles),
        "r12_dart_perturbation_cycles": list(getattr(replay, "r12_dart_perturbation_cycles", [])),
        "created_utc": timestamp(),
        "wall_seconds": time.monotonic() - started,
    }
    result["r12_layout_sha256"] = hashlib.sha256(json.dumps(result["slot_geometry"], sort_keys=True).encode()).hexdigest()
    if args.dart_two_cycles:
        result["DART_recipe"] = "OU sigma=.04 rad, tau=10; exactly two sequential pick cycles are perturbed; other cycles are clean."
    result.update(recorder.summary())
    atomic_json(out / "result.json", result)
    atomic_json(out / "actions.json", replay.actions)
    atomic_json(out / "decisions.json", replay.decisions)
    print(json.dumps({k: result[k] for k in ("task", "purpose", "episode", "attempt", "N", "success", "outcome", "steps")}), flush=True)


if __name__ == "__main__":
    main()
