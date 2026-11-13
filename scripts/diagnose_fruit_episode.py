#!/usr/bin/env python3
"""One diagnostic fruit-picking episode against a running fine-tuned policy server (not part of any protocol).

--mode closed: the official EpisodeEvaluator loop, as eval_fruit_episode.py, on any split's verified episodes.
--mode replay: the demonstration's own actions are executed while the policy is queried at every decision
  exactly as in closed loop (same buffers, memory and 16-step cadence); each predicted 16-step chunk is saved
  next to the demonstration's actions over the same steps (teacher-forced open-loop prediction error).
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from v_a_common import ECM, POLICY, SIM, atomic_json, read_json


def replay(official_eval, evaluator, runner, recorder, args, demo_actions, directory):
    client = official_eval._websocket_client_policy.MMEVLAWebsocketClientPolicy(args.host, args.port)
    client.reset()
    epstate = official_eval.EpisodeState()
    task_goal, _ = evaluator.init_episode(runner, epstate, directory)
    img, wrist, state = epstate.get_current_obs()
    decisions, outcome = [], "ongoing"
    for t in range(len(demo_actions)):
        if t % 16 == 0:
            chunk = np.asarray(evaluator.get_action_chunk(client, epstate, img, wrist, state, task_goal, None,
                                                          exec_horizon=16))
            epstate.clear_buffers()
            demo = demo_actions[t:t + 16]
            decisions.append({"t": t, "state": state.copy(), "pred": chunk[:len(demo)], "demo": demo,
                              "distance": recorder.rows[-1]["gt"]["fruit_distance"]})
        (img, wrist, state), stop, outcome = runner.step(demo_actions[t])
        epstate.add_observation(img, wrist, state)
        if stop:
            break
    np.savez_compressed(directory / "replay.npz", t=np.array([d["t"] for d in decisions]),
                        state=np.stack([d["state"] for d in decisions]),
                        distance=np.array([d["distance"] for d in decisions]),
                        pred=np.stack([np.pad(d["pred"], ((0, 16 - len(d["pred"])), (0, 0)), constant_values=np.nan)
                                       for d in decisions]),
                        demo=np.stack([np.pad(d["demo"], ((0, 16 - len(d["demo"])), (0, 0)), constant_values=np.nan)
                                       for d in decisions]))
    return outcome


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--context", type=Path, required=True, help="the server's context.json")
    p.add_argument("--root", type=Path, required=True, help="split directory with <task>/ep<e>_manifest.json")
    p.add_argument("--task", required=True)
    p.add_argument("--episode-id", type=int, required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--mode", choices=["closed", "replay"], required=True)
    p.add_argument("--attempt-dir", type=Path, required=True)
    p.add_argument("--joint-guard", choices=["off", "v1"], default="off")
    p.add_argument("--joint-guard-stats", type=Path)
    p.add_argument("--exec-horizon", type=int, choices=range(1, 17), default=None)
    p.add_argument("--log-execution", action="store_true", help="record raw and executed action even with guard off")
    a = p.parse_args()
    if a.joint_guard == "v1" and a.joint_guard_stats is None:
        p.error("--joint-guard v1 requires --joint-guard-stats")
    if a.mode != "closed" and (a.joint_guard != "off" or a.exec_horizon is not None):
        p.error("execution interventions are only supported in closed mode")
    a.attempt_dir.mkdir(parents=True, exist_ok=True)
    row = {"task": a.task, "split": a.root.name, "episode_id": a.episode_id, "condition": a.condition,
           "mode": a.mode, "outcome": "setup_error", "error": ""}
    runner = recorder = None
    try:
        sys.path[:0] = [str(POLICY / "examples/robomme"), str(POLICY / "packages/openpi-client/src"),
                        str(SIM / "src"), str(ECM), str(ECM / "src")]
        import eval as official_eval
        import h5py
        from subgoal_predictor import build_subgoal_predictor, NullSubgoalPredictor
        from memory_management.fruit_eval import FruitEnvRunner, FruitRawRecorder
        from memory_management.fruit_execution import configure_execution_horizon, load_joint_guard
        args = official_eval.Args(host="127.0.0.1", port=a.port, model_seed=7, model_ckpt_id=79999,
                                  policy_name="perceptual-framesamp-modul")
        assert args.max_steps == 1300 and args.use_history
        evaluator = official_eval.EpisodeEvaluator(args, a.attempt_dir)
        horizon = configure_execution_horizon(evaluator, a.exec_horizon)
        if a.exec_horizon is not None or a.log_execution or a.joint_guard != "off":
            row.update(exec_horizon=horizon, obs_horizon=args.obs_horizon, joint_guard=a.joint_guard,
                       joint_guard_stats=str(a.joint_guard_stats) if a.joint_guard_stats else None)
        predictor = build_subgoal_predictor(args, a.attempt_dir)
        assert isinstance(predictor, NullSubgoalPredictor)
        guard = load_joint_guard(a.joint_guard_stats) if a.joint_guard == "v1" else None
        runner = FruitEnvRunner(a.task, a.root, max_steps=args.max_steps, joint_guard=guard)
        runner.log_execution = a.log_execution
        man = runner.manifest(a.episode_id)
        row.update(environment_seed=man["seed"], n_pick=man.get("n_pick"))
        runner.make_env(a.episode_id)
        recorder = FruitRawRecorder(a.attempt_dir, a.task)
        recorder.install(runner)
        recorder.install_client(official_eval)
        atomic_json(a.context, {"run_id": a.attempt_dir.parent.name, "task": a.task, "episode_id": a.episode_id,
                                "variant": a.condition, "condition": a.condition, "directory": str(a.attempt_dir),
                                "rs_schedule": None})
        row["outcome"] = "error"
        if a.mode == "closed":
            row["outcome"] = evaluator.eval_each_episode(runner, predictor, a.attempt_dir)
        else:
            with h5py.File(runner.root / man["h5"], "r") as f:
                ep = f[f"episode_{man['episode']}"]
                n = sum(k.startswith("timestep_") for k in ep)
                demo = np.stack([ep[f"timestep_{t}/action/joint_action"][()] for t in range(n)]).astype(np.float64)
            row["outcome"] = replay(official_eval, evaluator, runner, recorder, args, demo, a.attempt_dir)
            row["demo_steps"] = len(demo)
    except BaseException:
        row["error"] = traceback.format_exc()
    finally:
        if recorder is not None:
            recorder.flush()
            row["executed_steps"] = recorder.control_step
            if recorder.rows:
                gt = recorder.rows[-1]["gt"]
                row.update(final_in_basket=gt["n_in_basket"], final_detached=sum(gt["detached"]))
        if runner is not None:
            runner.close_env()
    atomic_json(a.attempt_dir / "result.json", row)
    print(json.dumps(row), flush=True)
    return 0 if not row["error"] else 1


if __name__ == "__main__":
    sys.exit(main())
