#!/usr/bin/env python3
"""One fruit-picking closed-loop episode; the upstream EpisodeEvaluator loop is unchanged.

Same path as eval_teacher_episode.py with the fruit simulator behind the official runner interface
(memory_management/fruit_eval.FruitEnvRunner) and no expert branch. Episodes are the demonstrator's verified val
episodes under params["fruit"]["root"]; every frame's row in raw_frames.jsonl carries the simulator snapshot,
whose `fruit_distance` the policy server reads as the distance module's output (fruit_distance.RecordedDistance).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from v_a_common import ECM, POLICY, SIM, VALID, atomic_json, read_json, sha256, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--server", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    opts = parser.parse_args()
    start = time.monotonic()
    server_dir = opts.run_dir / f"server{opts.server}"
    config = read_json(server_dir / "config.json")
    protocol = read_json(opts.run_dir / "protocol.json")
    params = protocol["params"]
    row = {"run_id": opts.run_dir.name, "server": opts.server, "task": opts.task, "episode_id": opts.episode_id,
           "condition": opts.condition, "model_seed": config["model_seed"], "split": protocol["split"],
           "checkpoint_id": 79999, "outcome": "setup_error", "error": "", "attempt_directory": str(opts.attempt_dir),
           "checkpoint_path": (config.get("checkpoints") or {}).get(opts.condition, config["checkpoint_path"])}
    runner = recorder = None
    provenance = {"start_utc": timestamp(), "argv": sys.argv, "python": sys.executable,
                  "config_sha256": sha256(server_dir / "config.json")}
    atomic_json(opts.attempt_dir / "provenance.json", provenance)
    try:
        server = read_json(server_dir / "server_manifest.json")
        assert server["server_ready"] and server["status"] == "serving"
        for key in ("checkpoint_path", "model_seed", "port"):
            assert server[key] == config[key], key
        os.kill(server["server_pid"], 0)
        row["server_session"] = f"{server['server_pid']}@{server['started_utc']}"
        provenance["server_identity_snapshot"] = server
        atomic_json(opts.attempt_dir / "provenance.json", provenance)
        sys.path[:0] = [str(POLICY / "examples/robomme"), str(POLICY / "packages/openpi-client/src"),
                        str(SIM / "src"), str(ECM), str(ECM / "src")]
        import eval as official_eval
        from subgoal_predictor import build_subgoal_predictor, NullSubgoalPredictor
        from memory_management.fruit_eval import FruitEnvRunner, FruitRawRecorder
        from memory_management.validation.teacher_oracle import KEYFRAME
        assert opts.condition in KEYFRAME and opts.condition in params["conditions"], opts.condition
        assert Path(official_eval.__file__).resolve() == (POLICY / "examples/robomme/eval.py").resolve()
        args = official_eval.Args(host="127.0.0.1", port=config["port"], model_seed=config["model_seed"],
                                  model_ckpt_id=79999, policy_name="perceptual-framesamp-modul")
        assert args.obs_horizon == 16 and args.max_steps == 1300 and args.use_history
        evaluator = official_eval.EpisodeEvaluator(args, opts.run_dir)
        predictor = build_subgoal_predictor(args, opts.attempt_dir)
        assert isinstance(predictor, NullSubgoalPredictor)
        fruit = params["fruit"]
        root = Path(fruit["root"])
        assert root.name == protocol["split"] == "val", root
        runner = FruitEnvRunner(opts.task, root, max_steps=args.max_steps, robot=fruit["robot"])
        metadata = runner.manifest(opts.episode_id)
        row.update(environment_seed=metadata["seed"], n_pick=metadata.get("n_pick"))
        atomic_json(opts.attempt_dir / "episode_metadata.json", metadata)
        runner.make_env(opts.episode_id)
        recorder = FruitRawRecorder(opts.attempt_dir, opts.task)
        recorder.install(runner)
        recorder.install_client(official_eval)
        atomic_json(server_dir / "context.json",
                    {"run_id": opts.run_dir.name, "task": opts.task, "episode_id": opts.episode_id,
                     "variant": opts.condition, "condition": opts.condition, "directory": str(opts.attempt_dir),
                     "rs_schedule": None})
        row["outcome"] = "error"
        outcome = evaluator.eval_each_episode(runner, predictor, opts.attempt_dir)
        row["outcome"] = outcome if outcome in VALID else "error"
        if outcome not in VALID:
            row["error"] = f"Upstream evaluator returned {outcome!r}"
    except BaseException:
        row["error"] = traceback.format_exc()
        (opts.attempt_dir / "error.log").write_text(row["error"])
    finally:
        if recorder is not None:
            recorder.flush()
            row["executed_steps"] = recorder.control_step
            row["raw_frames"] = len(recorder.rows)
            row["exec_start_idx"] = recorder.exec_start
            if recorder.rows:
                gt = recorder.rows[-1]["gt"]
                row.update(n_pick=gt["n_pick"], final_in_basket=gt["n_in_basket"],
                           final_detached=sum(gt["detached"]))
        if runner is not None:
            try:
                runner.close_env()
            except BaseException:
                row["error"] += "\nCleanup error:\n" + traceback.format_exc()
                row["outcome"] = "cleanup_error"
        row["elapsed_seconds"] = round(time.monotonic() - start, 3)
    row["video_paths"] = json.dumps([str(p) for p in sorted(opts.attempt_dir.glob("*.mp4"))])
    atomic_json(opts.attempt_dir / "result.json", row)
    print(json.dumps(row), flush=True)
    return 0 if row["outcome"] in VALID else 1


if __name__ == "__main__":
    sys.exit(main())
