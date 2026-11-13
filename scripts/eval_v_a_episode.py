#!/usr/bin/env python3
"""One V-A closed-loop episode on the val split; the upstream EpisodeEvaluator loop is unchanged."""
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
    row = {"run_id": opts.run_dir.name, "server": opts.server, "task": opts.task, "episode_id": opts.episode_id,
           "condition": opts.condition, "model_seed": config["model_seed"], "split": protocol["split"],
           "checkpoint_id": 79999, "outcome": "setup_error", "error": "", "attempt_directory": str(opts.attempt_dir)}
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
        provenance["server_identity_snapshot"] = server
        atomic_json(opts.attempt_dir / "provenance.json", provenance)
        sys.path[:0] = [str(POLICY / "examples/robomme"), str(POLICY / "packages/openpi-client/src"),
                        str(SIM / "src"), str(ECM / "src")]
        import robomme
        import eval as official_eval
        from env_runner import EnvRunner
        from robomme.env_record_wrapper import BenchmarkEnvBuilder
        from subgoal_predictor import build_subgoal_predictor, NullSubgoalPredictor
        from memory_management.validation.acm_copy.diagnostic_logging import RawRecorder
        assert Path(robomme.__file__).resolve() == (SIM / "src/robomme/__init__.py").resolve()
        assert Path(official_eval.__file__).resolve() == (POLICY / "examples/robomme/eval.py").resolve()
        args = official_eval.Args(host="127.0.0.1", port=config["port"], model_seed=config["model_seed"],
                                  model_ckpt_id=79999, policy_name="perceptual-framesamp-modul")
        assert args.obs_horizon == 16 and args.max_steps == 1300 and args.use_history
        evaluator = official_eval.EpisodeEvaluator(args, opts.run_dir)
        predictor = build_subgoal_predictor(args, opts.attempt_dir)
        assert isinstance(predictor, NullSubgoalPredictor)
        runner = EnvRunner(opts.task, opts.attempt_dir, max_steps=args.max_steps)
        # EnvRunner hardcodes dataset="test"; V-A must only touch the val split.
        runner.env_builder = BenchmarkEnvBuilder(env_id=opts.task, dataset=protocol["split"],
                                                 action_space="joint_angle", gui_render=False,
                                                 max_steps=args.max_steps)
        assert runner.env_builder.dataset == protocol["split"] == "val"
        metadata = runner.env_builder.metadata_index[(opts.task, opts.episode_id)]
        row.update(environment_seed=metadata["seed"], difficulty=metadata["difficulty"])
        atomic_json(opts.attempt_dir / "episode_metadata.json", metadata)
        runner.make_env(opts.episode_id)
        recorder = RawRecorder(opts.attempt_dir, opts.task)
        recorder.install(runner)
        recorder.install_client(official_eval)
        atomic_json(server_dir / "context.json",
                    {"run_id": opts.run_dir.name, "task": opts.task, "episode_id": opts.episode_id,
                     "variant": opts.condition, "condition": opts.condition, "directory": str(opts.attempt_dir)})
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
                gt = recorder.rows[-1].get("gt", {})
                row["final_task_index"] = gt.get("current_task_index")
        if runner is not None:
            try:
                row["n_subtasks"] = len(getattr(runner.env.unwrapped, "task_list", []) or [])
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
