#!/usr/bin/env python3
"""K0/K4/K8 client with the unchanged G1 physical platform and evaluator."""
import argparse
from pathlib import Path
import sys
import traceback

from v_a_common import ECM, POLICY, SIM, atomic_json, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('port', 'episode-id'):
        p.add_argument('--' + name, type=int, required=True)
    for name in ('context', 'root', 'attempt-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--task', choices=('FruitCount', 'FruitOccludedRipe'), required=True)
    p.add_argument('--condition', choices=('K0', 'K4', 'K8'), required=True)
    p.add_argument('--model-seed', type=int, choices=(7, 8, 9), default=7)
    p.add_argument('--guard-stats', type=Path)
    a = p.parse_args()
    a.attempt_dir.mkdir(parents=True, exist_ok=True)
    row = {'task': a.task, 'episode_id': a.episode_id, 'condition': a.condition,
           'model_seed': a.model_seed, 'outcome': 'error', 'error': '', 'exec_horizon': 16,
           'obs_horizon': 16, 'observe_hold_steps': 30, 'total_step_budget': 1300,
           'joint_guard': 'v2' if a.guard_stats else 'off'}
    runner = recorder = None
    try:
        sys.path[:0] = [str(POLICY/'examples/robomme'), str(POLICY/'packages/openpi-client/src'),
                       str(SIM/'src'), str(ECM), str(ECM/'src')]
        import eval as official
        from subgoal_predictor import build_subgoal_predictor, NullSubgoalPredictor
        from memory_management.fruit_rescue_platform import (
            prefix_runner_class, prefix_recorder_class, install_observe_prefix)
        args = official.Args(host='127.0.0.1', port=a.port, model_seed=a.model_seed,
                             model_ckpt_id=11999, policy_name='perceptual-framesamp-modul')
        assert args.max_steps == 1300 and args.obs_horizon == 16 and args.use_history
        evaluator = official.EpisodeEvaluator(args, a.attempt_dir)
        predictor = build_subgoal_predictor(args, a.attempt_dir)
        assert isinstance(predictor, NullSubgoalPredictor)
        guard = read_json(a.guard_stats) if a.guard_stats else None
        runner = prefix_runner_class()(a.task, a.root, max_steps=1300, rescue_guard=guard)
        man = runner.manifest(a.episode_id)
        row.update(environment_seed=man['seed'], n_pick=man['n_pick'])
        runner.make_env(a.episode_id)
        recorder = prefix_recorder_class()(a.attempt_dir, a.task)
        recorder.install(runner)
        recorder.install_client(official)
        install_observe_prefix(evaluator, runner)
        atomic_json(a.context, {'run_id': a.attempt_dir.parent.name, 'task': a.task,
            'episode_id': a.episode_id, 'variant': a.condition, 'condition': a.condition,
            'directory': str(a.attempt_dir), 'rs_schedule': None})
        row['outcome'] = evaluator.eval_each_episode(runner, predictor, a.attempt_dir)
    except BaseException:
        row['error'] = traceback.format_exc()
    finally:
        if recorder is not None:
            recorder.flush()
            row['executed_steps'] = recorder.control_step
            if recorder.rows:
                gt = recorder.rows[-1]['gt']
                row.update(final_in_basket=gt['n_in_basket'], final_detached=sum(gt['detached']))
        if runner is not None:
            runner.close_env()
    atomic_json(a.attempt_dir/'result.json', row)
    print(row, flush=True)
    return int(bool(row['error']))


if __name__ == '__main__':
    raise SystemExit(main())
