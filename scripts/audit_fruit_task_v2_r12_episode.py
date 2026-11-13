#!/usr/bin/env python3
"""R12 replay wrapper. All extra behavior is process-local and opt-in."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

from v_a_common import ECM, atomic_json, read_json, sha256, timestamp
from fruit_r12_common import OUT, activate


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--r12-shield-scheme', choices=('old', 'new'), default='new')
    p.add_argument('--r12-layout-seed', type=int)
    p.add_argument('--r12-dart-two-cycles', action='store_true')
    p.add_argument('--r12-capture-front-hashes', action='store_true')
    p.add_argument('--r12-expect-visibility-failure', action='store_true')
    opts, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    activate()
    import gymnasium as gym
    from fruit_picking.solver import FruitPickSolver
    from fruit_picking.count_shield_v2_buildfix import install as install_constructor
    from fruit_shield_v2_runtime import install as install_v2
    from fruit_r12_runtime import Revision12Replay
    import fruit_r9_data
    import fruit_r9_common
    import fruit_r11_common
    import audit_fruit_task_v2_r9_episode as original
    from fruit_r9_episode_support import CheckFailed

    fruit_r11_common.OUT = OUT
    fruit_r9_common.OUT = OUT
    original.OUT = OUT
    install_v2()
    install_constructor()
    if opts.r12_shield_scheme == 'new':
        from fruit_picking.count_shield_r12_visual import install as install_false_bottom
        install_false_bottom()
    import fruit_shield_runtime
    from fruit_shield_runtime import ShieldRecorder
    fruit_r9_data.H5Recorder = ShieldRecorder
    original.R9Replay = Revision12Replay

    original_make = gym.make
    original_pick = FruitPickSolver.pick
    def pick(self, actor, _index, count):
        if opts.r12_dart_two_cycles:
            index = getattr(self, 'r12_pick_cycle_counter', 0)
            self.r12_pick_cycle_counter = index + 1
        else:
            index = _index
        return original_pick(self, actor, index, count)
    FruitPickSolver.pick = pick
    made = []
    def make(env_id, *args, **kwargs):
        env = original_make(env_id, *args, **kwargs)
        made.append(env)
        if kwargs.get('task') in ('count', 'occluded_ripe'):
            base_reset = env.reset
            def reset(*, seed=None, options=None):
                layout_seed = opts.r12_layout_seed if opts.r12_layout_seed is not None else seed
                env.r12_layout_seed = int(layout_seed) if layout_seed is not None else None
                env.r12_noise_seed = int(seed) if seed is not None else None
                env.r12_two_cycle_dart = bool(opts.r12_dart_two_cycles)
                env.r12_capture_front_hashes = bool(opts.r12_capture_front_hashes)
                return base_reset(seed=layout_seed, options=options)
            env.reset = reset
        return env
    gym.make = make

    directory = None
    try:
        try:
            original.main()
        except CheckFailed as exc:
            directory = Path(sys.argv[sys.argv.index('--out')+1])
            failure_path = directory/'check_failure.json'
            if not opts.r12_expect_visibility_failure or not failure_path.exists():
                raise
            failure = read_json(failure_path)
            if failure.get('reason') != 'BASKET_OR_CONTENT_VISIBLE':
                raise
            details = failure.get('details') or {}
            if not details.get('hand_camera', {}).get('basket_fruit', 0):
                raise
            atomic_json(directory/'r12_expected_stop.json', {'expected':True,'exception':str(exc),
                'failure':failure,'created_utc':timestamp()})
        directory = directory or Path(sys.argv[sys.argv.index('--out')+1])
    finally:
        replay = Revision12Replay.last_instance
        if replay is not None:
            if not replay.metrics.log.closed:
                replay.metrics.log.close()
            if directory is not None:
                atomic_json(directory/'case_metrics.json', replay.metrics.summary())
            replay.close_r12_logs()
        gym.make = original_make
        FruitPickSolver.pick = original_pick

    directory = directory or Path(sys.argv[sys.argv.index('--out')+1])
    replay = Revision12Replay.last_instance
    result_path = directory/'result.json'
    if result_path.exists():
        result = read_json(result_path)
        layout_json = replay.u.layout.to_json()
        if not isinstance(layout_json, str):
            layout_json = json.dumps(layout_json, sort_keys=True)
        result.update(instruction_revision=12,
            count_shield_mode=('fixed_v2' if result['task']=='count' else 'off'),
            shield_geometry_checks=replay.shield_intersections_checked,
            shield_intersection_failure=replay.intersection_failure,
            case_metrics_sha256=sha256(directory/'case_metrics.json'),
            frame_metrics_sha256=sha256(directory/'frame_metrics.jsonl'),
            observation_condition=('count_fixed_v2_r12_false_bottom' if opts.r12_shield_scheme == 'new'
                                   else 'count_fixed_v2_old_geometry'),
            r12_shield_scheme=opts.r12_shield_scheme,
            r12_layout_seed=getattr(replay, 'r12_layout_seed', result['seed']),
            r12_noise_seed=result['seed'],
            r12_layout_sha256=hashlib.sha256(layout_json.encode()).hexdigest(),
            r12_dart_two_cycle_recipe=bool(opts.r12_dart_two_cycles),
            r12_dart_perturbation_cycles=getattr(replay, 'r12_dart_perturbation_cycles', []),
            r12_front_frame_hashes=opts.r12_capture_front_hashes)
        if opts.r12_dart_two_cycles:
            result['DART_recipe'] = ('OU sigma=.04 rad, tau=10; two attempt-random pick_index cycles only; '
                'reach/pull/carry full scale, approach half; all other pick cycles are clean.')
        atomic_json(result_path, result)
        atomic_json(directory/'r12_episode.json', {'passed':True,'result':result,
            'scene_manifest_sha256':sha256(directory/'scene_manifest.json'),
            'front_frame_hashes_sha256':sha256(directory/'front_rgb_sha256.jsonl')
                if (directory/'front_rgb_sha256.jsonl').exists() else None})
    elif (directory/'r12_expected_stop.json').exists():
        atomic_json(directory/'r12_episode.json', {'passed':True,'expected_visibility_failure':True,
            'stop':read_json(directory/'r12_expected_stop.json'),
            'scene_manifest_sha256':sha256(directory/'scene_manifest.json'),
            'front_frame_hashes_sha256':None})
    else:
        raise RuntimeError(f'R12 episode produced neither result nor expected stop: {directory}')


if __name__ == '__main__':
    main()
