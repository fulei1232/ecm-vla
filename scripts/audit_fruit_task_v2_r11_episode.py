#!/usr/bin/env python3
"""One explicitly opted-in revision 11 replay; old task defaults stay untouched."""
from pathlib import Path
import sys

from fruit_r11_common import activate, OUT
from v_a_common import read_json, atomic_json, sha256


def main():
    activate()
    from fruit_shield_v2_runtime import install, ShieldRecorder
    from fruit_picking.count_shield_v2_buildfix import install as install_constructor
    from fruit_r11_runtime import Revision11Replay
    import fruit_r9_data
    import audit_fruit_task_v2_r9_episode as original
    install(); install_constructor()
    fruit_r9_data.H5Recorder = ShieldRecorder
    original.OUT, original.R9Replay = OUT, Revision11Replay
    directory = Path(sys.argv[sys.argv.index('--out')+1])
    try:
        original.main()
    finally:
        replay = Revision11Replay.last_instance
        if replay is not None:
            replay.metrics.log.close()
            atomic_json(directory/'case_metrics.json', replay.metrics.summary())
    result = read_json(directory/'result.json')
    replay = Revision11Replay.last_instance
    result.update(instruction_revision=11, count_shield_mode='fixed_v2' if result['task']=='count' else 'off',
                  observation_condition='count_fixed_shield_v2/occluded_original',
                  shield_geometry_checks=replay.shield_intersections_checked,
                  shield_intersection_failure=replay.intersection_failure,
                  case_metrics_sha256=sha256(directory/'case_metrics.json'),
                  frame_metrics_sha256=sha256(directory/'frame_metrics.jsonl'))
    atomic_json(directory/'result.json', result)


if __name__=='__main__':
    main()
