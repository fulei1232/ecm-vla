#!/usr/bin/env python3
"""The original audited expert with an opt-in fixed Count-v2 enclosure."""
from fruit_shield_common import activate, OUT


if __name__ == '__main__':
    activate()
    import audit_fruit_task_v2_r9_episode as original
    from fruit_shield_runtime import install_environment, ShieldReplay, ShieldRecorder
    import fruit_r9_data
    install_environment()
    fruit_r9_data.H5Recorder = ShieldRecorder
    original.OUT = OUT
    original.R9Replay = ShieldReplay
    original.main()
    from pathlib import Path
    import sys
    from v_a_common import read_json, atomic_json
    destination = Path(sys.argv[sys.argv.index('--out')+1])
    result = read_json(destination/'result.json')
    result['count_shield_mode'] = 'fixed_v1' if result['task'] == 'count' else 'off'
    result['observation_condition'] = 'count_fixed_shield_v1/occluded_original'
    result['shield_geometry_checks'] = ShieldReplay.last_instance.shield_intersections_checked
    atomic_json(destination/'result.json', result)
