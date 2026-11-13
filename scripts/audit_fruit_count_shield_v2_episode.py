#!/usr/bin/env python3
"""The original expert with the explicit optical seam repair only."""
from fruit_shield_v2_common import activate,OUT


if __name__=='__main__':
    activate()
    import audit_fruit_task_v2_r9_episode as original
    import fruit_r9_data
    from fruit_shield_v2_runtime import install,SeamReplay,ShieldRecorder
    install();fruit_r9_data.H5Recorder=ShieldRecorder
    original.OUT=OUT;original.R9Replay=SeamReplay
    original.main()
    from pathlib import Path
    import sys
    from v_a_common import read_json,atomic_json
    d=Path(sys.argv[sys.argv.index('--out')+1]);r=read_json(d/'result.json')
    r.update(count_shield_mode='fixed_v2' if r['task']=='count' else 'off',
        observation_condition='count_fixed_shield_v2/occluded_original',
        shield_geometry_checks=SeamReplay.last_instance.shield_intersections_checked)
    atomic_json(d/'result.json',r)
