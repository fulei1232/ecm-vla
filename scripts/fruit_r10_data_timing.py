"""Generation wall time and per-N episode costs; no change to demonstrations or sampling."""
from pathlib import Path

import numpy as np

from v_a_common import read_json,timestamp
from fruit_r10_common import OUT,proof,finish_data_timing


def describe(values):
    return {'n':len(values),'mean':float(np.mean(values)) if values else None,
            'std':float(np.std(values)) if values else None,
            'range':[min(values),max(values)] if values else None}


def summarize_episode_times(records):
    by_task={}
    for task in sorted({r['task'] for r in records}):
        own=[r for r in records if r['task']==task]
        groups={}
        for group in ('all_attempts','accepted','clean_accepted','DART_accepted'):
            rows=own if group=='all_attempts' else [r for r in own if r['success']]
            if group=='clean_accepted':rows=[r for r in rows if r['purpose']=='train_clean']
            if group=='DART_accepted':rows=[r for r in rows if r['purpose']=='train_dart']
            groups[group]={str(N):{
                'process_wall_seconds':describe([r['process_wall_seconds'] for r in rows if r['N']==N]),
                'validated_case_wall_seconds':describe([r['validated_case_wall_seconds'] for r in rows if r['N']==N]),
                'physical_steps':describe([r['physical_steps'] for r in rows if r['N']==N])}
                for N in sorted({r['N'] for r in own})}
        by_task[task]=groups
    return {'by_task_and_N':by_task,'attempts':len(records),'accepted':sum(r['success'] for r in records),
        'sum_attempt_process_seconds':sum(r['process_wall_seconds'] for r in records),
        'serial_sum_is_not_parallel_wall_time':True,
        'process_scope':'Subprocess start to exit: imports, initialization, planning, simulation, rendering, and demonstration writes.',
        'validated_case_scope':'Includes parent validity checking after process exit.',
        'evaluation_cost_limit':'Generation includes planning and HDF5 recording; future evaluation also incurs policy inference. These costs are estimates, not measured VLA evaluation times.'}


def generate_data_with_timing(state,execute,run_command,check_sources,abort):
    from fruit_r9_generation import generate_data
    delivery=generate_data(state,execute,run_command,check_sources,abort)
    records=[]
    for purpose in ('train_clean','train_dart'):
        for path in sorted((OUT/'replays'/purpose).glob('*/*/ep*/process_timing.json')):
            row=read_json(path)
            if row.get('result_available'):records.append(row)
    assert sum(r['success'] for r in records)==400
    costs=summarize_episode_times(records)
    costs.update(revision=10,created_utc=timestamp(),records=records)
    phases=finish_data_timing();costs['data_stage_wall_times']=phases
    proof('data_generation_timing.json',costs)
    delivery.update(revision=10,task_definition_revision=9,data_generation_timing=costs,
        raw_generation_total_wall_seconds=phases['raw_generation_wall_seconds'],
        full_data_pipeline_wall_seconds=phases['total_wall_seconds'])
    proof('data_delivery.json',delivery)
    return delivery
