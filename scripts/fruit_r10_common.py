"""Revision10 continuation bindings. R9 task, selector and rollout source files stay unchanged."""
from pathlib import Path
import sys
import time

from v_a_common import ECM, read_json, atomic_json, sha256, timestamp
from posthoc_fruit_memory_usage import tree_state

OUT = ECM / 'results/fruit_task_v2_R0_r10_20261106'
TRACKED = ECM / 'reports/fruit_task_v2_R0_r10_20261106'
REPORT = ECM / 'reports/fruit_task_v2_design_r10_20261106.md'
DOC = ECM / 'reports/fruit_task_v2_R0_r10_plan_20261106.md'
R9 = ECM / 'results/fruit_task_v2_R0_r9_20261106'
DATASET = ECM / 'data/finetune_kf/fruit_task_v2_r10_20261106'
RAW_CLEAN = ECM / 'data/fruit_task_v2_r10_clean_20261106/rm75b_eg2/train'
RAW_DART = ECM / 'data/fruit_task_v2_r10_dart_20261106/rm75b_eg2/train'
_phase_clock = None
_phase_record = None


def preserve():
    for path in (OUT/'preservation').glob('*_tree.json'):
        name=path.name[:-len('_tree.json')]
        assert tree_state(ECM/'results'/name)==read_json(path),name
    for name,expected in read_json(OUT/'preservation/files_sha256.json').items():
        assert sha256(name)==expected,name
    for name,expected in read_json(R9/'plan.json')['source_sha256'].items():
        assert sha256(name)==expected,name


def proof(name,value):
    for directory in (OUT,TRACKED):atomic_json(directory/name,value)


def _data_phase(state):
    global _phase_clock,_phase_record
    phase=state['state']
    if phase=='r10_generating_clean_DART':
        assert _phase_clock is None
        _phase_clock=time.monotonic()
        _phase_record={'started_utc':timestamp(),'raw_generation_wall_seconds':None,
                       'feature_build_wall_seconds':None,'normalization_and_delivery_wall_seconds':None,
                       'completed':False,'total_wall_seconds':None}
        proof('data_phase_timing.json',_phase_record)
    elif _phase_clock is not None and phase=='r10_building_shared_data':
        moment=time.monotonic()
        _phase_record['raw_generation_wall_seconds']=moment-_phase_clock
        _phase_record['raw_generation_finished_utc']=timestamp()
        _phase_record['build_started_monotonic']=moment
        proof('data_phase_timing.json',_phase_record)
    elif _phase_clock is not None and phase=='r10_computing_one_shared_normalization':
        moment=time.monotonic()
        _phase_record['feature_build_wall_seconds']=moment-_phase_record['build_started_monotonic']
        _phase_record['normalization_started_monotonic']=moment
        proof('data_phase_timing.json',_phase_record)
    elif _phase_clock is not None and phase.startswith('R10_'):
        _phase_record['elapsed_wall_seconds_at_stop']=time.monotonic()-_phase_clock
        _phase_record['stopped_utc']=timestamp()
        proof('data_phase_timing.json',_phase_record)


def finish_data_timing():
    assert _phase_record is not None
    moment=time.monotonic()
    _phase_record['total_wall_seconds']=moment-_phase_clock
    _phase_record['normalization_and_delivery_wall_seconds']=moment-_phase_record['normalization_started_monotonic']
    _phase_record['completed']=True;_phase_record['finished_utc']=timestamp()
    value={k:v for k,v in _phase_record.items() if not k.endswith('_monotonic')}
    proof('data_phase_timing.json',value)
    return value


def publish(record,**changes):
    record.update(changes,updated_utc=timestamp(),revision=10,instruction_revision=10)
    if record['state'].startswith('r9_'):record['state']='r10_'+record['state'][3:]
    _data_phase(record)
    atomic_json(OUT/'status.json',record)
    root=read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase='R0/data_only',instruction_revision=10,state=record['state'],updated_utc=record['updated_utc'],
                decision='Revision10: reuse accepted R9 checks; physical-step mean wall-time gate; no training.',
                next='Clock freeze, 350 Count audits, information gates, data-only generation, then stop; any failed gate stops.')
    root['links'].update(R0_r10_report=str(REPORT),R0_r10_summary=str(TRACKED/'summary.json'),
                         R0_r10_status=str(OUT/'status.json'))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)


def activate():
    """Bind existing helpers to the new output tree in this process only.

    The old Python files and all R9 artifacts are read-only. The task revision remains 9.
    """
    sys.path[:0]=[str(ECM),str(ECM/'src')]
    import fruit_r9_common as common
    values={'OUT':OUT,'TRACKED':TRACKED,'REPORT':REPORT,'DOC':DOC,'DATASET':DATASET,
            'RAW_CLEAN':RAW_CLEAN,'RAW_DART':RAW_DART}
    for name,value in values.items():setattr(common,name,value)
    common.proof=proof;common.publish=publish;common.preserve=preserve
    import fruit_r9_r0_support as support
    for name in ('OUT','TRACKED','REPORT','DOC'):setattr(support,name,values[name])
    support.proof=proof;support.preserve=preserve
    return support
