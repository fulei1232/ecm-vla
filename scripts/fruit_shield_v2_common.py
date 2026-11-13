"""Separate output bindings for the user-approved seam repair after fixed-v1 failed."""
import sys

from v_a_common import ECM,read_json,atomic_json,sha256,timestamp
from posthoc_fruit_memory_usage import tree_state

OUT=ECM/'results/fruit_count_shield_v2_R0_20261106'
TRACKED=ECM/'reports/fruit_count_shield_v2_R0_20261106'
REPORT=ECM/'reports/fruit_count_shield_v2_design_20261106.md'
DOC=ECM/'reports/fruit_count_shield_v2_plan_20261106.md'
PREVIOUS=ECM/'results/fruit_task_v2_R0_r10_20261106'
OLD_SHIELD=ECM/'results/fruit_count_shield_R0_20261106'
DATASET=ECM/'data/finetune_kf/fruit_task_v2_shield_v2_20261106'
RAW_CLEAN=ECM/'data/fruit_task_v2_shield_v2_clean_20261106/rm75b_eg2/train'
RAW_DART=ECM/'data/fruit_task_v2_shield_v2_dart_20261106/rm75b_eg2/train'
AUTHORITY='User approved correcting lower seam and coplanar surfaces, then continuing the original gated data-only queue'


def preserve():
    for name,h in read_json(OUT/'preservation/files_sha256.json').items():assert sha256(name)==h,name
    for p in (OUT/'preservation').glob('*_tree.json'):
        assert tree_state(ECM/'results'/p.name[:-len('_tree.json')])==read_json(p),str(p)
    for name,h in read_json(OLD_SHIELD/'plan.json')['source_sha256'].items():assert sha256(name)==h,name


def proof(name,value):
    for folder in (OUT,TRACKED):
        folder.mkdir(parents=True,exist_ok=True);atomic_json(folder/name,value)


def publish(record,**changes):
    import fruit_r10_common as clock
    record.update(changes,updated_utc=timestamp(),instruction_revision='10+approved_shield_v2',
                  observation_condition='count_fixed_shield_v2',training=False)
    if record['state'].startswith('r9_'):record['state']='r10_'+record['state'][3:]
    phase=dict(record)
    if phase['state'].startswith('SHIELD_'):phase['state']='R10_'+phase['state'][7:]
    clock._data_phase(phase)
    atomic_json(OUT/'status.json',record)
    root=read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase='Count shield v2/R0/data only',state=record['state'],instruction_revision=record['instruction_revision'],
        updated_utc=record['updated_utc'],decision=record.get('reason','Approved continuous optical liner; original thresholds and stop rules.'),
        next='ep112 diagnostic, fresh Count clock/350 audits/information, then fresh data; stop after data, no training.')
    root['links'].update(count_shield_v2_report=str(REPORT),count_shield_v2_summary=str(TRACKED/'summary.json'),count_shield_v2_status=str(OUT/'status.json'))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)


def activate():
    sys.path[:0]=[str(ECM),str(ECM/'src')]
    import fruit_shield_common as old
    for name,value in dict(OUT=OUT,TRACKED=TRACKED,REPORT=REPORT,DOC=DOC,PREVIOUS=PREVIOUS,
            DATASET=DATASET,RAW_CLEAN=RAW_CLEAN,RAW_DART=RAW_DART,AUTHORITY=AUTHORITY).items():setattr(old,name,value)
    old.proof=proof;old.publish=publish;old.preserve=preserve
    return old.activate()
