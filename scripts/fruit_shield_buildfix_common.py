"""Fresh run namespace for the approved fixed_v2 constructor correction."""
import sys

from v_a_common import ECM,read_json,atomic_json,sha256,timestamp
from posthoc_fruit_memory_usage import tree_state

OUT=ECM/'results/fruit_count_shield_v2_buildfix_R0_20261106'
TRACKED=ECM/'reports/fruit_count_shield_v2_buildfix_R0_20261106'
REPORT=ECM/'reports/fruit_count_shield_v2_buildfix_design_20261106.md'
DOC=ECM/'reports/fruit_count_shield_v2_buildfix_plan_20261106.md'
PREVIOUS=ECM/'results/fruit_task_v2_R0_r10_20261106'
OLD_SHIELD=ECM/'results/fruit_count_shield_R0_20261106'
OLD_V2=ECM/'results/fruit_count_shield_v2_R0_20261106'
DATASET=ECM/'data/finetune_kf/fruit_task_v2_shield_v2_buildfix_20261106'
RAW_CLEAN=ECM/'data/fruit_task_v2_shield_v2_buildfix_clean_20261106/rm75b_eg2/train'
RAW_DART=ECM/'data/fruit_task_v2_shield_v2_buildfix_dart_20261106/rm75b_eg2/train'
IMPLEMENTATION='visual_records_before_attachment_v1'
AUTHORITY='User explicitly approved fixing the SAPIEN construction-order error and continuing with the same settings'


def preserve():
    for name,h in read_json(OUT/'preservation/files_sha256.json').items():assert sha256(name)==h,name
    for p in (OUT/'preservation').glob('*_tree.json'):
        assert tree_state(ECM/'results'/p.name[:-len('_tree.json')])==read_json(p),str(p)
    for name,h in read_json(OLD_V2/'plan.json')['source_sha256'].items():assert sha256(name)==h,name


def proof(name,value):
    for folder in (OUT,TRACKED):
        folder.mkdir(parents=True,exist_ok=True);atomic_json(folder/name,value)


def publish(record,**changes):
    import fruit_r10_common as clock
    record.update(changes,updated_utc=timestamp(),instruction_revision='10+approved_shield_v2_buildfix',
                  observation_condition='count_fixed_shield_v2',implementation_revision=IMPLEMENTATION,training=False)
    if record['state'].startswith('r9_'):record['state']='r10_'+record['state'][3:]
    phase=dict(record)
    if phase['state'].startswith('SHIELD_'):phase['state']='R10_'+phase['state'][7:]
    clock._data_phase(phase)
    atomic_json(OUT/'status.json',record)
    root=read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase='Count fixed_v2 constructor correction/R0/data only',state=record['state'],
      instruction_revision=record['instruction_revision'],updated_utc=record['updated_utc'],
      decision=record.get('reason','Approved construction-time visual offsets; geometry, physics, seeds and gates unchanged.'),
      next='ep112 diagnostic, fresh clock/350 Count audits/information, then data only; stop after data, no training.')
    root['links'].update(count_shield_buildfix_report=str(REPORT),count_shield_buildfix_summary=str(TRACKED/'summary.json'),
                         count_shield_buildfix_status=str(OUT/'status.json'))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json',root)


def activate():
    sys.path[:0]=[str(ECM),str(ECM/'src')]
    import fruit_shield_v2_common as old
    for name,value in dict(OUT=OUT,TRACKED=TRACKED,REPORT=REPORT,DOC=DOC,PREVIOUS=PREVIOUS,OLD_SHIELD=OLD_SHIELD,
          DATASET=DATASET,RAW_CLEAN=RAW_CLEAN,RAW_DART=RAW_DART,AUTHORITY=AUTHORITY).items():setattr(old,name,value)
    old.proof=proof;old.publish=publish;old.preserve=preserve
    return old.activate()


def report(record,support):
    import fruit_shield_v2_reporting as previous
    for name,value in dict(OUT=OUT,TRACKED=TRACKED,REPORT=REPORT,DOC=DOC,proof=proof,preserve=preserve).items():setattr(previous,name,value)
    summary=previous.report(record,support)
    summary['implementation_revision']=IMPLEMENTATION
    summary['checks']['builder_tests.json']=read_json(OUT/'builder_tests.json')
    proof('summary.json',summary)
    content=REPORT.read_text().replace('固定遮挡v2续跑：','固定遮挡v2构建修复续跑：',1)
    REPORT.write_text(content);(OUT/'REPORT.md').write_text(content)
    return summary
