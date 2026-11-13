"""User-approved fixed-shield continuation after R10; no edits to old sources."""
from pathlib import Path
import sys

from v_a_common import ECM, read_json, atomic_json, sha256, timestamp
from posthoc_fruit_memory_usage import tree_state

OUT = ECM/'results/fruit_count_shield_R0_20261106'
TRACKED = ECM/'reports/fruit_count_shield_R0_20261106'
REPORT = ECM/'reports/fruit_count_shield_design_20261106.md'
DOC = ECM/'reports/fruit_count_shield_plan_20261106.md'
PREVIOUS = ECM/'results/fruit_task_v2_R0_r10_20261106'
DATASET = ECM/'data/finetune_kf/fruit_task_v2_shield_20261106'
RAW_CLEAN = ECM/'data/fruit_task_v2_shield_clean_20261106/rm75b_eg2/train'
RAW_DART = ECM/'data/fruit_task_v2_shield_dart_20261106/rm75b_eg2/train'
AUTHORITY = 'User approved fixed Count-v2 enclosure after R10 ep112 visibility failure'


def preserve():
    files = read_json(OUT/'preservation/files_sha256.json')
    for name, expected in files.items():
        assert sha256(name) == expected, name
    for path in (OUT/'preservation').glob('*_tree.json'):
        name = path.name[:-len('_tree.json')]
        assert tree_state(ECM/'results'/name) == read_json(path), name
    for name, expected in read_json(PREVIOUS/'plan.json')['source_sha256'].items():
        assert sha256(name) == expected, name


def proof(name, value):
    for directory in (OUT, TRACKED):
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory/name, value)


def publish(record, **changes):
    import fruit_r10_common as clock
    record.update(changes, updated_utc=timestamp(), instruction_revision='10+approved_shield_v1',
                  observation_condition='count_fixed_shield_v1', training=False)
    if record['state'].startswith('r9_'):
        record['state'] = 'r10_'+record['state'][3:]
    phase = dict(record)
    if phase['state'].startswith('SHIELD_'):
        phase['state'] = 'R10_'+phase['state'][7:]
    clock._data_phase(phase)
    atomic_json(OUT/'status.json', record)
    root = read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase='Count shield/R0/data only', state=record['state'],
                instruction_revision=record['instruction_revision'], updated_utc=record['updated_utc'],
                decision=record.get('reason', 'Approved fixed optical shield; original thresholds and stop rules.'),
                next='ep112 diagnostic, Count calibration/350 audits/information, then fresh data; stop after data, no training.')
    root['links'].update(count_shield_report=str(REPORT), count_shield_summary=str(TRACKED/'summary.json'),
                         count_shield_status=str(OUT/'status.json'))
    atomic_json(ECM/'reports/lowlevel_rescue_status.json', root)


def activate():
    sys.path[:0] = [str(ECM), str(ECM/'src')]
    import fruit_r9_common as old
    import fruit_r10_common as clock
    values = dict(OUT=OUT, TRACKED=TRACKED, REPORT=REPORT, DOC=DOC,
                  DATASET=DATASET, RAW_CLEAN=RAW_CLEAN, RAW_DART=RAW_DART)
    for module in (old, clock):
        for name, value in values.items():
            setattr(module, name, value)
        module.proof = proof; module.publish = publish; module.preserve = preserve
    import fruit_r9_r0_support as support
    for name in ('OUT', 'TRACKED', 'REPORT', 'DOC'):
        setattr(support, name, values[name])
    support.proof = proof; support.preserve = preserve
    return support
