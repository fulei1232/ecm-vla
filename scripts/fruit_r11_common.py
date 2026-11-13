"""Isolated revision 11 bindings; prior 76 source files and all runs stay read-only."""
import sys

from v_a_common import ECM, read_json, atomic_json, sha256, timestamp
from posthoc_fruit_memory_usage import tree_state

OUT = ECM/'results/fruit_task_v2_R0_r11_20261106'
TRACKED = ECM/'reports/fruit_task_v2_R0_r11_20261106'
REPORT = ECM/'reports/fruit_task_v2_design_r11_20261106.md'
DOC = ECM/'reports/fruit_task_v2_R0_r11_plan_20261106.md'
REGISTRY = ECM/'reports/split_registry_fruit_task_v2_r11_20261106.json'
OLD = ECM/'results/fruit_count_shield_v2_buildfix_R0_20261106'
PREVIOUS = ECM/'results/fruit_task_v2_R0_r10_20261106'
DATASET = ECM/'data/finetune_kf/fruit_task_v2_r11_20261106'
RAW_CLEAN = ECM/'data/fruit_task_v2_r11_clean_20261106/rm75b_eg2/train'
RAW_DART = ECM/'data/fruit_task_v2_r11_dart_20261106/rm75b_eg2/train'


def proof(name, value):
    for directory in (OUT, TRACKED):
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory/name, value)


def preserve():
    for name, h in read_json(OUT/'preservation/files_sha256.json').items():
        assert sha256(name) == h, name
    for path in (OUT/'preservation').glob('*_tree.json'):
        assert tree_state(ECM/'results'/path.name[:-len('_tree.json')]) == read_json(path), str(path)


def publish(record, **changes):
    import fruit_r10_common as clock
    record.update(changes, updated_utc=timestamp(), instruction_revision=11,
                  observation_condition='count_fixed_shield_v2/occluded_original', training=False)
    if record['state'].startswith('r9_'):
        record['state'] = 'r11_'+record['state'][3:]
    phase = dict(record)
    if phase['state'].startswith('r11_'):
        phase['state'] = 'r10_'+phase['state'][4:]
    elif phase['state'].startswith('R11_'):
        phase['state'] = 'R10_'+phase['state'][4:]
    clock._data_phase(phase)
    atomic_json(OUT/'status.json', record)
    root = read_json(ECM/'reports/lowlevel_rescue_status.json')
    root.update(phase='R11 shield Count R0/data only', state=record['state'], instruction_revision=11,
                updated_utc=record['updated_utc'],
                decision=record.get('reason', 'Revision 11: fresh shield Count R0, then 400 demonstrations only.'),
                next='Clock calibration/freeze → 350 Count audits → information → data/build/shared norm → stop; no training.')
    root['links'].update(R0_r11_report=str(REPORT), R0_r11_summary=str(TRACKED/'summary.json'), R0_r11_status=str(OUT/'status.json'))
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
        module.proof, module.publish, module.preserve = proof, publish, preserve
    # Several legacy helpers use `from fruit_r9_common import OUT`, so rebinding
    # their source module does not update a copy that was imported earlier.
    # Keep activation safe even when the R11 runtime is imported before activate().
    for module_name in ('fruit_r9_data', 'fruit_r9_generation', 'fruit_r10_data_timing'):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name, value in values.items():
            if hasattr(module, name):
                setattr(module, name, value)
        for name, value in (('proof', proof), ('publish', publish), ('preserve', preserve)):
            if hasattr(module, name):
                setattr(module, name, value)
    import fruit_r9_r0_support as support
    for name in ('OUT', 'TRACKED', 'REPORT', 'DOC'):
        setattr(support, name, values[name])
    support.proof, support.preserve = proof, preserve
    return support
