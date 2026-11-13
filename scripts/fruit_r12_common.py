"""Isolated R12 paths and bindings; R11 artifacts remain read-only."""
import sys

from v_a_common import ECM, atomic_json, read_json, sha256, timestamp

OUT = ECM/'results/fruit_task_v2_R0_r12_20261109'
TRACKED = ECM/'reports/fruit_task_v2_R0_r12_20261109'
REPORT = ECM/'reports/fruit_task_v2_design_r12_20261109.md'
DOC = ECM/'reports/fruit_task_v2_R0_r12_plan_20261109.md'
DATASET = ECM/'data/finetune_kf/fruit_task_v2_r12_20261109'
RAW_CLEAN = ECM/'data/fruit_task_v2_r12_clean_20261109/rm75b_eg2/train'
RAW_DART = ECM/'data/fruit_task_v2_r12_dart_20261109/rm75b_eg2/train'
R11_OUT = ECM/'results/fruit_task_v2_R0_r11_20261106'
R11_TRACKED = ECM/'reports/fruit_task_v2_R0_r11_20261106'


def proof(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    TRACKED.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT/name, value)
    atomic_json(TRACKED/name, value)


def publish(record, **changes):
    record.update(changes, updated_utc=timestamp(), instruction_revision=12,
                  observation_condition='count_fixed_v2_r12_false_bottom/occluded_original', training=False)
    atomic_json(OUT/'status.json', record)
    root_path = ECM/'reports/lowlevel_rescue_status.json'
    root = read_json(root_path)
    root.update(phase='R12 false-bottom Count/DART data only', state=record['state'],
                instruction_revision=12, updated_utc=record['updated_utc'],
                decision=record.get('reason', 'R12: CPU optical proof, GPU validation, then clean new data/build/norm only.'),
                next='GPU compare ep113 → paired 50 expert frame check → 400 new demonstrations → build/shared norm → stop before training.')
    root.setdefault('links', {}).update(
        R12_plan=str(DOC), R12_report=str(REPORT), R12_status=str(OUT/'status.json'),
        R12_summary=str(TRACKED/'summary.json'), R12_cpu_optics=str(OUT/'cpu_optics/result.json'))
    atomic_json(root_path, root)


def activate():
    sys.path[:0] = [str(ECM), str(ECM/'src')]
    values = {'OUT': OUT, 'TRACKED': TRACKED, 'REPORT': REPORT, 'DOC': DOC,
              'DATASET': DATASET, 'RAW_CLEAN': RAW_CLEAN, 'RAW_DART': RAW_DART}
    import fruit_r9_common as r9
    import fruit_r10_common as r10
    for module in (r9, r10):
        for name, value in values.items():
            if hasattr(module, name):
                setattr(module, name, value)
    for module_name in ('fruit_r9_data', 'fruit_r9_r0_support', 'fruit_r9_episode_support',
                        'fruit_r10_data_timing', 'fruit_r11_common', 'fruit_r11_validity'):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name, value in values.items():
            if hasattr(module, name):
                setattr(module, name, value)
    return values
