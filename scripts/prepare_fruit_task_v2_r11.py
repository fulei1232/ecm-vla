"""Freeze revision 11 without touching the reviewed diagnostics or old source."""
import shutil

from v_a_common import ECM, read_json, sha256, atomic_json, timestamp
from posthoc_fruit_memory_usage import tree_state
from fruit_r11_common import (OUT, TRACKED, DOC, REGISTRY, OLD, PREVIOUS, DATASET,
                              RAW_CLEAN, RAW_DART, activate, preserve, proof)

NEW_SOURCES = ('scripts/fruit_r11_common.py', 'scripts/fruit_r11_rules.py',
    'scripts/fruit_r11_metrics.py', 'scripts/fruit_r11_runtime.py',
    'scripts/fruit_r11_validity.py', 'scripts/fruit_r11_reporting.py',
    'scripts/prepare_fruit_task_v2_r11.py', 'scripts/run_fruit_task_v2_r11.py',
    'scripts/audit_fruit_task_v2_r11_episode.py', 'tests/fruit_picking/test_task_v2_r11.py')


def initialize_preservation():
    OUT.mkdir(parents=True, exist_ok=True)
    TRACKED.mkdir(parents=True, exist_ok=True)
    d = OUT/'preservation'
    d.mkdir(exist_ok=True)
    if (d/'files_sha256.json').exists():
        return
    for path in (OLD/'preservation').glob('*_tree.json'):
        shutil.copyfile(path, d/path.name)
    atomic_json(d/(OLD.name+'_tree.json'), tree_state(OLD))
    files = read_json(OLD/'preservation/files_sha256.json')
    files.update(read_json(OLD/'plan.json')['source_sha256'])
    for f in (ECM/'reports/fruit_count_shield_v2_buildfix_R0_20261106').glob('*'):
        if f.is_file():
            files[str(f)] = sha256(f)
    for name in ('fruit_count_shield_v2_buildfix_design_20261106.md',
                 'fruit_count_shield_v2_buildfix_plan_20261106.md',
                 'split_registry_fruit_count_shield_buildfix_20261106.json'):
        f = ECM/'reports'/name
        files[str(f)] = sha256(f)
    atomic_json(d/'files_sha256.json', files)
    (OUT/'baseline_sources').symlink_to(PREVIOUS/'baseline_sources', target_is_directory=True)


def prepare():
    initialize_preservation()
    support = activate()
    preserve()
    assert read_json(OUT/'revision11_tests.json')['passed']
    names = ('cpu_preflight.json', 'R8_information_definition_crosscheck.json',
             'R8_default_regression.json', 'memory_tensor_checks.json', 'early_start_check.json',
             'performance_reassessment_r10.json', 'budget_check.json', 'new_tests.json',
             'builder_tests.json', 'optical_geometry_check.json')
    for name in names:
        evidence = read_json(OLD/name)
        if name == 'optical_geometry_check.json':
            # This accepted proof records the measured clearance instead of a boolean.
            assert evidence['minimum_positive_occluder_separation_mm'] >= 1.0
        else:
            assert evidence['passed']
        for folder in (OUT, TRACKED):
            destination = folder/name
            if not destination.exists():
                shutil.copyfile(OLD/name, destination)
            assert sha256(destination) == sha256(OLD/name)
    proof('approved_reuse.json', {'instruction_revision': 11, 'created_utc': timestamp(),
        'proof_sha256': {str(OLD/name): sha256(OLD/name) for name in names},
        'accepted_checks_not_repeated': True, 'R10_R0_historical_unshielded_only': True,
        'old_83_demonstrations_not_mixed': True, 'fresh_Count_R0_and_all_400_data_required': True,
        'physics_expert_DART_seed_and_budget_unchanged': True}) if not (OUT/'approved_reuse.json').exists() else None
    if not (OUT/'ep112_reviewed_failure.json').exists():
        r = read_json(OLD/'diagnostics/ep112_shield_full/result.json')
        assert not r['success'] and r['outcome']=='robot_drop' and r['steps']==837
        proof('ep112_reviewed_failure.json', {'instruction_revision': 11, 'accepted_diagnostic': True,
            'result_path': str(OLD/'diagnostics/ep112_shield_full/result.json'),
            'result_sha256': sha256(OLD/'diagnostics/ep112_shield_full/result.json'),
            'review_path': str(OLD/'diagnostic_outcome_review.json'),
            'review_sha256': sha256(OLD/'diagnostic_outcome_review.json'),
            'outcome': 'robot_drop', 'success': False, 'repeated': False, 'included_in_training_data': False,
            'whole_episode_success_requirement_cancelled_by_revision11': True})
    previous = read_json(OLD/'plan.json')
    platform = {'instruction_revision': 11, 'count_env_id': 'FruitPickCountShieldV2-v0',
        'count_shield': 'fixed_v2', 'required_for': ['Count R0','Count data','Count training observations','Count evaluation'],
        'occluded_env_id': 'FruitPickTaskV2-v0', 'occluded_shield': 'off',
        'constructor_implementation': 'visual_records_before_attachment_v1',
        'shield_geometry_sha256': sha256(ECM/'fruit_picking/count_shield_v2_spec.py'),
        'constructor_sha256': sha256(ECM/'fruit_picking/count_shield_v2_buildfix.py'),
        'zero_new_collision_shapes': True, 'task_revision': 9, 'budget': 2800,
        'basket_body_pixels': 'report_only', 'basket_fruit_and_inactive_reserve_pixels': 'zero_fatal',
        'fruit_intersection_R0': 'fatal', 'fruit_intersection_data': 'fail_attempt; global stop if count>=3 and count/started>0.02',
        'gripper_distance': 'report_only; exact visible finger boxes and TCP vs full rendered shield volume',
        'legacy_defaults_unchanged': True, 'distance_signal': 'any_hanging', 'temporal_index': 'order',
        'arms': ['K0','K8','KA'], 'training': False}
    if (OUT/'platform_manifest.json').exists():
        assert read_json(OUT/'platform_manifest.json') == platform
    else:
        proof('platform_manifest.json', platform)
    plan = {'instruction': 11, 'task_definition_revision': 9,
        'source_sha256': {**previous['source_sha256'], **{str(ECM/n): sha256(ECM/n) for n in NEW_SOURCES}},
        'previous_plan_sha256': sha256(OLD/'plan.json'), 'doc_sha256': sha256(DOC),
        'registry_sha256': sha256(REGISTRY), 'tests_sha256': sha256(OUT/'revision11_tests.json'),
        'platform_sha256': sha256(OUT/'platform_manifest.json'),
        'reused_copies_sha256': {str(OUT/name): sha256(OUT/name) for name in names},
        'pinned_proofs_sha256': {str(OUT/name): sha256(OUT/name) for name in
            ('approved_reuse.json','ep112_reviewed_failure.json','platform_manifest.json')},
        **{k: previous[k] for k in ('dev_audit','dev_diagnostic','cohorts','audit_seed','train_seed','memory')},
        'budget': 2800, 'gpus': list(range(8)), 'distance_signal': 'any_hanging', 'temporal_index': 'order',
        'observation_condition': 'count_fixed_shield_v2/occluded_original',
        'DATASET': str(DATASET), 'RAW_CLEAN': str(RAW_CLEAN), 'RAW_DART': str(RAW_DART),
        'ep112_not_repeated': True, 'first_new_work': 'clock_calibration',
        'data_attempts_per_episode': 16, 'data_intersection_count_threshold': 3,
        'data_intersection_fraction_threshold_strict': .02, 'training': False, 'val_use': False}
    if (OUT/'plan.json').exists():
        assert read_json(OUT/'plan.json') == plan, 'No mixed source/config runs'
    else:
        proof('plan.json', plan)
    state = read_json(OUT/'status.json') if (OUT/'status.json').exists() else {
        'stage': 'R11 Count shield R0/data only', 'state': 'prepared_r11',
        'train_data_generated': False, 'training': False}
    return support, state, plan
