"""Freeze an independent shield continuation and preserve all previous products."""
import shutil
from pathlib import Path

from v_a_common import ECM, read_json, sha256, atomic_json, timestamp
from posthoc_fruit_memory_usage import tree_state
from fruit_shield_common import (OUT, TRACKED, DOC, PREVIOUS, DATASET, RAW_CLEAN, RAW_DART,
                                AUTHORITY, activate, preserve, proof)

NEW_SOURCES = ('fruit_picking/count_shield_spec.py','fruit_picking/count_shield_env.py',
    'scripts/fruit_shield_common.py','scripts/fruit_shield_runtime.py',
    'scripts/audit_fruit_count_shield_episode.py','scripts/diagnose_fruit_count_shield_ep112.py',
    'scripts/prepare_fruit_count_shield.py','scripts/fruit_shield_reporting.py',
    'scripts/run_fruit_count_shield.py','tests/fruit_picking/test_count_shield.py')


def initialize_preservation():
    OUT.mkdir(parents=True, exist_ok=True);TRACKED.mkdir(parents=True, exist_ok=True)
    directory=OUT/'preservation';directory.mkdir(exist_ok=True)
    if (directory/'files_sha256.json').exists():return
    old=PREVIOUS/'preservation'
    for path in old.glob('*_tree.json'):shutil.copyfile(path,directory/path.name)
    atomic_json(directory/(PREVIOUS.name+'_tree.json'),tree_state(PREVIOUS))
    files=read_json(old/'files_sha256.json')
    files.update(read_json(PREVIOUS/'plan.json')['source_sha256'])
    for f in (ECM/'reports/fruit_task_v2_R0_r10_20261106').glob('*'):
        if f.is_file():files[str(f)]=sha256(f)
    for name in ('fruit_task_v2_design_r10_20261106.md','fruit_task_v2_R0_r10_plan_20261106.md',
                 'split_registry_fruit_task_v2_r10_20261106.json'):
        f=ECM/'reports'/name;files[str(f)]=sha256(f)
    atomic_json(directory/'files_sha256.json',files)
    (OUT/'baseline_sources').symlink_to(PREVIOUS/'baseline_sources',target_is_directory=True)


def prepare():
    initialize_preservation();support=activate();preserve()
    tests=read_json(OUT/'new_tests.json');assert tests['passed']
    reuse_name=OUT/'approved_reuse.json'
    if not reuse_name.exists():
        names=('cpu_preflight.json','R8_information_definition_crosscheck.json','R8_default_regression.json',
               'memory_tensor_checks.json','early_start_check.json',
               'performance_reassessment_r10.json','budget_check.json')
        hashes={}
        for name in names:
            assert read_json(PREVIOUS/name)['passed']
            for folder in (OUT,TRACKED):shutil.copyfile(PREVIOUS/name,folder/name)
            hashes[str(PREVIOUS/name)]=sha256(PREVIOUS/name)
        proof('approved_reuse.json',{'authority':AUTHORITY,'created_utc':timestamp(),
            'proof_sha256':hashes,'unchanged_memory_and_early_start_checks_not_repeated':True,
            'old_pressure_used_for_budget_only':True,
            'budget_applicability_requires_zero_new_collision_shapes':True,
            'new_visibility_must_be_rechecked':True,'old_R0_results_not_carried_as_new_results':True})
    old=read_json(PREVIOUS/'plan.json')
    plan={'instruction':'10+approved_shield_v1','authority':AUTHORITY,'task_definition_revision':9,
        'observation_condition':'count_fixed_shield_v1/occluded_original',
        'source_sha256':{**old['source_sha256'],**{str(ECM/name):sha256(ECM/name) for name in NEW_SOURCES}},
        'previous_plan_sha256':sha256(PREVIOUS/'plan.json'),'doc_sha256':sha256(DOC),
        'registry_sha256':sha256(ECM/'reports/split_registry_fruit_count_shield_20261106.json'),
        'tests_sha256':sha256(OUT/'new_tests.json'),'reused_proofs_sha256':sha256(reuse_name),
        'reused_copies_sha256':{str(OUT/Path(name).name):h for name,h in read_json(reuse_name)['proof_sha256'].items()},
        'dev_audit':old['dev_audit'],'dev_diagnostic':old['dev_diagnostic'],'cohorts':old['cohorts'],
        'audit_seed':old['audit_seed'],'train_seed':old['train_seed'],'budget':2800,'gpus':list(range(8)),
        'memory':old['memory'],'temporal_index':'order','distance_signal':'any_hanging',
        'DATASET':str(DATASET),'RAW_CLEAN':str(RAW_CLEAN),'RAW_DART':str(RAW_DART),
        'ep112_first':True,'new_clock_fit_before_350':True,'information_gate_after_350':True,
        'fresh_count_data_required':True,'training':False,'val_use':False}
    if (OUT/'plan.json').exists():assert read_json(OUT/'plan.json')==plan,'No mixed source/config runs'
    else:proof('plan.json',plan)
    state=read_json(OUT/'status.json') if (OUT/'status.json').exists() else {
        'stage':'Count shield/R0/data only','state':'prepared_count_shield',
        'train_data_generated':False,'training':False}
    return support,state,plan
