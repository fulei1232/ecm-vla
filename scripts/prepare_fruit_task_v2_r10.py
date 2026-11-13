#!/usr/bin/env python3
"""R10 read-only reassessment and reuse of accepted R9 evidence; no simulator or model."""
from datetime import datetime
import csv
from pathlib import Path
import shutil

from v_a_common import ECM,read_json,sha256,timestamp
from fruit_r10_common import OUT,TRACKED,R9,activate,proof,preserve


def prepare_evidence():
    support=activate()
    if (OUT/'approved_reuse.json').exists():
        value=read_json(OUT/'approved_reuse.json')
        for name,expected in value['original_files_sha256'].items():assert sha256(name)==expected,name
        for name,expected in value['accepted_unit_proofs_sha256'].items():assert sha256(OUT/name)==expected,name
        assert read_json(OUT/'performance_reassessment_r10.json')['passed']
        assert read_json(OUT/'budget_check.json')['passed']
        return value
    original={}
    def remember(path):
        original[str(path)]=sha256(path)
    remember(R9/'plan.json')
    original_plan_sha=sha256(R9/'plan.json')
    for name,h in read_json(R9/'plan.json')['source_sha256'].items():assert sha256(name)==h,name
    copied=('cpu_preflight.json','R8_information_definition_crosscheck.json','R8_default_regression.json',
            'memory_tensor_checks.json','early_start_check.json')
    for name in copied:
        remember(R9/name);value=read_json(R9/name);assert value['passed']
        for destination in (OUT/name,TRACKED/name):shutil.copyfile(R9/name,destination)
    for purpose in ('budget_expert','budget_nonstop','early_start_probe'):
        link=OUT/'replays'/purpose;link.parent.mkdir(exist_ok=True)
        link.symlink_to(R9/'replays'/purpose,target_is_directory=True)
        for path in sorted(link.glob('*/*/ep*/validity.json')):
            validity=read_json(path);assert validity['passed'] and validity['plan_sha256']==original_plan_sha
            remember(path.resolve())
            for name,h in {**validity['source_sha256'],**validity['image_sha256']}.items():
                target=path.parent/name;assert sha256(target)==h;remember(target.resolve())
    rows=[]
    for purpose,level in [('budget_expert','history'),('budget_nonstop','nonstop')]:
        for episode in range(250,260):
            d=R9/'replays'/purpose/'count'/level/f'ep{episode}'
            baseline=ECM/'results/fruit_task_v2_R0_r8_20261105/replays'/purpose/'count'/level/f'ep{episode}'
            r,s=read_json(d/'result.json'),read_json(d/'started.json')
            br,bs=read_json(baseline/'result.json'),read_json(baseline/'started.json')
            for path in (d/'started.json',baseline/'result.json',baseline/'started.json',d/'cost_comparison.json'):remember(path)
            assert r['seed']==br['seed'] and r['slot_geometry']==br['slot_geometry'] and r['green_step']==br['green_step']==630
            current=(datetime.fromisoformat(r['created_utc'])-datetime.fromisoformat(s['utc'])).total_seconds()
            before=(datetime.fromisoformat(br['created_utc'])-datetime.fromisoformat(bs['utc'])).total_seconds()
            a,b=current/r['steps'],before/br['steps'];ratio=a/b
            rows.append({'episode':episode,'purpose':purpose,'R9_steps':r['steps'],'R8_steps':br['steps'],
                'R9_started_utc':s['utc'],'R9_created_utc':r['created_utc'],
                'R8_started_utc':bs['utc'],'R8_created_utc':br['created_utc'],
                'R9_total_seconds':current,'R8_total_seconds':before,
                'R9_mean_step_seconds':a,'R8_mean_step_seconds':b,'ratio':ratio,'passed':ratio<=2,
                'R9_source':str(d),'R8_source':str(baseline)})
    failed=ECM/'reports/fruit_task_v2_R0_r9_20261106/performance_gate_failure.json'
    remember(failed)
    reassessment={'revision':10,'authority':'HANDOFF §18.2; user-approved corrected definition',
        'created_utc':timestamp(),'definition':'((R9 result.created_utc - started.utc)/R9 physical steps) / ((R8 result.created_utc - started.utc)/R8 physical steps) <=2',
        'passed':all(r['passed'] for r in rows),'cases':20,'minimum_ratio':min(r['ratio'] for r in rows),
        'maximum_ratio':max(r['ratio'] for r in rows),'R8_unweighted_mean_step_seconds':sum(r['R8_mean_step_seconds'] for r in rows)/20,
        'R9_unweighted_mean_step_seconds':sum(r['R9_mean_step_seconds'] for r in rows)/20,
        'old_failure_record':str(failed),'old_failure_sha256':sha256(failed),'old_failure_unchanged':True,
        'new_rollouts':0,'records':rows}
    proof('performance_reassessment_r10.json',reassessment)
    for folder in (OUT,TRACKED):
        with (folder/'per_step_timing.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    assert reassessment['passed'],'Corrected per-step gate failed; stop'
    # The unmodified R9 budget function only aggregates these existing records.
    budget=support.budget_check()
    budget.update(instruction_revision=10,authority='HANDOFF §18.2(3)',
        existing_records_only=True,completed_after_R9_stop=True,new_rollouts=0,
        performance_gate_record=str(OUT/'performance_reassessment_r10.json'),
        legacy_cost_comparisons_are_historical_not_current_gate=True)
    proof('budget_check.json',budget);assert budget['passed']
    assert budget['max_finish_step_including_hold']==2598 and budget['minimum_remaining_steps']==202
    assert budget['max_Nplus1_detach_step']==2644
    copied_proofs={name:sha256(OUT/name) for name in copied}
    approved={'revision':10,'created_utc':timestamp(),'approved_by':'HANDOFF §18 and current user instruction',
        'accepted_unit_proofs_sha256':copied_proofs,'original_files_sha256':original,
        'reused_valid_diagnostic_cases':29,'new_checks_or_replays':0,'old_R9_plan_sha256':original_plan_sha,
        'old_failure_sha256':sha256(failed),'task_definition_revision':9,'performance_passed':True,
        'budget_passed':True,'memory_tensor_tests_reused':True,'early_start_negative_checks_reused':9,
        'appearance_and_invisibility_evidence_reused':True}
    proof('approved_reuse.json',approved)
    preserve()
    return approved


if __name__=='__main__':
    value=prepare_evidence()
    print({'approved_reuse':True,'diagnostic_cases_reused':value['reused_valid_diagnostic_cases'],'new_replays':0})
