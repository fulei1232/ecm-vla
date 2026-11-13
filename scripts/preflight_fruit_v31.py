#!/usr/bin/env python3
"""CPU preflight: preserve old behavior, native path repeats, official history flags."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np

from v_a_common import ECM,POLICY,atomic_json,read_json,sha256,timestamp
sys.path.insert(0,str(ECM))
from audit_fruit_canonical_planning import load_snapshot,make_planner
from fruit_picking.canonical_v31_planning import canonical_v31_plan
from fruit_picking.canonical_planning import result_digest
from fruit_picking.rescue_generation import mark_history
from scripts.generate_fruit_dart import NoisyExecution,NoisyRecorder,PHASE_SCALE


def main():
    out=ECM/'results/fruit_expert_v31_audit_20261102'
    diagnosis=read_json(out/'v3_failure_diagnosis/summary.json');assert diagnosis['failed_attempts']==24
    old=ast.parse(subprocess.run(['git','show','9d1082f:fruit_picking/solver.py'],cwd=ECM,capture_output=True,text=True,check=True).stdout)
    class RemoveV31(ast.NodeTransformer):
        def visit_If(self,node):
            if 'canonical_v31' in ast.unparse(node.test) and 'legacy' not in ast.unparse(node.test): return None
            return self.generic_visit(node)
        def visit_Tuple(self,node):
            node.elts=[item for item in node.elts if not isinstance(item,ast.Constant) or item.value!='canonical_v31']
            return self.generic_visit(node)
    stripped=RemoveV31().visit(ast.parse((ECM/'fruit_picking/solver.py').read_text()))
    assert ast.dump(old,include_attributes=False)==ast.dump(stripped,include_attributes=False)
    unchanged=['fruit_picking/canonical_solver.py','fruit_picking/canonical_planning.py','fruit_picking/joint_planning.py',
               'fruit_picking/generate.py','scripts/generate_fruit_dart.py']
    subprocess.run(['git','diff','--exit-code','9d1082f','--',*unchanged],cwd=ECM,check=True)
    replan=read_json(out/'v3_failure_replanning/summary.json');native=[]
    for phase in ('reach','carry'):
        case=next(c for c in replan['cases'] if c['phase']==phase and c['success'])
        data,meta=load_snapshot(Path(case['snapshot']));planner=make_planner(meta)
        previous=next(c for c in diagnosis['cases'] if c['snapshot']==case['snapshot'])
        runs=[canonical_v31_plan(planner,data['goal_pose'],data['current'],data['reference'],meta['dt'],meta['attach'],
            phase,data['point_cloud'],meta['ik_seed'],meta['ik_candidates'],previous['point_cloud_partitions']) for _ in range(3)]
        hashes=[result_digest(result) for result,trace in runs]
        assert len(set(hashes))==1 and hashes[0]==case['result_sha256']
        assert all(trace==runs[0][1] for result,trace in runs)
        native.append({'phase':phase,'snapshot':case['snapshot'],'snapshot_sha256':sha256(Path(case['snapshot'])),
            'three_exact_repetitions':True,'matches_saved_v31_probe':True,'method':runs[0][1]['method'],'sha256':hashes[0]})
    util=POLICY/'src/mme_vla_suite/dataset_builder/robomme_h5_utils.py'
    spec=importlib.util.spec_from_file_location('official_history_flags',util);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    processor=POLICY/'src/mme_vla_suite/dataset_builder/build_robomme_dataset.py'
    assertion=next(n for n in ast.walk(ast.parse(processor.read_text())) if isinstance(n,ast.Assert) and
        isinstance(n.msg,ast.Constant) and n.msg.value=='is_video_demo mismatch')
    compiled=compile(ast.Module(body=[assertion],type_ignores=[]),str(processor),'exec')
    with tempfile.TemporaryDirectory() as tmp:
        with h5py.File(Path(tmp)/'flags.h5','w') as f:
            ep=f.create_group('episode_0')
            for i in range(50):
                ep.create_dataset(f'timestep_{i}/info/is_video_demo',data=False)
                mark_history(type('Rec',(),{'g':ep,'t':i+1})(),{'phase':'observe' if i<30 else 'reach'},True)
            start=module.first_execution_step(ep);assert start==30
            for i in range(50): exec(compiled,{'ts':ep[f'timestep_{i}'],'is_video_demo':i<start})
    report={'generated_utc':timestamp(),'cpu_only':True,'formal_v31_audit_not_started':True,
        'old_solver_ast_unchanged_after_removing_v31_dispatch':True,'unchanged_v2_v3_files':unchanged,
        'native_repeats':native,'official_history_prefix_start':30,'official_all_50_frame_assertions_passed':True,
        'dart_original_classes_imported':True,'dart_phase_scale':PHASE_SCALE,'v3_failure_replan_successes':replan['successes'],
        'diagnosis_completed_before_expert_change':diagnosis['generated_utc']}
    atomic_json(out/'preflight.json',report);print(report,flush=True)


if __name__=='__main__': main()
