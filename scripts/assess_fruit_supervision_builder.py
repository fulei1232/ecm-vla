#!/usr/bin/env python3
"""Read-only assessment of official history-prefix support, using synthetic HDF5 flags."""
import ast
import importlib.util
from pathlib import Path
import tempfile

import h5py

from v_a_common import ECM,POLICY,atomic_json,sha256,timestamp


def main():
    directory=POLICY/'src/mme_vla_suite/dataset_builder'
    util=directory/'robomme_h5_utils.py';processor=directory/'build_robomme_dataset.py'
    spec=importlib.util.spec_from_file_location('official_h5_utils',util)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tree=ast.parse(processor.read_text())
    assertion=next(n for n in ast.walk(tree) if isinstance(n,ast.Assert) and
                   isinstance(n.msg,ast.Constant) and n.msg.value=='is_video_demo mismatch')
    check=compile(ast.Module(body=[assertion],type_ignores=[]),str(processor),'exec')
    cases={}
    with tempfile.TemporaryDirectory() as tmp:
        for name,flags in {'prefix':[True]*30+[False]*20,'middle':[True]*30+[False]*10+[True]*5+[False]*5}.items():
            with h5py.File(Path(tmp)/f'{name}.h5','w') as f:
                ep=f.create_group('episode_0')
                for i,flag in enumerate(flags): ep.create_dataset(f'timestep_{i}/info/is_video_demo',data=flag)
                start=module.first_execution_step(ep);failed=[]
                for step in range(len(flags)):
                    try: exec(check,{'ts':ep[f'timestep_{step}'],'is_video_demo':step<start})
                    except AssertionError: failed.append(step)
                cases[name]={'exec_start_idx':start,'failed_assertion_steps':failed,'passed':not failed}
    assert cases['prefix']['passed'] and cases['middle']['failed_assertion_steps']==list(range(40,45))
    result={'assessed_utc':timestamp(),'native_middle_unsupervised_supported':False,'prefix_supported':True,
            'cases':cases,'source_assertion_line':assertion.lineno,
            'sources':{str(p):sha256(p) for p in [util,processor,ECM/'scripts/build_fruit_dataset.py',ECM/'scripts/build_finetune_dataset.py']},
            'conclusion':'The current official processor requires is_video_demo == (step_idx < exec_start_idx), and ECM plans a contiguous supervised suffix. It cannot natively mark middle settle/done frames history-only.',
            'phase3_action':'Use observe history-only prefix; leave settle/done supervised. Preserve all frame timestamps, do not delete frames.',
            'proposal_not_implemented':'If later approved, add a supervision mask independent of is_video_demo, retain all per-frame features/step_idx, generate only selected anchor PKLs with contiguous sample IDs, and update planning/count validation and downstream-loader audits together.',
            'implementation_changed':False}
    out=ECM/'results/fruit_expert_v2_audit_20261102/builder_assessment.json';out.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(out,result);print(result)


if __name__=='__main__': main()
