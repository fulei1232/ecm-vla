#!/usr/bin/env python3
"""Native CPU integration probe; no simulation, inference or held-out layouts."""
import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mplib
import numpy as np

from v_a_common import ECM,atomic_json,sha256,timestamp
sys.path.insert(0,str(ECM))
from audit_fruit_canonical_planning import load_snapshot,make_planner,replan
from fruit_picking.agent import RM75BEG2
from fruit_picking.canonical_planning import result_digest
from fruit_picking.canonical_solver import configure,initialize,plan
from fruit_picking.layout import sample_layout
from fruit_picking.solver import FruitPickSolver,PRE_GRASP,horizontal_grasp_pose


def compatibility():
    class StripCanonical(ast.NodeTransformer):
        def visit_If(self,node):
            source=ast.unparse(node.test)
            if 'canonical' in source and 'nearest' not in source and 'legacy' not in source: return None
            return self.generic_visit(node)
        def visit_Tuple(self,node):
            node.elts=[x for x in node.elts if not isinstance(x,ast.Constant) or x.value!='canonical']
            return self.generic_visit(node)
        def visit_FunctionDef(self,node):
            if node.name=='run_episode':
                assert node.args.args[-1].arg=='canonical_options'
                node.args.args=node.args.args[:-1];node.args.defaults=node.args.defaults[:-1]
            return self.generic_visit(node)
    records=[]
    for name in ('fruit_picking/solver.py','fruit_picking/generate.py'):
        old=ast.parse(subprocess.run(['git','show','f992da7:'+name],cwd=ECM,capture_output=True,text=True,check=True).stdout)
        new=StripCanonical().visit(ast.parse((ECM/name).read_text()))
        oldf={x.name:x for x in ast.walk(old) if isinstance(x,ast.FunctionDef)}
        newf={x.name:x for x in ast.walk(new) if isinstance(x,ast.FunctionDef)}
        names=['__init__','qpos','fk','obstacles','refresh_obstacles','action','follow','hold','plan','move','pick','run'] if 'solver' in name else ['run_episode','verify']
        for fn in names: assert ast.dump(oldf[fn],include_attributes=False)==ast.dump(newf[fn],include_attributes=False),(name,fn)
        records.append({'path':name,'functions':names,'legacy_nearest_ast_unchanged':True})
    unchanged=['fruit_picking/joint_planning.py','scripts/repeat_fruit_expert.py','scripts/run_fruit_expert_v2_audit.py',
               'scripts/report_fruit_expert_v2.py','scripts/fruit_expert_audit_stats.py']
    subprocess.run(['git','diff','--exit-code','f992da7','--',*unchanged],cwd=ECM,check=True)
    return {'mode_dispatch_checks':records,'v2_files_unchanged':unchanged}


def main():
    out=ECM/'results/fruit_expert_v3_audit_20261102';out.mkdir(parents=True,exist_ok=True)
    reference=np.asarray(RM75BEG2.keyframes['rest'].qpos,dtype=np.float64)
    records=[]
    for task in ('count','occluded_ripe'):
        layout=sample_layout(task,100000);target=layout.targets[0]
        positions=np.asarray([fruit['pos'] for fruit in layout.fruits]);center=positions[target]
        direction=np.r_[center[:2],0.]/np.linalg.norm(center[:2])
        pose=horizontal_grasp_pose(center-PRE_GRASP*direction,direction)
        shell=object.__new__(FruitPickSolver);shell.L=layout
        shell.u=SimpleNamespace(fruit_positions=lambda:positions,attached=[False]*len(positions),agent=RM75BEG2)
        shell.planner=mplib.Planner(urdf=RM75BEG2.urdf_path,srdf=RM75BEG2.urdf_path.replace('.urdf','.srdf'),move_group='tcp')
        shell.planner.set_base_pose(np.array([0,0,0,1.,0,0,0]))
        shell.planner.joint_vel_limits=np.asarray(shell.planner.joint_vel_limits)*2
        shell.planner.joint_acc_limits=np.asarray(shell.planner.joint_acc_limits)*2
        shell.qpos=lambda:reference.copy();shell.step_fn=lambda *a:None
        shell.dt=.05;shell.ik_seed=0;shell.ik_candidates=64;shell.trajectory_audit=[]
        initialize(shell);configure(shell,{'snapshot_dir':out/'preflight_states'/task})
        shell.canonical_labels={'phase':'reach','target':target};shell.canonical_phase='reach'
        shell.canonical_point_cloud=shell.obstacles(skip=(target,))
        result=plan(shell,pose,False);trace=shell.trajectory_audit[-1]
        assert result is not None,trace
        snapshot=Path(trace['snapshot']);data,meta=load_snapshot(snapshot);planner=make_planner(meta)
        repetitions=[replan(planner,data,meta) for _ in range(3)]
        hashes=[result_digest(r) for r,t in repetitions]
        assert all(hash_value==result_digest(result) for hash_value in hashes)
        assert all(t==repetitions[0][1] for r,t in repetitions)
        records.append({'task':task,'seed':100000,'snapshot':str(snapshot),'snapshot_sha256':sha256(snapshot),
            'method':trace['method'],'waypoint':trace.get('waypoint'),'steps':len(result['position']),
            'three_restored_paths_bitwise_identical':True,'matches_original_path':True,
            'goal_arm_qpos':trace['goal_arm_qpos'],'sha256':hashes[0]})
    checks={'generated_utc':timestamp(),'cpu_native_planning_only':True,'full_audit_not_run':True,
        'gpu_devices_visible':bool(list(Path('/dev').glob('nvidia*'))),
        'compatibility':compatibility(),'native_snapshot_roundtrips':records}
    atomic_json(out/'preflight.json',checks);print(checks,flush=True)


if __name__=='__main__':main()
