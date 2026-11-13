"""New rescue-only generation; original v2/v3 generators remain untouched."""
import json
import os
from pathlib import Path
import re

import numpy as np

from .generate import IMAGE_KW,Recorder,TASK_NAME,atomic_write


def mark_history(rec,labels,prefix):
    if prefix and rec.t<=30:
        assert labels['phase']=='observe'
        rec.g[f'timestep_{rec.t-1}/info/is_video_demo'][...]=True


class PrefixRecorder(Recorder):
    history_prefix=False
    def step(self,action,labels):
        result=super().step(action,labels);mark_history(self,labels,self.history_prefix);return result


def run_episode(task,episode,seed,out,ik_mode='canonical_v31',dart=False,history_prefix=False,snapshot_dir=None):
    import gymnasium as gym
    import h5py
    import fruit_picking.env  # noqa
    from .solver import FruitPickSolver,PlanningFailure
    from .canonical_solver import configure
    from scripts.generate_fruit_dart import NoisyExecution,NoisyRecorder,PHASE_SCALE

    class PrefixNoisyRecorder(NoisyRecorder):
        def step(self,action,labels):
            result=super().step(action,labels);mark_history(self,labels,self.history_prefix);return result

    out=Path(out);name=f'{TASK_NAME[task]}_ep{episode}_seed{seed}'
    for folder in ('hdf5_files','gt'): (out/folder).mkdir(parents=True,exist_ok=True)
    h5=out/'hdf5_files'/f'{name}.h5';gt=out/'gt'/f'{name}.jsonl';temporary=h5.with_suffix('.h5.tmp')
    base=gym.make('FruitPick-v1',task=task,robot_uids='rm75b_eg2',obs_mode='rgb+depth+segmentation',
                  control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='cpu')
    execution=NoisyExecution(base,.04,10.,np.random.default_rng([seed,1])) if dart else base
    try:
        obs,_=base.reset(seed=seed);u=base.unwrapped
        with h5py.File(temporary,'w') as f:
            ep=f.create_group(f'episode_{episode}');setup=ep.create_group('setup')
            for key,value in {'task_goal':[u.layout.instruction],'seed':seed,'difficulty':'easy','task':task,
                              'robot':'rm75b_eg2','layout_json':u.layout.to_json()}.items(): setup.create_dataset(key,data=value)
            for cam,key in (('front_camera','front'),('hand_camera','wrist')):
                setup.create_dataset(f'{key}_camera_intrinsic',data=obs['sensor_param'][cam]['intrinsic_cv'][0].cpu().numpy().astype(np.float32))
            solver=None
            rec=(PrefixNoisyRecorder if dart else PrefixRecorder)(execution,lambda q:solver.fk(q),ep)
            rec.history_prefix=history_prefix;rec.observe(obs)
            solver=FruitPickSolver(execution,rec.step,ik_mode=ik_mode,ik_seed=0,ik_candidates=64)
            configure(solver,{'settle_steps':2,'snapshot_dir':snapshot_dir})
            try: ok=solver.run();reason=''
            except PlanningFailure as exc: ok=False;reason=str(exc)
            if ok and not rec.gt[-1]['success']: ok=False;reason='success not reached before last recorded step'
            setup.create_dataset('n_timesteps',data=rec.t)
        labels=getattr(solver,'canonical_labels',{})
        target=labels.get('target');match=re.search(r'fruit (\d+)',reason)
        if match: target=int(match.group(1))
        context={'phase':labels.get('phase'),'target':target,'pick_index':labels.get('pick_index'),
                 'initial_position':u.layout.fruits[target]['pos'] if target is not None else None}
        result={'ok':bool(ok),'reason':reason,'steps':rec.t,'n_fruits':len(u.layout.fruits),'n_pick':u.layout.n_pick,
            'instruction':u.layout.instruction,'solver_version':solver.solver_version,'ik_mode':ik_mode,
            'ik_seed':0,'ik_candidates':64,'settle_steps':2,'history_prefix_steps':30 if history_prefix else 0,
            'planning_audit':solver.trajectory_audit,'q7_all_attempt_steps':solver.canonical_q7,
            'deferred_basket_checks':solver.canonical_basket_checks,'failure_context':None if ok else context,
            'layout_targets':[{'target':i,'position':u.layout.fruits[i]['pos']} for i in u.layout.targets]}
        if dart:
            noisy=[r['noise_tcp_m'] for r in rec.gt if r['phase'] in PHASE_SCALE]
            result.update(noise={'sigma_rad':.04,'tau_steps':10.,'phase_scale':PHASE_SCALE,'kind':'ornstein-uhlenbeck'},
                noise_tcp_m={'median':float(np.median(noisy)) if noisy else None,
                             'p90':float(np.percentile(noisy,90)) if noisy else None,'noisy_steps':len(noisy)})
        atomic_write(gt,''.join(json.dumps(row)+'\n' for row in rec.gt))
        if ok: os.replace(temporary,h5);result.update(h5=h5,gt=gt)
        else:
            failed=h5.with_name(h5.stem+'.failed.h5');os.replace(temporary,failed)
            result.update(failed_h5=failed,failed_gt=gt)
        return result
    finally:
        base.close();temporary.unlink(missing_ok=True)
