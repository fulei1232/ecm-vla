"""Rescue stage3-only prefix and v2 guard; old evaluation/guard code is intact."""
import numpy as np


def guard_v2(proposed,measured,limits):
    proposed=np.asarray(proposed,dtype=np.float64);q=np.asarray(measured,dtype=np.float64)[:7]
    if proposed.shape!=(8,) or q.shape!=(7,) or not np.isfinite(proposed).all() or not np.isfinite(q).all():
        raise ValueError('nonfinite or invalid joint target/state')
    lower=np.asarray(limits['lower']);upper=np.asarray(limits['upper']);delta=np.asarray(limits['v_max'])
    lo=np.maximum(lower,q-delta);hi=np.minimum(upper,q+delta);projection=lo>hi
    target=np.clip(proposed[:7],lower,upper)
    result=proposed.copy();result[:7]=np.where(projection,q+np.clip(target-q,-delta,delta),np.clip(proposed[:7],lo,hi))
    clipped=result[:7]!=proposed[:7]
    return result,{'joint_guard':'v2','guard_clipped':bool(clipped.any()),'guard_clipped_joints':clipped.tolist(),
        'guard_projection':bool(projection.any()),'guard_projection_joints':projection.tolist(),'measured_arm_qpos':q.tolist()}


def install_observe_prefix(evaluator,runner,steps=30):
    """Run fixed holds through the recorder and original loop's observation path."""
    original=evaluator.init_episode
    def init(env_runner,epstate,directory):
        task_goal,video=original(env_runner,epstate,directory)
        state=epstate.get_current_obs()[2]
        fixed=np.r_[np.asarray(state)[:7],1.]
        runner.warming_up=True
        try:
            for _ in range(steps):
                (front,wrist,q),stop,outcome=env_runner.step(fixed.copy())
                if stop or front is None: raise RuntimeError(f'observe prefix ended early: {outcome}')
                epstate.add_observation(front,wrist,q)
                video.record(image=front.copy(),wrist_image=wrist.copy(),state=q.copy(),action=fixed.copy(),is_video_demo=True,subgoal=None)
        finally: runner.warming_up=False
        epstate.exec_start_idx=steps;epstate.count=steps
        return task_goal,video
    evaluator.init_episode=init


def prefix_runner_class():
    from .fruit_eval import FruitEnvRunner
    class PrefixRunner(FruitEnvRunner):
        warming_up=False
        def __init__(self,*args,rescue_guard=None,**kwargs):
            super().__init__(*args,**kwargs);self.rescue_guard=rescue_guard
        def make_env(self,episode_id):
            # Base runner adds1 to max_steps. Stage3's budget is exactly1300.
            budget=self.max_steps;self.max_steps=budget-1
            try: super().make_env(episode_id)
            finally: self.max_steps=budget
        def step(self,action):
            proposed=np.asarray(action).copy();executed=proposed.copy();details=None
            if self.rescue_guard is not None and not self.warming_up:
                q=self.env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
                executed,details=guard_v2(proposed,q,self.rescue_guard)
            result=super().step(executed)
            self.last_execution={'proposed_action':proposed.tolist(),'action':executed.tolist(),
                **(details or {'joint_guard':'off','guard_clipped':False,'guard_clipped_joints':[False]*7}),
                'observe_hold':self.warming_up}
            return result
    return PrefixRunner


def prefix_recorder_class():
    import time
    from .fruit_eval import FruitRawRecorder,snapshot
    from .validation.acm_copy.diagnostic_logging import append,digest,simple
    class PrefixRecorder(FruitRawRecorder):
        def install(self,runner):
            self.runner=runner;original_init,original_step=runner.get_init_obs,runner.step
            self.exec_start=30
            def add_frame(front,wrist,state):
                self.add(front,wrist,state,{'front_sha256':digest(front),'gt':snapshot(runner.env),
                    'timestamp_ns':time.time_ns(),'policy_control_step':self.control_step-30},
                    len(self.rows),self.control_step,self.control_step<30)
            def init():
                result=original_init()
                for frame in zip(result['images'],result['wrist_images'],result['states']): add_frame(*frame)
                self.flush();return result
            def step(action):
                result=original_step(action);self.control_step+=1
                (front,wrist,state),stop,outcome=result
                if front is not None: add_frame(front,wrist,state)
                append(self.directory/'control_actions.jsonl',{'control_step':self.control_step,
                    'action':simple(action),'stop':bool(stop),'outcome':outcome,
                    'raw_frame_id':len(self.rows)-1 if front is not None else None,**runner.last_execution})
                return result
            runner.get_init_obs,runner.step=init,step
    return PrefixRecorder
