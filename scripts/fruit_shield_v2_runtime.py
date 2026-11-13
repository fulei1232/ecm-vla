"""Continuous-liner runtime; visibility rules unchanged and camera forensics complete."""
import numpy as np

from v_a_common import atomic_json
from fruit_shield_runtime import ShieldReplay as OriginalShieldReplay, ShieldRecorder
from fruit_shield_runtime import scene_manifest as original_manifest, save_forensics as original_forensics
from fruit_r9_episode_support import R9Replay
from fruit_picking.count_shield_spec import sphere_intersections


def scene_manifest(u):
    result=original_manifest(u)
    result.update(shield_new_obstacle_audit_boxes=getattr(u,'count_shield_audit_boxes',[]),
        basket_decoration_shifts=getattr(u,'basket_decoration_shifts',[]),
        basket_collision_signature=getattr(u,'basket_collision_signature',None),
        audit_geometry='Render volume minus pre-existing basket collision volume; normal basket contact is not a new optical obstruction.')
    return result


def save_forensics(directory,obs,u,stage):
    from pathlib import Path
    directory=Path(directory)
    original_forensics(directory,obs,u,stage)
    arrays={}
    for camera,params in obs['sensor_param'].items():
        for name in ('intrinsic_cv','extrinsic_cv','cam2world_gl'):
            if name in params:arrays[camera+'_'+name]=params[name][0].cpu().numpy()
    np.savez_compressed(directory/(stage+'_camera_parameters.npz'),**arrays)
    atomic_json(directory/(stage+'_physical_state.json'),{'step':u.control_steps,
        'qpos':u.agent.robot.get_qpos()[0].cpu().numpy().tolist(),
        'qvel':u.agent.robot.get_qvel()[0].cpu().numpy().tolist(),
        'tcp_position':u.agent.tcp_pose.p[0].cpu().numpy().tolist(),
        'tcp_quaternion':u.agent.tcp_pose.q[0].cpu().numpy().tolist()})


class SeamReplay(OriginalShieldReplay):
    def record(self,phase):
        R9Replay.record(self,phase)
        boxes=getattr(self.u,'count_shield_audit_boxes',[])
        if boxes:
            ids=np.flatnonzero(self.u.present)
            hit=sphere_intersections(self.u.fruit_positions()[ids],[self.u.layout.fruits[int(i)]['radius'] for i in ids],boxes)
            self.shield_intersections_checked+=1
            if np.any(hit):self.fail('FRUIT_INTERSECTS_FIXED_SHIELD',{'actors':ids[hit].tolist()})


def install():
    import gymnasium as gym
    import fruit_picking.count_shield_v2_env  # noqa: F401
    import fruit_shield_runtime as old
    make_original=gym.make
    def make(env_id,*args,**kwargs):
        if kwargs.get('task')=='count' and (env_id=='FruitPickTaskV2-v0'
                or env_id=='FruitPickCountShield-v0' and kwargs.get('count_shield')=='fixed_v1'):
            env_id='FruitPickCountShieldV2-v0';kwargs['count_shield']='fixed_v2'
        return make_original(env_id,*args,**kwargs)
    gym.make=make
    old.scene_manifest=scene_manifest;old.save_forensics=save_forensics;old.ShieldReplay=SeamReplay
