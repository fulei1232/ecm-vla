"""R9 supervised records, created only after the R0 gate. No training routines."""
import json
from pathlib import Path

import numpy as np

from v_a_common import read_json, sha256
from fruit_r9_common import OUT


class H5Recorder:
    def __init__(self, directory, env, obs, episode, seed, fk):
        import h5py
        import transforms3d
        assert read_json(OUT / 'R0_gate.json')['passed'], 'No training data before R0 passes'
        self.directory = Path(directory);self.u = env.unwrapped;self.fk = fk
        self.euler = lambda q: np.asarray(transforms3d.euler.quat2euler(q, 'sxyz'))
        self.path = self.directory / 'demo.h5';self.gt_path = self.directory / 'demo_gt.jsonl'
        self.file = h5py.File(self.path, 'x');self.gt = self.gt_path.open('x')
        self.group = self.file.create_group(f'episode_{episode}');setup = self.group.create_group('setup')
        for key, value in {'task_goal': [self.u.layout.instruction], 'seed': seed,
                           'difficulty': 'easy', 'task': self.u.task, 'robot': 'rm75b_eg2',
                           'layout_json': self.u.layout.to_json(), 'task_revision': 9,
                           'distance_signal': 'any_hanging'}.items():setup.create_dataset(key, data=value)
        for camera, label in [('front_camera', 'front'), ('hand_camera', 'wrist')]:
            setup.create_dataset(f'{label}_camera_intrinsic',
                data=obs['sensor_param'][camera]['intrinsic_cv'][0].cpu().numpy().astype(np.float32))
        self.frames = 0;self.wait = 0;self.perturbations = 0;self.finished = False
        self.previous_subgoal = None

    def record(self, obs, clean, labels, offset, executed):
        u = self.u;t = self.frames
        assert u.control_steps == t, 'One record per physical observation; terminal observation included once'
        q = u.agent.robot.get_qpos()[0].cpu().numpy();tcp = u.agent.tcp_pose
        position, quaternion = tcp.p[0].cpu().numpy(), tcp.q[0].cpu().numpy()
        step = self.group.create_group(f'timestep_{t}');o = step.create_group('obs')
        for camera, label in [('front_camera', 'front'), ('hand_camera', 'wrist')]:
            data = obs['sensor_data'][camera];params = obs['sensor_param'][camera]
            o.create_dataset(f'{label}_rgb', data=data['rgb'][0].cpu().numpy().astype(np.uint8), compression='gzip', compression_opts=4)
            o.create_dataset(f'{label}_depth', data=np.clip(data['depth'][0].cpu().numpy(), -32768, 32767).astype(np.int16), compression='gzip', compression_opts=4)
            o.create_dataset(f'{label}_camera_extrinsic', data=params['extrinsic_cv'][0].cpu().numpy().astype(np.float32))
        o.create_dataset('joint_state', data=q[:7].astype(np.float32))
        o.create_dataset('gripper_state', data=q[7:9].astype(np.float64))
        o.create_dataset('eef_state', data=np.r_[position, self.euler(quaternion)].astype(np.float32))
        o.create_dataset('eef_quat', data=quaternion.astype(np.float32))
        o.create_dataset('is_gripper_close', data=bool(clean[7] < 0))
        target = self.fk(np.r_[clean[:7], q[7:]])
        action = step.create_group('action');action.create_dataset('joint_action', data=np.asarray(clean, np.float64))
        action.create_dataset('eef_action', data=np.r_[target.p, self.euler(target.q), clean[7]])
        info = step.create_group('info');subgoal = labels['subgoal']
        for key in ('simple_subgoal', 'grounded_subgoal', 'simple_subgoal_online', 'grounded_subgoal_online'):
            info.create_dataset(key, data=subgoal)
        evaluated = u.evaluate()
        info.create_dataset('is_subgoal_boundary', data=subgoal != self.previous_subgoal)
        info.create_dataset('is_video_demo', data=t < 30)
        info.create_dataset('is_completed', data=bool(evaluated['success'][0]))
        self.previous_subgoal = subgoal
        seg = obs['sensor_data']['front_camera']['segmentation'][0, ..., 0].cpu().numpy()
        positions = u.fruit_positions();inside = u.in_basket();hanging = u.hanging()
        from memory_management.fruit_v2_distance import any_hanging
        noisy = np.r_[clean[:7] + offset, q[7:]]
        noise_tcp = float(np.linalg.norm(self.fk(noisy).p - target.p)) if np.any(offset) else 0.
        row = {'timestep': t, 'phase': labels['phase'], 'subgoal': subgoal, 'target': labels.get('target'),
               'pick_index': labels.get('pick_index'), 'tcp_pos': position.tolist(), 'tcp_quat': quaternion.tolist(),
               'gripper_qpos': q[7:9].tolist(), 'gripper_action': float(clean[7]),
               'fruits': [{'pos': positions[i].tolist(), 'attached': bool(u.attached[i]),
                           'present': bool(u.present[i]), 'wind_detached': bool(u.wind_detached[i]),
                           'in_basket': bool(inside[i]),
                           'front_pixels': int(np.sum(seg == int(u.fruits[i].per_scene_id[0])))}
                          for i in range(len(positions))],
               'occluder_push': list(u.occluder_push), 'n_in_basket': int(inside.sum()),
               'success': bool(evaluated['success'][0]), 'lamp_green': u.lamp_green,
               'fruit_distance': any_hanging(position, positions, hanging), 'distance_signal': 'any_hanging',
               'executed_offset': offset.tolist(), 'noise_tcp_m': noise_tcp,
               'executed_action': None if executed is None else executed.tolist(),
               'terminal_observation': executed is None}
        self.gt.write(json.dumps(row) + '\n')
        self.frames += 1;self.wait += labels['phase'] == 'wait_for_green'
        self.perturbations += bool(np.any(offset))

    def close(self):
        if self.finished:return
        self.group['setup'].create_dataset('n_timesteps', data=self.frames)
        self.gt.close();self.file.close();self.finished = True

    def summary(self):
        assert self.finished
        return {'h5': self.path.name, 'gt': self.gt_path.name, 'h5_sha256': sha256(self.path),
                'gt_sha256': sha256(self.gt_path), 'data_frames': self.frames, 'history_prefix_steps': 30,
                'wait_supervised_frames': self.wait, 'dart_perturbations': self.perturbations,
                'terminal_observation_included': True,
                'DART_recipe': 'OU sigma=.04rad tau=10, reach/pull/carry full, approach half; additive execution perturbation after the guarded clean command; clean expert target is label.'}
