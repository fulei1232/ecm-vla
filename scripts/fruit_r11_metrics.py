"""Native segmentation and exact finger-box distances; descriptive only."""
import json
import xml.etree.ElementTree as ET

import numpy as np


class GripperDistance:
    def __init__(self, boxes, urdf_path):
        self.boxes = boxes
        if not boxes:
            return
        from mplib.pymp.collision_detection import fcl
        self.fcl = fcl
        self.request = fcl.DistanceRequest()
        self.objects = [fcl.CollisionObject(fcl.Box(2*np.asarray(b['half'])), b['center']) for b in boxes]
        self.centers = np.asarray([b['center'] for b in boxes])
        self.halves = np.asarray([b['half'] for b in boxes])
        root = ET.parse(urdf_path).getroot()
        self.fingers = []
        for name, attr in (('left_finger', 'finger1_link'), ('right_finger', 'finger2_link')):
            link = root.find(f"link[@name='{name}']")
            collision = link.find('collision')
            origin = collision.find('origin')
            assert np.array_equal(np.fromstring(origin.get('rpy'), sep=' '), [0, 0, 0])
            center = np.fromstring(origin.get('xyz'), sep=' ')
            side = np.fromstring(collision.find('geometry/box').get('size'), sep=' ')
            # These collision boxes are also the actual visible finger boxes.
            assert link.find('visual/geometry/box').get('size') == collision.find('geometry/box').get('size')
            assert link.find('visual/origin').attrib == origin.attrib
            self.fingers.append((attr, center, fcl.Box(side)))

    def measure(self, agent):
        if not self.boxes:
            return {'entered': False, 'finger_entered': False, 'tcp_entered': False,
                    'minimum_distance_m': None, 'finger_distance_m': None, 'tcp_distance_m': None}
        tcp = agent.tcp_pose.p[0].cpu().numpy()
        delta = np.maximum(np.abs(tcp-self.centers)-self.halves, 0.)
        tcp_distance = float(np.linalg.norm(delta, axis=1).min())
        finger_distance = float('inf')
        for attr, center, geometry in self.fingers:
            pose = getattr(agent, attr).pose
            matrix = pose.to_transformation_matrix()[0].cpu().numpy()
            position = matrix[:3, :3]@center + matrix[:3, 3]
            obj = self.fcl.CollisionObject(geometry, position, pose.q[0].cpu().numpy())
            for shield in self.objects:
                distance = self.fcl.distance(obj, shield, self.request).min_distance
                assert np.isfinite(distance)
                finger_distance = min(finger_distance, max(0., float(distance)))
        return {'entered': tcp_distance == 0. or finger_distance == 0.,
                'finger_entered': finger_distance == 0., 'tcp_entered': tcp_distance == 0.,
                'minimum_distance_m': min(tcp_distance, finger_distance),
                'finger_distance_m': finger_distance, 'tcp_distance_m': tcp_distance}


class FrameMetrics:
    def __init__(self, directory, distance):
        self.directory, self.distance = directory, distance
        self.frames = 0
        self.visible_frames = 0
        self.maximum_pixels = 0
        self.total_pixels = 0
        self.camera = {}
        self.entered = self.finger_entered = self.tcp_entered = 0
        self.minimum_distance = None
        self.peak = None
        self.last_step = None
        self.log = (directory/'frame_metrics.jsonl').open('x')

    def observe(self, replay, visibility):
        step = replay.u.control_steps
        if step == self.last_step:
            return
        self.last_step = step
        gripper = self.distance.measure(replay.u.agent)
        self.frames += 1
        self.visible_frames += any(v['basket'] for v in visibility.values())
        peak_camera, peak_pixels = max(((c, v['basket']) for c, v in visibility.items()), key=lambda x:x[1])
        for camera, v in visibility.items():
            row = self.camera.setdefault(camera, {'body_visible_frames': 0, 'total_body_pixels': 0, 'max_body_pixels': 0,
                                                 'max_basket_fruit_pixels': 0, 'max_inactive_reserve_pixels': 0})
            row['body_visible_frames'] += bool(v['basket'])
            row['total_body_pixels'] += v['basket']
            row['max_body_pixels'] = max(row['max_body_pixels'], v['basket'])
            for name in ('basket_fruit', 'inactive_reserve'):
                row['max_'+name+'_pixels'] = max(row['max_'+name+'_pixels'], v[name])
        self.total_pixels += sum(v['basket'] for v in visibility.values())
        if peak_pixels > self.maximum_pixels:
            self.maximum_pixels = peak_pixels
            replay.capture('basket_body_peak', replace=True)
            self.peak = {'step': step, 'camera': peak_camera, 'pixels': peak_pixels,
                         'files': replay.snapshots['basket_body_peak']['files']}
        self.entered += gripper['entered']
        self.finger_entered += gripper['finger_entered']
        self.tcp_entered += gripper['tcp_entered']
        d = gripper['minimum_distance_m']
        if d is not None:
            self.minimum_distance = d if self.minimum_distance is None else min(self.minimum_distance, d)
        self.log.write(json.dumps({'step': step, 'visibility': visibility, 'gripper': gripper})+'\n')

    def summary(self):
        return {'observations': self.frames, 'basket_body': {
                    'visible_frames_either_camera': self.visible_frames, 'maximum_single_camera_pixels': self.maximum_pixels,
                    'total_pixels_both_cameras': self.total_pixels, 'by_camera': self.camera, 'peak_screenshot': self.peak},
                'gripper_shield': {'entered_frames': self.entered, 'finger_entered_frames': self.finger_entered,
                    'tcp_entered_frames': self.tcp_entered, 'minimum_distance_m': self.minimum_distance,
                    'report_only': True, 'geometry': 'Full 14 rendered shield boxes; native URDF visible/collision finger OBBs and TCP point',
                    'distance': 'Nonnegative Euclidean separation; intersection/contact = 0; no shield = null',
                    'observations_include_initial_and_terminal': True}}
