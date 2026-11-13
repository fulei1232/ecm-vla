#!/usr/bin/env python3
"""R0 diagnostic replay only: no HDF5, training examples, model or val rollout."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from v_a_common import ECM, atomic_json, timestamp, sha256
sys.path[:0] = [str(ECM), str(ECM/'src')]
from fruit_picking.task_v2_spec import choose_count
from memory_management.fruit_rescue_platform import guard_v2


class EpisodeEnd(Exception):
    pass


class TargetGone(Exception):
    pass


def visible_view(u, obs):
    """Only visible objects, their observable geometry, and currently available color labels."""
    records = []
    for i, actor in enumerate(u.fruits):
        identity = int(actor.per_scene_id[0])
        pixels = {camera: int((obs['sensor_data'][camera]['segmentation'][0, ..., 0].cpu().numpy() == identity).sum())
            for camera in ('front_camera', 'hand_camera')}
        pos = u.fruit_positions()[i]
        on_tree_geometry = np.linalg.norm(pos-np.asarray(u.layout.fruits[i]['pos'])) < .03
        if max(pixels.values()) < u.v2_spec.visible_pixels or not on_tree_geometry:
            continue
        color = bool(u.layout.fruits[i]['ripe']) if u.task == 'count' or u.v2_color_alpha == 0 else None
        marker = False
        for occ, cluster in zip(u.layout.occluders, u.occluders):
            if occ['fruit'] == i:
                marker = any(int((obs['sensor_data'][c]['segmentation'][0, ..., 0].cpu().numpy()
                    == int(cluster.per_scene_id[0])).sum()) >= u.v2_spec.visible_pixels
                    for c in ('front_camera', 'hand_camera'))
        records.append({'id': i, 'position': pos.tolist(), 'ripe_visible': color, 'pixels': pixels,
            'leaf_marker': marker})
    return records


def visible_basket_count(u, obs):
    count = 0
    for actor, inside in zip(u.fruits, u.in_basket()):
        if inside and any(int((obs['sensor_data'][c]['segmentation'][0, ..., 0].cpu().numpy()
                == int(actor.per_scene_id[0])).sum()) >= u.v2_spec.visible_pixels
                for c in ('front_camera', 'hand_camera')):
            count += 1
    return count


class AuditReplay:
    def __init__(self, env, obs, directory, level, guard):
        self.env, self.u, self.obs = env, env.unwrapped, obs
        self.directory, self.level, self.guard = directory, level, guard
        self.ledger, self.previous_inside, self.release_history = [], set(), []
        self.actions, self.decisions = [], []
        self.snapshots, self.basket_seen = {}, {c: [] for c in ('front_camera', 'hand_camera')}
        self.trace = (directory/'trace.jsonl').open('x')
        self.initial = visible_view(self.u, obs)
        self.ended, self.success, self.reason = False, False, ''
        self.record('opening')

    def record(self, phase):
        u, obs = self.u, self.obs
        inside = {int(i) for i in np.flatnonzero(u.in_basket()) if not u.wind_detached[i]}
        for i in sorted(inside-self.previous_inside):
            self.ledger.append({'step': u.control_steps, 'fruit': i, 'event': 'robot_basket_entry'})
        self.previous_inside = inside
        basket_pixels = {}
        for camera in self.basket_seen:
            seg = obs['sensor_data'][camera]['segmentation'][0, ..., 0].cpu().numpy()
            n = int((seg == int(u.basket.per_scene_id[0])).sum())
            basket_pixels[camera] = n
            if n:
                self.basket_seen[camera].append({'step': u.control_steps, 'pixels': n, 'phase': phase})
        stage = 'opening' if u.control_steps == 0 else 'post_gust' if u.control_steps == 20 else 'release' if phase == 'release' else None
        if stage:self.capture(stage)
        if u.task == 'occluded_ripe' and u.control_steps >= 20:
            for items in u.v2_materials:
                for material, _ in items:
                    if not np.allclose(material.base_color[:3], u.v2_spec.neutral_rgb, atol=1e-7, rtol=0):
                        raise RuntimeError('SEMANTICS_MISMATCH: nonneutral fruit material after step20')
        info = u.evaluate()
        self.trace.write(json.dumps({'physical_step': u.control_steps, 'phase': phase,
            'qpos': u.agent.robot.get_qpos()[0].cpu().numpy().tolist(),
            'fruit_positions': u.fruit_positions().tolist(), 'robot_or_wind_detached': list(u.attached),
            'wind_detached': list(u.wind_detached), 'wind_events': list(u.wind_events),
            'in_basket': u.in_basket().tolist(), 'basket_pixels': basket_pixels,
            'visible': visible_view(u, obs), 'color_alpha': u.v2_color_alpha,
            'finish_streak': u.finish_streak, 'success': bool(info['success'][0]),
            'fail': bool(info['fail'][0]), 'reason': info.get('v2_reason')})+'\n')

    def capture(self, stage):
        from PIL import Image
        if stage in self.snapshots:return
        files, pixels = {}, {}
        for camera in self.basket_seen:
            image = self.obs['sensor_data'][camera]['rgb'][0].cpu().numpy().astype(np.uint8)
            name = f'{stage}_{camera}.png';Image.fromarray(image).save(self.directory/name)
            files[camera] = name
            seg = self.obs['sensor_data'][camera]['segmentation'][0, ..., 0].cpu().numpy()
            pixels[camera] = int((seg == int(self.u.basket.per_scene_id[0])).sum())
        self.snapshots[stage] = {'step': self.u.control_steps, 'files': files, 'basket_pixels': pixels}

    def step(self, action, labels):
        u = self.u
        proposed = np.asarray(action, dtype=float)
        if labels['phase'] == 'grasp':self.capture('before_grasp')
        executed, clipping = guard_v2(proposed, u.agent.robot.get_qpos()[0].cpu().numpy(), self.guard)
        before = u.control_steps
        obs, _, terminated, truncated, info = self.env.step(executed)
        self.obs = obs
        self.actions.append({'step': u.control_steps, 'execution_block16': max(0, before-30)//16,
            'phase': labels['phase'], 'target': labels.get('target'), 'proposed': proposed.tolist(),
            'executed': executed.tolist(), **clipping})
        self.record(labels['phase'])
        if bool(terminated[0]) or bool(truncated[0]):
            self.ended, self.success = True, bool(info['success'][0])
            self.reason = info.get('v2_reason') or ('success' if self.success else 'original_task_fail_or_budget')
            raise EpisodeEnd()
        target = labels.get('target')
        if target is not None and labels['phase'] in ('reach', 'approach') and u.wind_detached[target]:
            raise TargetGone()
        return obs, info

    def choose(self):
        u, level = self.u, self.level
        current = visible_view(u, self.obs)
        initial = self.initial if level != 'current' else None
        if level == 'history':
            current = [{'id': i, 'position': u.fruit_positions()[i].tolist(), 'ripe_visible': bool(f['ripe'])}
                for i, f in enumerate(u.layout.fruits) if not u.attached[i]]
        provided = {'current_tree': current, 'N': u.layout.n_pick if u.task == 'count' else None,
            'frame0_tree': initial, 'delivery_events': self.release_history.copy() if level == 'history' else None,
            'visible_basket_count': visible_basket_count(u, self.obs)}
        if u.task == 'count':
            chosen = choose_count(level, u.layout.n_pick, current, initial,
                len(self.release_history) if level == 'history' else None, provided['visible_basket_count'])
        elif level == 'history':
            candidates = [f for f in current if f['ripe_visible']]
            chosen = max(candidates, key=lambda f: f['position'][1])['id'] if candidates else None
        elif level == 'initial_current':
            ripe_ids = {f['id'] for f in initial if f['ripe_visible']}
            candidates = [f for f in current if f['id'] in ripe_ids]
            chosen = max(candidates, key=lambda f: f['position'][1])['id'] if candidates else None
        else:
            candidates = [f for f in current if f['ripe_visible'] is True or f['leaf_marker']]
            chosen = max(candidates, key=lambda f: f['position'][1])['id'] if candidates else None
        if level == 'current':
            assert provided['frame0_tree'] is None and provided['delivery_events'] is None
        elif level == 'initial_current':
            assert provided['delivery_events'] is None
        self.decisions.append({'decision': len(self.decisions), 'physical_step': u.control_steps,
            'level': level, 'provided': provided, 'chosen_fruit': chosen})
        return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=('count', 'occluded_ripe'), required=True)
    p.add_argument('--level', choices=('current', 'initial_current', 'history'), required=True)
    p.add_argument('--episode', type=int, required=True);p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--task-version', choices=('v1', 'v2'), default='v1')
    args = p.parse_args()
    if args.task_version != 'v2':p.error('R0 replay requires explicit --task-version v2; default leaves original behavior enabled')
    if not 200 <= args.episode <= 249:p.error('R0 uses only dev200–249; train and val are forbidden')
    if args.seed != 4200000+16*args.episode:p.error('R0 requires the registered new dev seed')
    allowed=ECM/'results/fruit_task_v2_R0_20261105/replays'
    args.out=args.out.resolve()
    if not args.out.is_relative_to(allowed):p.error('Write only under the independent R0 diagnostic replay directory')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'result.json').exists():p.error('Do not overwrite an existing R0 attempt')
    import gymnasium as gym
    import fruit_picking.task_v2_env  # noqa: F401
    from fruit_picking.solver import FruitPickSolver, PlanningFailure
    from fruit_picking.canonical_solver import configure
    guard_path = ECM/'results/fruit_k0v2_gate_20261102/guard_v2.json'
    guard = json.loads(guard_path.read_text())
    assert guard['version'] == 'v2' and all(k in guard for k in ('lower', 'upper', 'v_max'))
    env = gym.make('FruitPickTaskV2-v0', task=args.task, task_version=args.task_version, robot_uids='rm75b_eg2',
        obs_mode='rgb+depth+segmentation', control_mode='pd_joint_pos', render_mode='rgb_array', sim_backend='cpu', max_episode_steps=1300)
    replay = None;engineering = None;failure = ''
    try:
        obs, _ = env.reset(seed=args.seed)
        replay = AuditReplay(env, obs, args.out, args.level, guard)
        solver = FruitPickSolver(env, replay.step, ik_mode='canonical_v31', ik_seed=0, ik_candidates=64)
        configure(solver, {'settle_steps': 2})
        try:
            solver.hold(30, {'phase': 'observe', 'target': None, 'subgoal': 'initial observation'})
            while not replay.ended:
                target = replay.choose()
                if target is None:
                    solver.gripper = 1.
                    solver.refresh_obstacles()
                    labels = {'phase': 'carry', 'target': 0, 'subgoal': 'return to rest to finish'}
                    solver.move(solver.home, labels)
                    # For occluded task this is a timeout, since its termination remains unchanged.
                    solver.hold(1300, {'phase': 'finish', 'target': None, 'subgoal': 'hold still with open gripper'})
                else:
                    try:
                        solver.pick(target, 0, env.unwrapped.layout.n_pick)
                        replay.release_history.append({'event': 'own_release_attempt', 'fruit': target,
                            'step': env.unwrapped.control_steps})
                    except TargetGone:
                        # A wind-lost target cancels the current skill, without providing any count to the chooser.
                        solver.hold(1, {'phase': 'inspect', 'target': None, 'subgoal': 'observe changed target position'})
        except EpisodeEnd:
            pass
        except PlanningFailure as exc:
            failure = str(exc);replay.reason = 'planning_or_grasp_failure'
        actions = [a for a in replay.actions if a['step'] > 30]
        clipped = sum(a['guard_clipped'] for a in actions)
        u=env.unwrapped;inside=u.in_basket();robot=np.asarray(u.attached)&~np.asarray(u.wind_detached)
        dropped=robot&~inside&(u.fruit_positions()[:,2]<.06)
        final_basket=int((inside&~np.asarray(u.wind_detached)).sum())
        mode=None
        if not replay.success:
            if failure:mode='planning_or_grasp_failure'
            elif dropped.any():mode='robot_drop'
            elif args.task=='count':
                mode=('under_count_stop' if replay.reason=='finish_signal' and final_basket<u.layout.n_pick
                    else 'budget_under_count' if replay.reason=='budget' and final_basket<u.layout.n_pick
                    else replay.reason)
            elif any(inside[i] and not f['ripe'] for i,f in enumerate(u.layout.fruits)):mode='unripe_in_basket'
            elif u.control_steps>=1300:mode='timeout_incomplete_ripe_targets'
            else:mode='other_original_task_failure'
        result = {'task': args.task, 'task_version': args.task_version, 'level': args.level,
            'episode': args.episode, 'seed': args.seed, 'success': replay.success, 'outcome': replay.reason,
            'planning_failure': failure, 'engineering_error': engineering, 'steps': env.unwrapped.control_steps,
            'failure_mode':mode,'final_robot_in_basket':final_basket,'final_robot_picked':int(robot.sum()),
            'failure_context':None if replay.success else getattr(solver,'canonical_labels',{}),
            'guard_v2_sha256': sha256(guard_path), 'guard_clipped_steps': clipped,
            'guard_checked_steps': len(actions), 'guard_clip_rate': clipped/len(actions) if actions else None,
            'snapshots': replay.snapshots, 'basket_seen': replay.basket_seen,
            'wind_plan': getattr(env.unwrapped, 'v2_wind_plan', None), 'wind_events': env.unwrapped.wind_events,
            'robot_basket_entry_history': replay.ledger, 'own_release_history': replay.release_history,
            'layout': json.loads(env.unwrapped.layout.to_json()),
            'spec': env.unwrapped.v2_spec.to_dict(), 'solver_version': solver.solver_version,
            'decisions': len(replay.decisions), 'completed_utc': timestamp(), 'diagnostic_only': True}
        atomic_json(args.out/'actions.json', replay.actions);atomic_json(args.out/'decisions.json', replay.decisions)
        atomic_json(args.out/'result.json', result)
        print({k: result[k] for k in ('task','level','episode','success','outcome','steps','guard_clip_rate')}, flush=True)
    finally:
        if replay is not None:replay.trace.close()
        env.close()


if __name__ == '__main__':main()
