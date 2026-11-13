#!/usr/bin/env python3
"""Recorded-command A/B prefix and full native ep112 diagnostic; no training demos."""
import hashlib
from pathlib import Path
import sys

import numpy as np

from v_a_common import ECM, atomic_json, read_json, read_jsonl, timestamp
from fruit_shield_v2_common import OUT, PREVIOUS, OLD_SHIELD, activate, proof


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prefix(mode, reference, baseline=None):
    import gymnasium as gym
    import fruit_picking.count_shield_v2_env  # noqa: F401
    from fruit_picking.task_v2_audit_r8 import pixel_counts
    from fruit_shield_v2_runtime import scene_manifest, save_forensics
    from PIL import Image
    d = OUT/'diagnostics'/('ep112_original_prefix' if mode == 'off' else 'ep112_shield_prefix')
    d.mkdir(parents=True)
    env = gym.make('FruitPickCountShieldV2-v0', task='count', task_version='v2', task_revision=9,
        count_shield=mode, robot_uids='rm75b_eg2', obs_mode='rgb+depth+segmentation',
        control_mode='pd_joint_pos', render_mode='rgb_array', sim_backend='cpu', max_episode_steps=2800)
    frames = []; first_visible = None
    try:
        obs, _ = env.reset(seed=4101792);u = env.unwrapped
        manifest = scene_manifest(u);atomic_json(d/'scene_manifest.json', manifest)
        for t in range(579):
            positions = u.fruit_positions();q = u.agent.robot.get_qpos()[0].cpu().numpy()
            counts = pixel_counts(obs);basket = int(u.basket.per_scene_id[0])
            visible = {cam:{'basket':c.get(basket, 0),
                'basket_fruit':sum(c.get(int(u.fruits[int(i)].per_scene_id[0]),0) for i in np.flatnonzero(u.in_basket())),
                'inactive_reserve':sum(c.get(int(u.fruits[int(i)].per_scene_id[0]),0) for i in np.flatnonzero(~u.present))}
                for cam,c in counts.items()}
            if any(v['basket'] or v['basket_fruit'] for v in visible.values()) and first_visible is None:
                first_visible = t
            front = obs['sensor_data']['front_camera'];seg = front['segmentation'][0,...,0].cpu().numpy()
            tree_ids = [int(u.fruits[int(i)].per_scene_id[0]) for i in np.flatnonzero(u.hanging())]
            tree_mask = np.isin(seg, tree_ids)
            rgb = front['rgb'][0].cpu().numpy()
            source_frame=reference[min(t,577)];target=source_frame.get('target')
            target_pixels=counts['hand_camera'].get(int(u.fruits[int(target)].per_scene_id[0]),0) if target is not None else 0
            outside_port=bool(target is not None and positions[target,2]-u.layout.fruits[target]['radius']>.376)
            row = {'step':t, 'qpos':q.tolist(), 'positions':positions.tolist(),
                'tcp_position':u.agent.tcp_pose.p[0].cpu().numpy().tolist(), 'visibility':visible,
                'front_tree_mask_sha256':digest(tree_mask), 'front_tree_rgb_sha256':digest(rgb[tree_mask]),
                'lamp_front_pixels':counts['front_camera'].get(int(u.work_lamp.per_scene_id[0]),0),
                'hand_target_pixels':target_pixels,'target_outside_port':outside_port,'phase':source_frame['phase']}
            if t < 578:
                old = reference[t]
                row['original_tcp_max_abs_diff'] = float(np.max(np.abs(np.asarray(row['tcp_position'])-old['tcp_pos'])))
                row['original_fruit_max_abs_diff'] = float(np.max(np.abs(positions-np.asarray([f['pos'] for f in old['fruits']]))))
            frames.append(row)
            if t in (0,578):
                save_forensics(d, obs, u, f'step{t}')
                for cam,data in obs['sensor_data'].items():
                    Image.fromarray(data['rgb'][0].cpu().numpy()).save(d/f'step{t}_{cam}.png')
            if mode != 'off':
                assert not any(v[k] for v in visible.values() for k in v), f'Shield prefix visibility at {t}'
                assert row['lamp_front_pixels'] == baseline[t]['lamp_front_pixels']
                assert row['front_tree_mask_sha256'] == baseline[t]['front_tree_mask_sha256'], f'Tree occluded at {t}'
                assert row['front_tree_rgb_sha256'] == baseline[t]['front_tree_rgb_sha256'], f'Tree appearance changed at {t}'
                if outside_port and source_frame['phase'] in ('grasp','pull','carry') and baseline[t]['hand_target_pixels']>=16:
                    assert target_pixels==baseline[t]['hand_target_pixels'], f'Grasp/carry target occluded at {t}'
            if t == 578:break
            obs, _, terminated, truncated, _ = env.step(np.asarray(reference[t]['executed_action']))
            assert not bool(terminated[0]) and not bool(truncated[0])
        with (d/'physical_and_visibility.jsonl').open('w') as stream:
            for row in frames:stream.write(__import__('json').dumps(row)+'\n')
        if mode == 'off':
            assert first_visible == 578, f'Original failure not reproduced: {first_visible}'
            assert frames[-1]['visibility']['hand_camera']['basket'] > 0
        else:
            assert first_visible is None
        return frames, manifest
    except Exception as exc:
        save_forensics(d,obs,u,'check_failure')
        for cam,data in obs['sensor_data'].items():
            Image.fromarray(data['rgb'][0].cpu().numpy()).save(d/f'check_failure_{cam}.png')
        atomic_json(d/'check_failure.json',{'step':u.control_steps,'error':str(exc),'mode':mode})
        with (d/'physical_and_visibility.jsonl').open('w') as stream:
            for row in frames:stream.write(__import__('json').dumps(row)+'\n')
        raise
    finally:env.close()


def native_full(reference):
    import gymnasium as gym
    from fruit_shield_v2_runtime import SeamReplay as ShieldReplay, save_forensics
    from fruit_r9_episode_support import EpisodeEnd, TargetGone
    from fruit_picking.solver import FruitPickSolver, PlanningFailure
    from fruit_picking.canonical_solver import configure
    d = OUT/'diagnostics/ep112_shield_full';d.mkdir(parents=True)
    env = gym.make('FruitPickCountShieldV2-v0', task='count', task_version='v2', task_revision=9,
        count_shield='fixed_v2', robot_uids='rm75b_eg2', obs_mode='rgb+depth+segmentation',
        control_mode='pd_joint_pos', render_mode='rgb_array', sim_backend='cpu', max_episode_steps=2800)
    replay = None
    try:
        obs, _ = env.reset(seed=4101792);u=env.unwrapped
        replay=ShieldReplay(env,obs,d,'history',read_json(ECM/'results/fruit_k0v2_gate_20261102/guard_v2.json'))
        solver=FruitPickSolver(env,replay.step,ik_mode='canonical_v31',ik_seed=0,ik_candidates=64)
        configure(solver,{'settle_steps':2});replay.solver_fk=solver.fk
        replay.noise=np.zeros(7);replay.noise_rng=np.random.default_rng([4101792,1])
        fixed=np.r_[u.agent.robot.get_qpos()[0].cpu().numpy()[:7],1.]
        try:
            solver.hold(30,{'phase':'observe','target':None,'subgoal':'observe initial scene'})
            while replay.provided()['lamp_green'] is not True:
                solver.step_fn(fixed.copy(),{'phase':'wait_for_green','target':None,'subgoal':'wait at rest for green light'})
            while not replay.ended:
                slot=replay.choose()
                if slot is None:
                    solver.gripper=1.;solver.refresh_obstacles()
                    solver.move(solver.home,{'phase':'carry','target':0,'subgoal':'return to rest to finish'})
                    solver.hold(2800,{'phase':'finish','target':None,'subgoal':'hold open at rest until termination'})
                else:
                    actor=u.hanging_slot[slot]
                    if actor is None:
                        solver.hold(1,{'phase':'inspect','target':None,'subgoal':'observe current tree occupancy'});continue
                    try:
                        solver.pick(actor,0,u.layout.n_pick)
                        replay.releases.append({'step':u.control_steps,'actor':actor,'slot':slot,'event':'own_release_attempt'})
                        while actor not in u.replenished:
                            solver.hold(1,{'phase':'await_replenishment','target':None,'subgoal':'confirm current placement and replenishment'})
                    except TargetGone:
                        solver.hold(1,{'phase':'inspect','target':None,'subgoal':'observe wind-lost current target'})
        except EpisodeEnd:pass
        except PlanningFailure as exc:
            atomic_json(d/'planning_failure.json',{'reason':str(exc),'step':u.control_steps});raise
        atomic_json(d/'actions.json',replay.actions);atomic_json(d/'decisions.json',replay.decisions)
        equal_noise = sum(a['phase']==reference[i]['phase'] and np.array_equal(a['noise_offset'],reference[i]['executed_offset'])
                          for i,a in enumerate(replay.actions[:len(reference)]))
        value={'passed':bool(replay.success),'success':bool(replay.success),'steps':u.control_steps,
            'seed':4101792,'N':u.layout.n_pick,'attempt':0,'outcome':replay.reason,
            'noise_generator':'unchanged OU, default_rng([4101792,1]), sigma=.04 tau=10',
            'original_noise_phase_matches':equal_noise,'original_prefix_actions':len(reference),
            'max_visibility':replay.max_visibility,'shield_intersections_checked':replay.shield_intersections_checked,
            'training_data_generated':False,'HDF5_recording':False}
        atomic_json(d/'result.json',value)
        assert value['passed'],'Full original-seed DART diagnostic did not succeed'
        return value
    finally:
        if replay is not None:
            atomic_json(d/'actions.json',replay.actions);atomic_json(d/'decisions.json',replay.decisions)
            replay.trace.close()
        env.close()


def main():
    activate()
    from fruit_shield_v2_runtime import install
    install()
    source=PREVIOUS/'replays/train_dart/count/history/ep112_attempt0'
    reference=read_jsonl(source/'demo_gt.jsonl');assert len(reference)==578
    assert all(r['timestep']==i and r['executed_action'] is not None for i,r in enumerate(reference))
    before,bm=prefix('off',reference)
    after,am=prefix('fixed_v2',reference,before)
    old_actors=bm['actors'];new_actors={k:v for k,v in am['actors'].items() if k!='count_fixed_optical_shield'}
    assert {k:v['collision_shapes'] for k,v in old_actors.items()} == {k:v['collision_shapes'] for k,v in new_actors.items()}
    assert am['actors']['count_fixed_optical_shield']['collision_shapes']==0
    physical_diff={key:max(float(np.max(np.abs(np.asarray(a[key])-np.asarray(b[key])))) for a,b in zip(before,after))
                   for key in ('qpos','tcp_position','positions')}
    # Replay determinism remains descriptive, as in the approved expert audits.
    value={'passed':True,'created_utc':timestamp(),'original_failure_reproduced_step':578,
        'recorded_commands_reused_exactly':578,'physics_collision_shapes_unchanged':True,
        'actor_segmentation_IDs_equal_report_only':old_actors==new_actors,
        'physical_max_abs_difference_report_only':physical_diff,'front_tree_RGB_and_mask_preserved':True,
        'lamp_pixel_counts_preserved':True,'shield_prefix_zero_visibility':True,
        'hand_target_visibility_preserved_outside_port':True,
        'hand_target_visibility_check_frames':sum(r['target_outside_port'] and r['phase'] in ('grasp','pull','carry')
            and r['hand_target_pixels']>=16 for r in before),
        'native_full':native_full(reference),'original_failure_retained':str(source),
        'training_data_generated':False,'observation_condition':'count_fixed_shield_v2'}
    prior=read_json(OLD_SHIELD/'diagnostics/ep112_shield_full/actions.json')
    current=read_json(OUT/'diagnostics/ep112_shield_full/actions.json')
    matched=sum(a['phase']==b['phase'] and np.array_equal(a['noise_offset'],b['noise_offset'])
        and np.array_equal(a['executed'],b['executed']) for a,b in zip(prior,current))
    assert matched==len(prior)==581,'The previously failing 581-command stream changed'
    value.update(previous_581_commands_exact_matches=matched,
        old_shield_failure_retained=str(OLD_SHIELD/'diagnostics/ep112_shield_full/check_failure.json'))
    proof('ep112_diagnostic.json',value)
    print(value,flush=True)


if __name__=='__main__':main()
