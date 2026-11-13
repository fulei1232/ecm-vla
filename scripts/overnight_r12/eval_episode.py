#!/usr/bin/env python3
"""Closed-loop dev episode for the overnight R12 diagnosis (exploratory).

Runs one episode of the R12 task definition (Count: FruitPickCountShieldV2 with fixed_v2 shield + R12 false bottom,
budget 2800; Occluded: FruitPickTaskV2 budget 1300) against a policy server, with the official MME-VLA client loop
(16-step execution, history buffer). 30 observe steps are history-only, as in training.

  --memory uniform|ka|none  how the server selects memory (sent to the server via the context file)
  --oracle-count            append " fruits already in the basket: k." (simulator truth) to the prompt (OR arm)
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ECM = Path('/media/fulei/fuleivla/ecm_vla')
POLICY = Path('/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning')
sys.path[:0] = [str(POLICY / 'examples/robomme'), str(POLICY / 'packages/openpi-client/src'),
                str(ECM), str(ECM / 'src'), str(ECM / 'scripts')]
DEV_SEED_BASE = 4200000
GUARD = json.loads((ECM / 'results/fruit_k0v2_gate_20261102/guard_v2.json').read_text())  # D5 platform


def make_env(task):
    import gymnasium as gym
    import fruit_picking.task_v2_env  # noqa: F401
    budget = 2800 if task == 'count' else 1300
    if task == 'count':
        from fruit_picking.count_shield_v2_buildfix import install as install_constructor
        from fruit_shield_v2_runtime import install as install_v2
        from fruit_picking.count_shield_r12_visual import install as install_false_bottom
        install_v2(); install_constructor(); install_false_bottom()
    env = gym.make('FruitPickTaskV2-v0', task=task, task_version='v2', task_revision=9,
                   robot_uids='rm75b_eg2', obs_mode='rgb+depth+segmentation', control_mode='pd_joint_pos',
                   render_mode='rgb_array', sim_backend='cpu', max_episode_steps=budget + 10)
    return env, budget


class Runner:
    def __init__(self, task, episode, oracle_count):
        self.task, self.episode, self.oracle_count = task, episode, oracle_count
        self.env, self.budget = make_env(task)
        self.seed = DEV_SEED_BASE + 16 * episode
        self.u = None
        self.trace = []

    def frame(self, obs):
        sd = obs['sensor_data']
        q = obs['agent']['qpos'][0].cpu().numpy()
        n = len(self.u.agent.arm_joint_names)
        return (sd['front_camera']['rgb'][0].cpu().numpy().astype(np.uint8),
                sd['hand_camera']['rgb'][0].cpu().numpy().astype(np.uint8),
                np.concatenate([q[:n], q[n:n + 1]]).astype(np.float32))

    def snapshot(self):
        u = self.u
        from memory_management.fruit_v2_distance import any_hanging
        tcp = u.agent.tcp_pose.p[0].cpu().numpy()
        q = u.agent.robot.get_qpos()[0].cpu().numpy()
        return {'q': [round(float(x), 4) for x in q[:8]], 'step': int(u.control_steps), 'n_in_basket': int(u.in_basket().sum()),
                'distance': float(any_hanging(tcp, u.fruit_positions(), u.hanging()))}

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        self.u = self.env.unwrapped
        self.instruction = self.u.layout.instruction
        self.trace = [self.snapshot()]
        return self.frame(obs)

    def step(self, action, guard=True):
        action = np.asarray(action, dtype=np.float64)
        if guard and GUARD is not None:
            from memory_management.fruit_rescue_platform import guard_v2
            q = self.u.agent.robot.get_qpos()[0].cpu().numpy()
            action, _ = guard_v2(action, q, GUARD)
        obs, _, terminated, truncated, info = self.env.step(np.asarray(action, dtype=np.float32))
        self.trace.append(self.snapshot())
        success, fail = bool(info['success'][0]), bool(info['fail'][0])
        timeout = bool(truncated) or self.u.control_steps >= self.budget
        outcome = 'success' if success else 'fail' if fail else 'timeout' if timeout else 'ongoing'
        self.last_info = info
        return self.frame(obs), bool(terminated) or timeout or success or fail, outcome

    def prompt(self):
        if not self.oracle_count:
            return self.instruction.lower()
        return self.instruction.lower() + f' fruits already in the basket: {self.trace[-1]["n_in_basket"]}.'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--task', choices=('count', 'occluded_ripe'), required=True)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--memory', choices=('uniform', 'ka', 'none'), required=True)
    p.add_argument('--oracle-count', action='store_true')
    p.add_argument('--context', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-steps', type=int, help='smoke test only: stop after this many control steps')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    row = {'task': a.task, 'episode': a.episode, 'memory': a.memory, 'oracle_count': a.oracle_count,
           'outcome': 'error', 'error': ''}
    runner = None
    started = time.monotonic()
    try:
        from openpi_client import websocket_client_policy as wcp
        from utils import pack_buffer  # official helper
        runner = Runner(a.task, a.episode, a.oracle_count)
        front, wrist, state = runner.reset()
        row.update(seed=runner.seed, N=int(runner.u.layout.n_pick), instruction=runner.instruction)
        # The server reads the context at reset to choose the memory selector and the distance source.
        a.context.write_text(json.dumps({'memory': a.memory, 'trace': str(a.out / 'trace.jsonl')}))
        (a.out / 'trace.jsonl').write_text('')
        client = wcp.MMEVLAWebsocketClientPolicy('127.0.0.1', a.port)
        resp = client.reset()
        while not resp.get('reset_finished', False):
            time.sleep(0.1)
        images, wrists, states = [front], [wrist], [state]
        fixed = np.r_[state[:7], 1.].astype(np.float32)
        for _ in range(30):  # history-only observe prefix, as in the training data
            (front, wrist, state), stop, outcome = runner.step(fixed, guard=False)
            images.append(front); wrists.append(wrist); states.append(state)
        exec_start = 30
        pending = (images, states)
        outcome = 'ongoing'
        while outcome == 'ongoing' and not (a.max_steps and runner.u.control_steps >= a.max_steps):
            with (a.out / 'trace.jsonl').open('a') as f:
                for s in runner.trace[len(runner.trace) - len(pending[0]):]:
                    f.write(json.dumps(s) + '\n')
            client.add_buffer(pack_buffer(pending[0], pending[1], exec_start))
            chunk = client.infer({'observation/image': front, 'observation/wrist_image': wrist,
                                  'observation/state': state, 'prompt': runner.prompt()})['actions'][:16]
            new_images, new_states = [], []
            with (a.out / 'actions.jsonl').open('a') as f:
                f.write(json.dumps({'step': int(runner.u.control_steps), 'chunk': np.asarray(chunk).round(4).tolist()}) + '\n')
            for action in chunk:
                (front, wrist, state), stop, outcome = runner.step(action)
                new_images.append(front); new_states.append(state)
                if stop:
                    break
            pending = (new_images, new_states)
        with (a.out / 'trace.jsonl').open('a') as f:
            for s_ in runner.trace[len(runner.trace) - len(pending[0]):]:
                f.write(json.dumps(s_) + '\n')
        with (a.out / 'actions.jsonl').open('a') as f:
            pass
        info = runner.last_info
        row.update(outcome=outcome, steps=int(runner.u.control_steps),
                   n_in_basket=int(info['n_in_basket'][0]) if 'n_in_basket' in info else runner.trace[-1]['n_in_basket'],
                   reason=info.get('v2_reason'),
                   robot_picked=int(info['v2_robot_picked'][0]) if 'v2_robot_picked' in info else None)
    except BaseException:
        row['error'] = traceback.format_exc()
    finally:
        row['wall_s'] = round(time.monotonic() - started, 1)
        if runner is not None:
            runner.env.close()
    (a.out / 'result.json').write_text(json.dumps(row, indent=1))
    print(json.dumps({k: v for k, v in row.items() if k != 'error'}), flush=True)
    return int(bool(row['error']))


if __name__ == '__main__':
    raise SystemExit(main())
