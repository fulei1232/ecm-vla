"""Revision8 opt-in semantics. Frozen v1 code and archived R0 results stay intact."""
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from .layout import sample_layout


@dataclass(frozen=True)
class TaskV2Spec:
    version: str = 'v1'
    prefix_steps: int = 30
    finish_hold_steps: int = 40
    rest_joint_tolerance_rad: float = .05
    static_velocity_rad_s: float = .05
    open_finger_min_m: float = .03
    budget: int = 1800
    gust_start: int = 8
    gust_end: int = 20
    neutral_rgb: tuple = (.55, .55, .55)
    wind_step_min: int = 34
    wind_step_max: int = 66
    wind_protect_tcp_m: float = .15
    visible_pixels: int = 16
    start_delay_max: int = 600
    receipt_steps: int = 2
    replenish_clearance_m: float = .15
    count_n_max: int = 5
    early_start_failure: bool = False
    early_start_joint_tolerance_rad: float = .10
    early_start_tcp_tolerance_m: float = .03

    def __post_init__(self):
        if self.version not in ('v1', 'v2'):
            raise ValueError('task version must be v1 or v2')

    def to_dict(self):
        return asdict(self)


def task_layout(task, seed, version='v1', spec=None):
    original = sample_layout(task, int(seed))
    if version == 'v1':
        return original
    spec = spec or TaskV2Spec(version=version)
    result = deepcopy(original)
    if task == 'count':
        rng = np.random.default_rng([int(seed), 8, 0])
        result.n_pick = int(rng.integers(2, spec.count_n_max + 1))
        order = sorted(range(len(result.fruits)), key=lambda i: -result.fruits[i]['pos'][1])
        result.targets = order
        result.instruction = (f'Wait until the work light turns green, then put exactly {result.n_pick} fruits '
            'into the basket. Do not count windfalls. When finished, return to the initial rest pose, '
            'open the gripper, and hold still.')
    else:
        result.instruction = ('Remember which fruits are red initially. Put all those fruits into the basket '
            'after the gust turns every fruit gray. Do not put initially green fruits into the basket.')
    return result


def early_start(q, reference_q, tcp, reference_tcp, lamp_green, spec):
    """R9 only; observed physical displacement before green is an immediate failure."""
    if not spec.early_start_failure or lamp_green:
        return False
    joint = np.max(np.abs(np.asarray(q)[:7] - np.asarray(reference_q)[:7]))
    distance = np.linalg.norm(np.asarray(tcp) - np.asarray(reference_tcp))
    return bool(joint > spec.early_start_joint_tolerance_rad or distance > spec.early_start_tcp_tolerance_m)


def start_delay(seed, spec):
    return int(np.random.default_rng([int(seed), 8, 1]).integers(0, spec.start_delay_max+1))


def wind_plan(seed, spec):
    rng = np.random.default_rng([int(seed), 7, 1])
    return {'count': int(rng.integers(1, 3)),
        'step': int(rng.integers(spec.wind_step_min, spec.wind_step_max+1)),
        'order_seed': [int(seed), 7, 2]}


def neutral_alpha(step, spec):
    a = float(np.clip((step-spec.gust_start)/(spec.gust_end-spec.gust_start), 0, 1))
    return a*a*(3-2*a)


def bounded_inside(positions, center, inner_half, height):
    rel = np.asarray(positions, dtype=float)-np.asarray(center, dtype=float)
    return ((np.abs(rel[:, 0]) < inner_half) & (np.abs(rel[:, 1]) < inner_half)
        & (rel[:, 2] > 0) & (rel[:, 2] < height))


def count_outcome(step, streak, n_pick, inside, robot_picked, dropped, spec):
    """No success merely for reaching N; robot harvesting N+1 is an immediate failure."""
    reason = 'running'
    if dropped:
        reason = 'robot_drop'
    elif robot_picked > n_pick or inside > n_pick:
        reason = 'overpick'
    elif step >= spec.budget:
        reason = 'budget'
    elif streak >= spec.finish_hold_steps:
        reason = 'finish_signal'
    done = reason != 'running'
    success = done and reason == 'finish_signal' and inside == n_pick and not dropped
    return {'success': bool(success), 'fail': bool(done and not success), 'reason': reason,
        'terminated': done, 'n_in_basket': int(inside), 'robot_picked': int(robot_picked)}


def choose_count(level, n_pick, current_tree, initial_tree=None, deliveries=None, visible_basket_count=0,
                 ground_count=0, current_step=None, clock=None, onset=None):
    """Fixed shortcuts, declared before results. No seed, clock, GT count or hidden wind flags."""
    if level == 'history':
        stop = int(deliveries) >= n_pick
    elif level == 'initial_current':
        stop = len(initial_tree)-len(current_tree) >= n_pick
    elif level == 'initial_ground':
        stop = len(initial_tree)-len(current_tree)-ground_count >= n_pick
    elif level == 'current':
        stop = len(current_tree) <= 1
    elif level in ('clock', 'clock_onset', 'clock_onset_k0'):
        if clock is None or current_step is None:raise ValueError('A frozen clock fit and t are required')
        if level in ('clock_onset', 'clock_onset_k0') and onset is None:
            raise ValueError('The reporting clock needs an available onset estimate')
        elapsed = current_step-(onset if level in ('clock_onset', 'clock_onset_k0') else 0)
        estimate = int(np.rint((elapsed-clock['a'])/clock['b']+clock['bias']))
        stop = estimate >= n_pick
    elif level == 'nonstop':
        stop = False
    else:
        raise ValueError(level)
    if stop or (level!='nonstop' and visible_basket_count >= n_pick) or not current_tree:
        return None
    return max(current_tree, key=lambda f: f['position'][1])['id']
