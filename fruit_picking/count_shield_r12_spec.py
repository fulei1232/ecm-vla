"""Fixed visual false bottom for Count-v2; it never changes collision geometry."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CountShieldR12Spec:
    mode: str = 'off'
    false_bottom_z_m: float = .310
    liner_inner_half_m: float = .090
    edge_overlap_m: float = .002
    half_thickness_m: float = .001
    rgb: tuple = (.30, .32, .34)

    def __post_init__(self):
        if self.mode not in ('off', 'fixed_v2_r12'):
            raise ValueError('R12 visual geometry mode must be off or fixed_v2_r12')
        if self.edge_overlap_m < .002:
            raise ValueError('The false bottom must overlap all four liner edges by at least 2 mm')

    def to_dict(self):
        return asdict(self)


def false_bottom_box(basket_center, spec=None):
    """Opaque render-only plate sealing the entire square drop passage."""
    spec = spec or CountShieldR12Spec(mode='fixed_v2_r12')
    if spec.mode == 'off':
        return None
    c = np.asarray(basket_center, dtype=float)
    if c.shape != (3,) or not np.allclose(c[2], .150, atol=1e-9, rtol=0):
        raise ValueError(f'R12 expects the frozen Count-v2 basket z=.15 m, got {c!r}')
    half = spec.liner_inner_half_m + spec.edge_overlap_m
    if half < spec.liner_inner_half_m + .002 - 1e-12:
        raise AssertionError('False bottom does not cover the liner overlap')
    return {
        'center': [float(c[0]), float(c[1]), spec.false_bottom_z_m],
        'half': [float(half), float(half), spec.half_thickness_m],
        'rgb': list(spec.rgb),
        'purpose': 'visual-only sealed drop-passage floor; fruit passes through; zero collision shapes'
    }


def segment_box_interval(start, end, box):
    """Return the closed segment parameter interval inside an AABB, or None."""
    start = np.asarray(start, dtype=float)
    delta = np.asarray(end, dtype=float) - start
    center = np.asarray(box['center'], dtype=float)
    half = np.asarray(box['half'], dtype=float)
    enter, leave = 0., 1.
    for axis in range(3):
        low, high = center[axis] - half[axis], center[axis] + half[axis]
        if abs(delta[axis]) < 1e-15:
            if start[axis] < low or start[axis] > high:
                return None
            continue
        a, b = sorted(((low - start[axis]) / delta[axis],
                       (high - start[axis]) / delta[axis]))
        enter, leave = max(enter, a), min(leave, b)
        if enter > leave:
            return None
    return (enter, leave)

