"""Fixed optical enclosure for the approved Count-v2 visibility repair.

No fruit state, count, phase, camera segmentation or RNG enters this geometry.
The enclosure has an open square drop port and no collision shapes.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CountShieldSpec:
    mode: str = 'off'
    inner_half_m: float = .140
    port_half_m: float = .075
    thickness_m: float = .006
    bottom_offset_m: float = -.050
    top_offset_m: float = .210
    collar_inner_half_m: float = .090
    collar_bottom_offset_m: float = .118
    rgb: tuple = (.30, .32, .34)

    def __post_init__(self):
        if self.mode not in ('off', 'fixed_v1'):
            raise ValueError('count_shield must be off or fixed_v1')

    def to_dict(self):
        return asdict(self)


def shield_boxes(center, spec):
    """Outer enclosure plus four inner port baffles; all geometry is constant."""
    if spec.mode == 'off':
        return []
    c = np.asarray(center, dtype=float)
    h, a, t = spec.inner_half_m, spec.port_half_m, spec.thickness_m
    outer = h + t
    lo, hi = spec.bottom_offset_m, spec.top_offset_m
    rows = []
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rows.append((np.array([sx*(h+t/2), sy*(h+t/2), (lo+hi)/2]),
                     np.array([t/2 if sx else outer, t/2 if sy else outer, (hi-lo)/2])))
    for sign in (-1, 1):
        rows.append((np.array([sign*(outer+a)/2, 0, hi+t/2]),
                     np.array([(outer-a)/2, outer, t/2])))
        rows.append((np.array([0, sign*(outer+a)/2, hi+t/2]),
                     np.array([a, (outer-a)/2, t/2])))
    rows.append((np.array([0, 0, lo-t/2]), np.array([outer, outer, t/2])))
    # A roof opening alone admits oblique rays to the far basket rim.
    # Extend the existing basket's inner wall up to the roof, leaving a clear
    # straight drop corridor. Its lower end overlaps the visual rim by 2 mm.
    inner, bottom = spec.collar_inner_half_m, spec.collar_bottom_offset_m
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rows.append((np.array([sx*(inner+t/2), sy*(inner+t/2), (bottom+hi)/2]),
                     np.array([t/2 if sx else inner+t, t/2 if sy else inner+t, (hi-bottom)/2])))
    return [{'center': (c+p).tolist(), 'half': half.tolist()} for p, half in rows]


def sphere_intersections(positions, radii, boxes):
    """Report fruit/enclosure intersections; never alter observations or physics."""
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if not boxes or not len(positions):
        return np.zeros(len(positions), dtype=bool)
    centers = np.asarray([b['center'] for b in boxes])
    halves = np.asarray([b['half'] for b in boxes])
    gap = np.maximum(np.abs(positions[:, None]-centers[None])-halves[None], 0)
    return np.any(np.sum(gap**2, axis=-1) < np.asarray(radii)[:, None]**2, axis=1)


def segment_intersects_boxes(start, end, boxes):
    """Geometric design check for sight lines; independent of renderer/fruit truth."""
    start=np.asarray(start,dtype=float);delta=np.asarray(end,dtype=float)-start
    for box in boxes:
        low=np.asarray(box['center'])-box['half'];high=np.asarray(box['center'])+box['half']
        enter,leave=0.,1.
        for i in range(3):
            if abs(delta[i])<1e-12:
                if start[i]<low[i] or start[i]>high[i]:enter=2.;break
            else:
                a,b=sorted(((low[i]-start[i])/delta[i],(high[i]-start[i])/delta[i]))
                enter=max(enter,a);leave=min(leave,b)
        if enter<=leave:return True
    return False
