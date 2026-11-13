"""Continuous optical liner, separated from the original basket decoration.

Geometry is fixed. No counts, fruit classifications or action phases are inputs.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .count_shield_spec import CountShieldSpec, shield_boxes, sphere_intersections


@dataclass(frozen=True)
class CountShieldV2Spec:
    mode: str = 'off'
    decoration_gap_m: float = .002
    inner_half_m: float = .140
    port_half_m: float = .075
    thickness_m: float = .006
    bottom_offset_m: float = -.050
    top_offset_m: float = .210
    collar_inner_half_m: float = .090
    rgb: tuple = (.30, .32, .34)

    def __post_init__(self):
        if self.mode not in ('off','fixed_v2'):
            raise ValueError('count_shield must be off or fixed_v2')

    def to_dict(self):return asdict(self)


def basket_boxes(center, inner_half=.09, height=.12, wall=.006):
    c=np.asarray(center,dtype=float)
    boxes=[{'center':(c+[0,0,wall/2]).tolist(),'half':[inner_half+wall,inner_half+wall,wall/2]}]
    for sx,sy in ((1,0),(-1,0),(0,1),(0,-1)):
        boxes.append({'center':(c+[sx*(inner_half+wall/2),sy*(inner_half+wall/2),height/2]).tolist(),
                      'half':[wall/2 if sx else inner_half+wall,wall/2 if sy else inner_half+wall,height/2]})
    return boxes


def basket_visual_shifts(spec):
    g=spec.decoration_gap_m if spec.mode!='off' else 0.
    return [[0,0,-g],[g,0,0],[-g,0,0],[0,g,0],[0,-g,0]]


def optical_geometry(center, spec, basket_height=.12):
    if spec.mode=='off':return {'render':[],'new_obstacle_audit':[]}
    c=np.asarray(center,dtype=float);t=spec.thickness_m
    outside=shield_boxes(center,CountShieldSpec(mode='fixed_v1',inner_half_m=spec.inner_half_m,
        port_half_m=spec.port_half_m,thickness_m=spec.thickness_m,bottom_offset_m=spec.bottom_offset_m,
        top_offset_m=spec.top_offset_m,rgb=spec.rgb))[:9]
    render=list(outside);audit=list(outside)
    h=spec.collar_inner_half_m;top=spec.top_offset_m
    for sx,sy in ((1,0),(-1,0),(0,1),(0,-1)):
        horizontal=np.array([sx*(h+t/2),sy*(h+t/2),0.])
        half_xy=[t/2 if sx else h+t,t/2 if sy else h+t]
        render.append({'center':(c+horizontal+[0,0,top/2]).tolist(),'half':[*half_xy,top/2]})
        # Below the existing rim these surfaces exactly occupy pre-existing
        # collision walls. Only the extension adds an optical obstruction.
        audit.append({'center':(c+horizontal+[0,0,(basket_height+top)/2]).tolist(),
                      'half':[*half_xy,(top-basket_height)/2]})
    render.append({'center':(c+[0,0,t/2]).tolist(),'half':[h+t,h+t,t/2]})
    return {'render':render,'new_obstacle_audit':audit}


def shifted_decoration(center,spec):
    return [{'center':(np.asarray(b['center'])+s).tolist(),'half':b['half']}
            for b,s in zip(basket_boxes(center),basket_visual_shifts(spec))]


def ray_depths(origins,directions,boxes):
    """Offline AABB optical-depth calculation. No rendering or environment steps."""
    directions=np.asarray(directions,float);origins=np.broadcast_to(np.asarray(origins,float),directions.shape)
    centers=np.asarray([b['center'] for b in boxes]);half=np.asarray([b['half'] for b in boxes])
    parallel=np.abs(directions[:,None,:])<1e-15
    divisor=np.where(parallel,1.,directions[:,None,:])
    a=(centers[None]-half[None]-origins[:,None,:])/divisor
    b=(centers[None]+half[None]-origins[:,None,:])/divisor
    inside=(origins[:,None,:]>=centers[None]-half[None])&(origins[:,None,:]<=centers[None]+half[None])
    a=np.where(parallel,np.where(inside,-np.inf,np.inf),a)
    b=np.where(parallel,np.where(inside,np.inf,-np.inf),b)
    enter=np.max(np.minimum(a,b),axis=-1);leave=np.min(np.maximum(a,b),axis=-1)
    bad_parallel=np.any(parallel&~inside,axis=-1)
    return np.min(np.where((leave>=enter)&(leave>0)&~bad_parallel,np.where(enter>0,enter,leave),np.inf),axis=-1)
