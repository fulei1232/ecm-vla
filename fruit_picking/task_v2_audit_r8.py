"""Pure R8 observation filters, equivalence checks and clock fits; no simulator startup."""
import numpy as np

from .layout import GROUND_Z


def pixel_counts(obs):
    result={}
    for camera in ('front_camera','hand_camera'):
        seg=obs['sensor_data'][camera]['segmentation'][0,...,0].cpu().numpy().reshape(-1)
        values,counts=np.unique(seg,return_counts=True)
        result[camera]={int(v):int(n) for v,n in zip(values,counts)}
    return result


def filtered_view(u, counts):
    """Stable slot identities only: physical spare IDs/generations never reach the chooser."""
    positions=u.fruit_positions();inside=u.in_basket();tree=[];ground=[];basket_visible=0
    hanging=u.hanging()
    for i in np.flatnonzero(u.present):
        i=int(i);identity=int(u.fruits[i].per_scene_id[0])
        pixels={c:counts[c].get(identity,0) for c in counts}
        if max(pixels.values())<u.v2_spec.visible_pixels:continue
        if inside[i]:basket_visible+=1
        if hanging[i]:
            slot=u.slot_of[i]
            ripe=bool(u.layout.fruits[i]['ripe']) if u.task=='count' or u.v2_color_alpha==0 else None
            marker=any(o['fruit']==slot and any(counts[c].get(int(a.per_scene_id[0]),0)>=16 for c in counts)
                for o,a in zip(u.layout.occluders,u.occluders))
            tree.append({'slot':slot,'id':slot,'position':positions[i].tolist(),'ripe_visible':ripe,
                'leaf_marker':bool(marker),'pixels':pixels})
        if abs(positions[i,2]-(GROUND_Z+u.layout.fruits[i]['radius']))<=.025 and not inside[i]:
            ground.append({'position':positions[i].tolist(),'pixels':pixels})
    tree.sort(key=lambda f:f['slot'])
    ground.sort(key=lambda f:tuple(f['position']))
    lamp={c:counts[c].get(int(u.work_lamp.per_scene_id[0]),0) for c in counts} if u.task=='count' else {c:0 for c in counts}
    observed_green=(bool(u.lamp_green) if max(lamp.values())>=16 else None) if u.task=='count' else True
    return {'tree':tree,'ground':ground,'visible_basket_count':basket_visible,'lamp_green':observed_green,'lamp_pixels':lamp}


def appearance_check(base_rgb, base_masks, rgb, masks, wind_slots, excluded=None):
    slots=[i for i in base_masks if i not in wind_slots]
    if not slots:raise ValueError('No nonwind slot exists for equivalence')
    a=np.logical_or.reduce([base_masks[i] for i in slots]);b=np.logical_or.reduce([masks[i] for i in slots])
    if excluded is not None:a&=~excluded;b&=~excluded
    union=a|b;n=int(union.sum())
    if not n:raise ValueError('Empty fruit region in equivalence check')
    iou=float((a&b).sum()/n)
    diff=np.abs(np.asarray(base_rgb,dtype=np.int16)-np.asarray(rgb,dtype=np.int16))[union]
    mean=float(diff.mean());maximum=int(diff.max())
    return {'passed':iou>=.98 and mean<=2.,'mask_iou':iou,'mean_abs_color_diff_255':mean,
        'max_abs_color_diff_255':maximum,'mean_abs_color_diff':mean/255,'max_abs_color_diff':maximum/255,
        'pixels':n,'included_slots':slots,'excluded_wind_slots':sorted(wind_slots)}


def fit_clock(cases, use_onset=False):
    """OLS timing fit; choose bias on the same ten calibration trajectories, never audit data."""
    x=[];y=[]
    for case in cases:
        for k,t in enumerate(case['basket_entry_steps'],1):
            x.append([1.,float(k)]);y.append(float(t-(case['green_step'] if use_onset else 0)))
    if len(x)<2:raise ValueError('Not enough observed basket entries for clock fit')
    a,b=np.linalg.lstsq(np.asarray(x),np.asarray(y),rcond=None)[0]
    if not np.isfinite([a,b]).all() or b<=0:raise ValueError('Nonpositive/nonfinite fitted clock period')
    breaks=[]
    for case in cases:
        for d in case['decision_points']:
            elapsed=d['step']-(case['green_step'] if use_onset else 0)
            breaks.append(case['N']-.5-(elapsed-a)/b)
    ordered=sorted(set(breaks));candidates={0.}
    for v in ordered:candidates.update([float(v),float(np.nextafter(v,-np.inf)),float(np.nextafter(v,np.inf))])
    for l,r in zip(ordered,ordered[1:]):candidates.add(float((l+r)/2))
    if ordered:candidates.update([ordered[0]-1.,ordered[-1]+1.])
    def successes(bias):
        total=0
        for case in cases:
            first=None
            for d in case['decision_points']:
                elapsed=d['step']-(case['green_step'] if use_onset else 0)
                if int(np.rint((elapsed-a)/b+bias))>=case['N']:
                    first=d['completed_releases'];break
            total+=first==case['N'] and case['reference_success']
        return int(total)
    bias=min(candidates,key=lambda z:(-successes(z),abs(z),z))
    return {'a':float(a),'b':float(b),'bias':float(bias),'rounding':'numpy.rint (ties to even)',
        'calibration_successes':successes(bias),'calibration_episodes':len(cases),'basket_entry_observations':len(x),
        'time_basis':'t-green_step' if use_onset else 'absolute t',
        'bias_search':'all empirical stop-threshold breakpoints, adjacent float values and intervals; max episode success, then min abs bias'}
