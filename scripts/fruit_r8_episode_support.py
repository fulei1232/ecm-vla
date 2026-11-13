"""R8 diagnostic recorder and strictly filtered script decisions; no training outputs."""
import json
import sys
from pathlib import Path

import numpy as np

from v_a_common import ECM,atomic_json
sys.path[:0]=[str(ECM),str(ECM/'src')]
from fruit_picking.task_v2_audit_r8 import pixel_counts,filtered_view,appearance_check
from fruit_picking.task_v2_spec import choose_count
from memory_management.fruit_rescue_platform import guard_v2
from memory_management.fruit_v2_distance import any_hanging,maturity_sensitive


class EpisodeEnd(Exception):pass
class TargetGone(Exception):pass
class CheckFailed(RuntimeError):pass


class R8Replay:
    def __init__(self,env,obs,directory,level,guard,clock=None):
        self.env,self.u,self.obs=env,env.unwrapped,obs
        self.directory,self.level,self.guard,self.clock=directory,level,guard,clock
        self.actions=[];self.decisions=[];self.releases=[];self.basket_entries=[];self.entered=set()
        self.ended=False;self.success=False;self.reason='';self.snapshots={};self.appearance=[]
        self.counts=pixel_counts(obs);self.initial=filtered_view(self.u,self.counts)
        self.base_rgb=obs['sensor_data']['front_camera']['rgb'][0].cpu().numpy().copy()
        self.base_masks=self.slot_masks();self.base_lamp=self.lamp_mask()
        self.seen_replenishments=0
        self.max_visibility={c:{'basket':0,'basket_fruit':0,'inactive_reserve':0,'lamp':0} for c in self.counts}
        self.observed_onset=None
        self.trace=(directory/'trace.jsonl').open('x')
        self.record('observe')

    def slot_masks(self):
        seg=self.obs['sensor_data']['front_camera']['segmentation'][0,...,0].cpu().numpy()
        masks={i:np.zeros(seg.shape,dtype=bool) for i in range(self.u.original_slots)}
        for i in np.flatnonzero(self.u.hanging()):
            masks[self.u.slot_of[i]] |= seg==int(self.u.fruits[int(i)].per_scene_id[0])
        return masks

    def lamp_mask(self):
        seg=self.obs['sensor_data']['front_camera']['segmentation'][0,...,0].cpu().numpy()
        return seg==int(self.u.work_lamp.per_scene_id[0]) if self.u.task=='count' else np.zeros(seg.shape,dtype=bool)

    def fail(self,reason,details=None):
        atomic_json(self.directory/'check_failure.json',{'reason':reason,'details':details,'step':self.u.control_steps})
        self.capture('check_failure')
        raise CheckFailed(reason)

    def capture(self,stage,replace=False):
        from PIL import Image
        if stage in self.snapshots and not replace:return
        files={}
        for camera in self.counts:
            name=f'{stage}_{camera}.png'
            Image.fromarray(self.obs['sensor_data'][camera]['rgb'][0].cpu().numpy().astype(np.uint8)).save(self.directory/name)
            files[camera]=name
        self.snapshots[stage]={'step':self.u.control_steps,'files':files}

    def check_appearance(self,stage):
        if self.u.task!='count':return
        check=appearance_check(self.base_rgb,self.base_masks,
            self.obs['sensor_data']['front_camera']['rgb'][0].cpu().numpy(),self.slot_masks(),self.u.wind_slots,
            self.base_lamp|self.lamp_mask())
        check.update(step=self.u.control_steps,stage=stage)
        self.appearance.append(check)
        if not check['passed']:self.fail('REPLENISHMENT_EQUIVALENCE_FAILED',check)

    def record(self,phase):
        u=self.u;self.counts=pixel_counts(self.obs)
        inside=u.in_basket();physical=u.present;hanging=u.hanging()
        for i in np.flatnonzero(inside & ~np.asarray(u.wind_detached)):
            i=int(i)
            if i not in self.entered:
                self.entered.add(i);self.basket_entries.append({'step':u.control_steps,'actor':i,'slot':u.slot_of[i]})
        visibility={}
        for camera,counts in self.counts.items():
            v={'basket':counts.get(int(u.basket.per_scene_id[0]),0),
               'basket_fruit':sum(counts.get(int(u.fruits[int(i)].per_scene_id[0]),0) for i in np.flatnonzero(inside)),
               'inactive_reserve':sum(counts.get(int(u.fruits[int(i)].per_scene_id[0]),0) for i in np.flatnonzero(~physical)),
               'lamp':counts.get(int(u.work_lamp.per_scene_id[0]),0) if u.task=='count' else 0}
            visibility[camera]=v
            for k,n in v.items():self.max_visibility[camera][k]=max(self.max_visibility[camera][k],n)
            if v['basket'] or v['basket_fruit']:self.fail('BASKET_OR_CONTENT_VISIBLE',visibility)
            if v['inactive_reserve']:self.fail('INACTIVE_RESERVE_VISIBLE',visibility)
        if u.task=='count' and u.control_steps==0 and visibility['front_camera']['lamp']<16:
            self.fail('WORK_LAMP_NOT_VISIBLE_AT_START',visibility)
        if u.task=='occluded_ripe' and u.control_steps>=20:
            if any(not np.allclose(m.base_color[:3],u.v2_spec.neutral_rgb,atol=1e-7,rtol=0) for items in u.v2_materials for m,_ in items):
                self.fail('FRUIT_MATURITY_COLOR_RETURNED')
        view=filtered_view(u,self.counts)
        if u.task=='count' and view['lamp_green'] and self.observed_onset is None:self.observed_onset=u.control_steps
        if u.control_steps==0:self.capture('before_green' if u.task=='count' else 'opening')
        if u.task=='count':
            if u.control_steps==u.green_step:self.capture('after_green')
            if u.wind_events:
                ids=[i for e in u.wind_events for i in e['fruits']]
                if all(u.fruit_positions()[i,2]<-.32 for i in ids):self.capture('after_wind')
            if len(u.replenish_events)>self.seen_replenishments:
                self.check_appearance('post_replenishment');self.capture('after_replenishment')
                self.seen_replenishments=len(u.replenish_events)
        elif u.control_steps==20:self.capture('post_gust')
        if phase=='release':self.capture('release')
        tcp=u.agent.tcp_pose.p[0].cpu().numpy();positions=u.fruit_positions()
        ripe=[bool(f['ripe']) for f in u.layout.fruits]
        signals={'mature':maturity_sensitive(tcp,positions,ripe,hanging),'any_hanging':any_hanging(tcp,positions,hanging)}
        info=u.evaluate()
        self.trace.write(json.dumps({'step':u.control_steps,'phase':phase,'tcp_position':tcp.tolist(),
            'positions':positions.tolist(),'hanging':hanging.tolist(),'present':physical.tolist(),
            'detached':list(u.attached),'wind_detached':list(u.wind_detached),'inside':inside.tolist(),
            'visibility':visibility,'visible':view,'signals':signals,'lamp_green':u.lamp_green,
            'replenishments':len(u.replenish_events),'finish_streak':u.finish_streak,
            'success':bool(info['success'][0]),'fail':bool(info['fail'][0]),'reason':info.get('v2_reason')})+'\n')

    def step(self,action,labels):
        u=self.u
        if labels['phase']=='grasp':self.capture('before_grasp')
        proposed=np.asarray(action,dtype=float)
        executed,clipping=guard_v2(proposed,u.agent.robot.get_qpos()[0].cpu().numpy(),self.guard)
        obs,_,terminated,truncated,info=self.env.step(executed);self.obs=obs
        self.actions.append({'step':u.control_steps,'block16':max(0,u.control_steps-31)//16,
            'phase':labels['phase'],'target':labels.get('target'),'proposed':proposed.tolist(),'executed':executed.tolist(),**clipping})
        self.record(labels['phase'])
        if bool(terminated[0]) or bool(truncated[0]):
            self.ended=True;self.success=bool(info['success'][0])
            self.reason=info.get('v2_reason') or ('success' if self.success else 'original_fail_or_timeout')
            if self.reason=='wind_basket_anomaly':self.fail('REAL_WIND_BASKET_ANOMALY')
            if u.task=='count' and labels['phase']=='finish':
                self.check_appearance('last_completion_hold');self.capture('finish_hold')
            raise EpisodeEnd()
        target=labels.get('target')
        if target is not None and labels['phase'] in ('reach','approach') and u.wind_detached[target]:raise TargetGone()
        return obs,info

    def provided(self):
        u=self.u;v=filtered_view(u,self.counts)
        p={'tree':v['tree'],'ground':v['ground'],'lamp_green':v['lamp_green'],
            'N':u.layout.n_pick if u.task=='count' else None,'visible_basket_count':v['visible_basket_count'],
            'frame0':self.initial if self.level!='current' else None,
            'release_events':list(self.releases) if self.level=='history' else None,
            't':u.control_steps if self.level in ('clock','clock_onset') else None,
            'green_step':self.observed_onset if self.level=='clock_onset' else None}
        if u.task=='occluded_ripe' and self.level=='history':
            p['tree']=[{'slot':u.slot_of[i],'id':u.slot_of[i],'position':u.fruit_positions()[i].tolist(),
                'ripe_visible':bool(u.layout.fruits[i]['ripe'])} for i in np.flatnonzero(u.hanging())]
        return p

    def choose(self,inspect_slot=None):
        p=self.provided();u=self.u
        if u.task=='count':
            target=choose_count(self.level,p['N'],p['tree'],p['frame0']['tree'] if p['frame0'] else None,
                len(p['release_events']) if p['release_events'] is not None else None,p['visible_basket_count'],
                len(p['ground']),p['t'],self.clock,p['green_step'])
        elif inspect_slot is not None:target=inspect_slot
        else:
            if self.level=='history':candidates=[f for f in p['tree'] if f['ripe_visible']]
            elif self.level=='initial_current':
                ripe={f['slot'] for f in p['frame0']['tree'] if f['ripe_visible']}
                candidates=[f for f in p['tree'] if f['slot'] in ripe]
            else:candidates=[f for f in p['tree'] if f['ripe_visible'] is True or f['leaf_marker']]
            target=max(candidates,key=lambda f:f['position'][1])['slot'] if candidates else None
        estimate=None
        if self.level in ('clock','clock_onset'):
            elapsed=p['t']-(p['green_step'] if self.level=='clock_onset' else 0)
            estimate=int(np.rint((elapsed-self.clock['a'])/self.clock['b']+self.clock['bias']))
        self.decisions.append({'step':u.control_steps,'level':self.level,'provided':p,'chosen_slot':target,
            'clock_estimate':estimate,'purpose':'inspect_initial_location' if inspect_slot is not None else 'pick_or_finish'})
        return target
