"""New seam geometry, explicit defaults and optical coverage; no simulator needed."""
from pathlib import Path
from itertools import product
import ast
import json
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT)]
from fruit_picking.count_shield_v2_spec import (CountShieldV2Spec,optical_geometry,basket_boxes,
    shifted_decoration,basket_visual_shifts,ray_depths,sphere_intersections)

C=np.array([-.06,-.34,.15])
OLD=ROOT/'results/fruit_count_shield_R0_20261106'


def camera_positions(qs):
    joints={j.find('child').get('link'):j for j in ET.parse(ROOT/'fruit_picking/assets/rm75b_eg2/rm75b_eg2.urdf').getroot().findall('joint')}
    def rotation(axis,a):
        axis=np.asarray(axis,float);axis/=np.linalg.norm(axis);x,y,z=axis
        k=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        return np.eye(3)+np.sin(a)*k+(1-np.cos(a))*(k@k)
    def fk(q,link='eg2_base'):
        if link=='base_link':return np.eye(4)
        j=joints[link];o=j.find('origin');r,p,y=[float(v) for v in o.get('rpy','0 0 0').split()]
        a=np.eye(4);a[:3,3]=[float(v) for v in o.get('xyz','0 0 0').split()]
        a[:3,:3]=rotation([0,0,1],y)@rotation([0,1,0],p)@rotation([1,0,0],r)
        if j.get('type')=='revolute':
            b=np.eye(4);b[:3,:3]=rotation([float(v) for v in j.find('axis').get('xyz').split()],q[int(j.get('name').split('_')[1])-1]);a=a@b
        return fk(q,j.find('parent').get('link'))@a
    return [fk(q)[:3,3]+fk(q)[:3,:3]@np.array([0,-.06,.075]) for q in qs]


class SeamTests(unittest.TestCase):
    def setUp(self):
        self.spec=CountShieldV2Spec(mode='fixed_v2');self.geom=optical_geometry(C,self.spec)

    def test_default_off_and_scope(self):
        self.assertEqual(optical_geometry(C,CountShieldV2Spec()),{'render':[],'new_obstacle_audit':[]})
        self.assertEqual(basket_visual_shifts(CountShieldV2Spec()),[[0,0,0]]*5)
        with self.assertRaises(ValueError):CountShieldV2Spec(mode='fixed_v1')
        tree=ast.parse((ROOT/'fruit_picking/count_shield_v2_env.py').read_text())
        init=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        defaults=dict(zip([a.arg for a in init.args.kwonlyargs],init.args.kw_defaults))
        self.assertEqual(ast.literal_eval(defaults['count_shield']),'off')

    def test_continuous_liner_and_two_mm_decoration_separation(self):
        self.assertEqual(len(self.geom['render']),14);self.assertEqual(len(self.geom['new_obstacle_audit']),13)
        for box in self.geom['render'][9:13]:
            self.assertAlmostEqual(box['center'][2]-box['half'][2],.15)
            self.assertAlmostEqual(box['center'][2]+box['half'][2],.36)
        old=basket_boxes(C);moved=shifted_decoration(C,self.spec)
        self.assertAlmostEqual(old[0]['center'][2]+old[0]['half'][2]-(moved[0]['center'][2]+moved[0]['half'][2]),.002)
        for a,b in zip(old[1:],moved[1:]):self.assertAlmostEqual(np.linalg.norm(np.asarray(a['center'])-b['center']),.002)
        self.assertFalse(sphere_intersections([C+[0,0,.213]],[.029],self.geom['render'])[0])

    def test_audit_excludes_only_existing_physical_wall_volume(self):
        physical=basket_boxes(C)
        for box in self.geom['render']:
            for rel in product([-.999,0,.999],repeat=3):
                p=np.asarray(box['center'])+np.asarray(box['half'])*rel
                inside=lambda b:np.all(np.abs(p-b['center'])<=np.asarray(b['half'])+1e-14)
                self.assertTrue(any(inside(b) for b in self.geom['new_obstacle_audit']+physical))
        contact=[-.1245,-.34,.182]
        self.assertTrue(sphere_intersections([contact],[.026],self.geom['render'])[0])
        self.assertFalse(sphere_intersections([contact],[.026],self.geom['new_obstacle_audit'])[0])
        self.assertTrue(sphere_intersections([[-.15,-.34,.32]],[.01],self.geom['new_obstacle_audit'])[0])

    def test_the_four_previous_failure_rays_have_positive_cover_depth(self):
        case=json.loads((OLD/'cause_analysis.json').read_text())
        origin=np.array(case['pose_fit']['camera_position']);rotation=np.array(case['pose_fit']['camera_rotation_cv'])
        pix=np.array([r['pixel_yx'] for r in case['basket_pixels']])
        rays=np.stack([(pix[:,1]+.5-128)/128,(pix[:,0]+.5-128)/128,np.ones(len(pix))],axis=1)@rotation.T
        a=ray_depths(origin,rays,self.geom['render']);b=ray_depths(origin,rays,shifted_decoration(C,self.spec))
        margin=(b-a)*np.linalg.norm(rays,axis=1)
        self.assertTrue(np.all(np.isfinite(a)&np.isfinite(b)&(margin>=.001)))
        self.previous_ray_margin_mm=(margin*1000).tolist()

    def test_existing_trajectory_and_release_neighborhood_all_direction_sightlines(self):
        actions=json.loads((OLD/'diagnostics/ep112_shield_full/actions.json').read_text())
        origins=camera_positions([a['measured_arm_qpos'] for a in actions])
        estimate=np.array(json.loads((OLD/'cause_analysis.json').read_text())['pose_fit']['camera_position'])
        neighborhood=[estimate+np.array(v) for v in product([-.03,0,.03],repeat=3)]
        origins+=neighborhood
        targets=[]
        for box in shifted_decoration(C,self.spec):
            for axis in range(3):
                for sign in [-1,1]:
                    for u,v in product([-.999,-.8,-.4,0,.4,.8,.999],repeat=2):
                        rel=np.zeros(3);rel[axis]=sign;rel[(axis+1)%3]=u;rel[(axis+2)%3]=v
                        targets.append(np.asarray(box['center'])+np.asarray(box['half'])*rel)
        targets=np.asarray(targets);minimum=np.inf;tested=0
        for origin in origins:
            rays=targets-origin
            a=ray_depths(origin,rays,self.geom['render']);b=ray_depths(origin,rays,shifted_decoration(C,self.spec))
            margin=(b-a)*np.linalg.norm(rays,axis=1)
            self.assertTrue(np.all(np.isfinite(a)&np.isfinite(b)&(margin>=.001)))
            minimum=min(minimum,float(margin.min()));tested+=len(rays)
        output={'viewpoints':len(origins),'trajectory_viewpoints':len(actions),'release_grid_viewpoints':len(neighborhood),
                'rays':tested,'minimum_positive_occluder_separation_mm':minimum*1000,
                'scope':'All sight directions to sampled basket-decoration surfaces; entire recorded trajectory and +/-3cm xyz release neighborhood. No FOV restriction or new replay.'}
        from fruit_shield_v2_common import proof
        proof('optical_geometry_check.json',output)

    def test_new_output_bindings_and_old_58_sources_unchanged(self):
        import fruit_shield_v2_common as common
        support=common.activate()
        import fruit_r9_generation as generation
        import fruit_r9_data as recorder
        self.assertEqual(support.OUT,common.OUT);self.assertEqual(generation.OUT,common.OUT)
        self.assertEqual(generation.DATASET,common.DATASET);self.assertEqual(recorder.OUT,common.OUT)
        from v_a_common import read_json,sha256
        old=read_json(OLD/'plan.json');self.assertEqual(len(old['source_sha256']),58)
        for name,h in old['source_sha256'].items():self.assertEqual(sha256(name),h,name)


if __name__=='__main__':unittest.main()
