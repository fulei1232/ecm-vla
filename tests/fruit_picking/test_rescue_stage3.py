import ast
import collections
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from memory_management.fruit_rescue_platform import guard_v2,install_observe_prefix
from fruit_rescue_stage3_support import make_training_spec
from report_fruit_expert_v31 import choose_expert,position_cell


class Stage3Tests(unittest.TestCase):
    def test_projection_moves_toward_range_without_exceeding_step_limit(self):
        limits={'lower':[-1.]*7,'upper':[1.]*7,'v_max':[.1]*7}
        proposed=np.r_[np.zeros(7),-.7];measured=np.full(7,2.)
        result,event=guard_v2(proposed,measured,limits)
        np.testing.assert_allclose(result[:7],1.9);self.assertEqual(result[7],-.7)
        self.assertTrue(event['guard_projection']);self.assertTrue(event['guard_clipped'])

    def test_guard_ordinary_intersection_and_gripper_unchanged(self):
        limits={'lower':[-1.]*7,'upper':[1.]*7,'v_max':[.2]*7}
        result,event=guard_v2(np.r_[np.full(7,.8),1.],np.zeros(7),limits)
        np.testing.assert_allclose(result[:7],.2);self.assertEqual(result[7],1.)
        self.assertFalse(event['guard_projection'])

    def test_actual_baseline_recipe_has_only_allowed_differences(self):
        spec=make_training_spec(Path('/tmp/rescue_run'),Path('/tmp/rescue_data'))
        self.assertEqual(spec['train']['steps'],12000);self.assertEqual(spec['train']['batch_size'],64)
        self.assertEqual(spec['train']['fsdp_devices'],8);self.assertEqual(spec['gpus'],list(range(8)))
        self.assertEqual(spec['sampler']['events'],0)

    def test_expert_selection_respects_success_fallback_and_core_regression(self):
        stats=lambda failed:{'by_task':{'all':{'success':200,'failed':failed}}}
        gates={key:True for key in ('planning_determinism','target_continuity','method_consistency','wrist_consistency','q7_range','attempt_success')}
        self.assertEqual(choose_expert(gates,stats(5),stats(24))[0],'canonical_v31')
        gates['attempt_success']=False
        self.assertEqual(choose_expert(gates,stats(24),stats(24))[0],'canonical_v31')
        self.assertEqual(choose_expert(gates,stats(30),stats(24))[0],'canonical')
        gates['target_continuity']=False
        self.assertEqual(choose_expert(gates,stats(5),stats(24))[0],'canonical')
        self.assertEqual(position_cell([.35,.081,.54]),'left/upper')

    def test_coverage_counts_production_attempts_without_planning_snapshots(self):
        import json,tempfile,h5py
        from report_fruit_expert_v31 import coverage
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=root/'count';(task/'attempts').mkdir(parents=True)
            layout={'fruits':[{'pos':[.4,-.1,.5]},{'pos':[.35,.12,.6]}],'targets':[1,0]}
            with h5py.File(task/'failed.h5','w') as f:
                f.create_dataset('episode_0/setup/layout_json',data=json.dumps(layout))
            row={'ok':False,'task':'count','seed':100000,'episode':0,'failed_h5':'failed.h5',
                'planning_audit':[{'phase':'reach','method':'detour','snapshot':None,'target':1},
                                  {'phase':'carry','method':'straight','snapshot':None,'target':1}],
                'failure_context':{'phase':'carry','target':1},'reason':'no plan to basket'}
            (task/'attempts/ep0_seed100000.json').write_text(json.dumps(row))
            result=coverage(root)
            self.assertEqual(result['by_task']['count']['left/upper'],{'target_attempts':1,'failed_attempts':1})
            self.assertEqual(result['methods']['segments'],2)

    def test_real_official_loop_prefix_clock_and_exact_total_budget(self):
        policy=Path('/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/examples/robomme')
        executed=[];ingested=[];starts=[];first_query=[]
        class Client:
            def __init__(self,*a): pass
            def reset(self): return {'reset_finished':True}
            def add_buffer(self,obs):
                ingested.extend(int(img[0,0,0,0]) for img in obs['images']);starts.append(obs['exec_start_idx'])
                return {'add_buffer_finished':True}
            def infer(self,obs):
                first_query.append(len(executed));return {'actions':np.repeat(np.arange(16)[:,None],8,axis=1)}
        class Recorder:
            def __init__(self,*a,**kw): pass
            def record(self,**kw): pass
            def save_video(self,*a): pass
        class Runner:
            env_id='FruitCount';episode_id=200;difficulty='easy';warming_up=False
            def get_init_obs(self):
                image=np.zeros((2,2,3),dtype=np.int32)
                return {'images':[image],'wrist_images':[image],'states':[np.zeros(8)],'task_goal':'pick'}
            def step(self,action):
                executed.append(np.asarray(action).copy());t=len(executed)
                image=np.full((2,2,3),t,dtype=np.int32)
                return (image,image,np.zeros(8)),t==1300,'timeout' if t==1300 else 'ongoing'
        ns={'np':np,'collections':collections,'Args':object,'EnvRunner':object,'SubgoalPredictorBase':object,
            'Path':Path,'Tuple':__import__('typing').Tuple,'Optional':__import__('typing').Optional,'Any':object,
            'RolloutRecorder':Recorder,'TASK_WITH_VIDEO_DEMO':[],
            '_websocket_client_policy':SimpleNamespace(MMEVLAWebsocketClientPolicy=Client),
            'pack_buffer':lambda images,states,idx:{'images':np.stack(images)[:,None],'exec_start_idx':idx}}
        for filename,name in (('utils.py','EpisodeState'),('eval.py','EpisodeEvaluator')):
            node=next(n for n in ast.parse((policy/filename).read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(policy/filename),'exec'),ns)
        args=SimpleNamespace(obs_horizon=16,host='localhost',port=1,subgoal_type=None,subgoal_keep_period=1,use_history=True,max_steps=1300)
        evaluator=ns['EpisodeEvaluator'](args,Path('.'));runner=Runner();install_observe_prefix(evaluator,runner)
        predictor=SimpleNamespace(start_episode=lambda *a:None,step=lambda *a:None,get_subgoal=lambda *a:(None,False),end_episode=lambda *a:None)
        self.assertEqual(evaluator.eval_each_episode(runner,predictor,Path('.')),'timeout')
        self.assertEqual(len(executed),1300);self.assertEqual(first_query[0],30)
        np.testing.assert_array_equal(executed[:30],np.repeat(np.r_[np.zeros(7),1.][None],30,axis=0))
        self.assertEqual(starts[0],30);self.assertTrue(all(x==0 for x in starts[1:]))
        self.assertEqual(ingested,list(range(1295)))


if __name__=='__main__': unittest.main()
