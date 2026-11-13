"""Pure CPU tests of deterministic IK sampling, ranking and path fallback."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fruit_picking.joint_planning import DeterministicIKModel, nearest_equivalent, nearest_plan, rank_candidates


class JointPlanningTests(unittest.TestCase):
    def setUp(self):
        self.q = np.zeros(9)
        self.q[6] = -1.4
        self.bounds = np.tile([-6.28, 6.28], (9, 1))
        self.indices = np.arange(7)

    def test_periodic_wrist_nearest_within_limits_then_rejected_if_outside_pi(self):
        self.assertAlmostEqual(nearest_equivalent(2*np.pi-1.5, -1.4, -6.28, 6.28), -1.5)
        a = self.q.copy(); a[6] = 2*np.pi-1.5
        b = self.q.copy(); b[6] = 2.5
        ranked = rank_candidates([b, a], self.q, self.indices, self.bounds)
        self.assertEqual(len(ranked), 1)
        self.assertAlmostEqual(ranked[0][6], -1.5)

    def test_weights_prefer_preserving_base_and_nonarm_state(self):
        a = self.q.copy(); a[0] = .2; a[7:] = 4
        b = self.q.copy(); b[4] = .3; b[7:] = 5
        ranked = rank_candidates([a, b], self.q, self.indices, self.bounds)
        np.testing.assert_array_equal(ranked[0][:7], b[:7])
        np.testing.assert_array_equal(ranked[0][7:], self.q[7:])

    def test_local_sampler_is_independent_of_global_rng(self):
        one = DeterministicIKModel(SimpleNamespace(), self.bounds, self.indices, self.q, 7)
        np.random.seed(999)
        two = DeterministicIKModel(SimpleNamespace(), self.bounds, self.indices, self.q, 7)
        for _ in range(3):
            a, b = one.get_random_configuration(), two.get_random_configuration()
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a[7:], self.q[7:])

    def planner(self, collision=False, fail_first=False):
        current = self.q
        q1, q2 = current.copy(), current.copy()
        q1[0], q2[0] = .1, .2
        class Planner:
            move_group_joint_indices = list(range(7))
            joint_limits = self.bounds
            pinocchio_model = SimpleNamespace(marker='original')
            planning_world = SimpleNamespace(set_use_point_cloud=lambda value: None, set_use_attach=lambda value: None)
            rrt_calls = []
            def IK(self, goal, start, n_init_qpos):
                self.pinocchio_model.get_random_configuration()
                return 'Success', [q2, q1]
            def check_for_self_collision(self, qpos): return ['collision'] if collision else []
            def check_for_env_collision(self, **kwargs): return []
            def TOPP(self, path, dt): return np.arange(len(path))*dt, path, path*0, path*0, (len(path)-1)*dt
            def plan_qpos_to_qpos(self, goals, start, **kwargs):
                self.rrt_calls.append(goals)
                if fail_first and len(self.rrt_calls)==1: return {'status':'RRT Failed'}
                return {'status':'Success','position':np.stack([start[:7],goals[0][:7]])}
        return Planner()

    def test_straight_path_has_priority_and_model_provider_is_restored(self):
        p = self.planner()
        original = p.pinocchio_model
        result, log = nearest_plan(p, np.zeros(7), self.q, .05, False, candidates=2)
        self.assertEqual(result['status'],'Success')
        self.assertEqual(log['method'],'straight')
        self.assertEqual(len(p.rrt_calls),0)
        self.assertIs(p.pinocchio_model,original)

    def test_rrt_gets_only_one_sorted_goal_and_next_only_after_failure(self):
        p = self.planner(collision=True, fail_first=True)
        result, log = nearest_plan(p, np.zeros(7), self.q, .05, True, candidates=2)
        self.assertEqual(result['status'],'Success')
        self.assertEqual(log['method'],'rrt')
        self.assertEqual(log['selected_rank'],2)
        self.assertTrue(all(len(call)==1 for call in p.rrt_calls))
        self.assertEqual([call[0][0] for call in p.rrt_calls],[.1,.2])


if __name__=='__main__':
    unittest.main()
