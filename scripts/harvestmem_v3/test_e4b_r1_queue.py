"""Formal-queue boundary gates and concurrent result visibility checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_e4b_r1 as queue


class QueueTests(unittest.TestCase):
    def sample(self, n=60):
        records = [{'arm':'DP1R','episode':200+i,'planner_calls':1,
                    'planner_failures':[{}] if i < 3 else [],'reason':'robot_drop',
                    'handoffs':[{}]} for i in range(n)]
        audits = [{'arm':'DP1R','count_stop_pass':True,
                   'checks':[{'policy_at_end':True,'jump_pass':i >= 3,
                              'distance8_pass':i >= 6}]} for i in range(n)]
        return records, audits

    def test_exact_thresholds_pass_and_exceeding_any_threshold_stops(self):
        records, audits = self.sample()
        totals, failures = queue.engineering(records, audits)
        self.assertEqual(failures, [])
        self.assertEqual(totals['DP1R']['jump_good'],57)
        self.assertEqual(totals['DP1R']['distance8_good'],54)
        for which in ('planner','jump','distance','stop'):
            with self.subTest(which=which):
                rr, aa = copy.deepcopy(records), copy.deepcopy(audits)
                if which == 'planner': rr[3]['planner_failures'] = [{}]
                if which == 'jump': aa[3]['checks'][0]['jump_pass'] = False
                if which == 'distance': aa[6]['checks'][0]['distance8_pass'] = False
                if which == 'stop': aa[2]['count_stop_pass'] = False
                self.assertTrue(queue.engineering(rr,aa)[1])

    def test_early_failures_wait_for_failure_count_allowance(self):
        for n in (1,3,6):
            with self.subTest(n=n):
                self.assertFalse(queue.engineering(*self.sample(n))[1])
        for which in ('planner','jump','distance'):
            with self.subTest(which=which):
                rr, aa = self.sample(7)
                if which == 'planner': rr[3]['planner_failures'] = [{}]
                if which == 'jump': aa[3]['checks'][0]['jump_pass'] = False
                if which == 'distance': aa[6]['checks'][0]['distance8_pass'] = False
                self.assertTrue(queue.engineering(rr,aa)[1])

    def test_final_check_uses_actual_event_denominators(self):
        records, audits = self.sample(59)
        for r in records:
            r['planner_calls'] = 0
            r['planner_failures'] = []
        for a in audits:
            a['checks'] = []
        records[0]['planner_calls'] = 1
        records[0]['planner_failures'] = [{}]
        audits[0]['checks'] = [{'policy_at_end':True,'jump_pass':False,'distance8_pass':False}]
        self.assertFalse(queue.engineering(records,audits)[1])
        records.append({'arm':'DP1R','episode':259,'planner_calls':0,'planner_failures':[],
                        'reason':'robot_drop','handoffs':[{}]})
        audits.append({'arm':'DP1R','count_stop_pass':True,'checks':[]})
        self.assertEqual(len(queue.engineering(records,audits)[1]),3)

    def test_multiple_events_use_larger_midrun_denominators(self):
        records, audits = self.sample(6)
        for r, a in zip(records,audits):
            r['planner_calls'] = 20
            a['checks'] = [{'policy_at_end':True,'jump_pass':True,'distance8_pass':True} for _ in range(20)]
        records[0]['planner_failures'] = [{}] * 4
        for c in audits[0]['checks'][:6]:
            c['jump_pass'] = False
        self.assertFalse(queue.engineering(records,audits)[1])
        # Total calls/jumps are 120: six failures pass, seven fail.
        records[0]['planner_failures'].append({})
        audits[0]['checks'][6]['jump_pass'] = False
        self.assertEqual(len(queue.engineering(records,audits)[1]),2)

    def test_hard_violations_still_stop_after_first_episode(self):
        for which in ('stop','early_finish'):
            records, audits = self.sample(1)
            if which == 'stop':
                audits[0]['count_stop_pass'] = False
            else:
                records[0]['reason'] = 'finish_signal'
                records[0]['handoffs'] = []
            self.assertTrue(queue.engineering(records,audits)[1])

    def test_recover_does_not_block_on_small_sample_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'DP1R/count/ep200'
            path.mkdir(parents=True)
            records, audits = self.sample(1)
            (path/'result.json').write_text(json.dumps({**records[0],'outcome':'fail','error':''}))
            (path/'engineering.json').write_text(json.dumps(audits[0]))
            with mock.patch.object(queue,'OUT',root):
                recovered, failures = queue.summarize(recover=True)
            self.assertEqual(len(recovered),1)
            self.assertEqual(failures,[])

    def test_concurrent_uncommitted_result_is_skipped_and_resume_rejects_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'DP1R/count/ep200'
            path.mkdir(parents=True)
            (path/'result.json').write_text('{')
            with mock.patch.object(queue,'OUT',root):
                records, failures = queue.summarize()
                self.assertEqual(records,[])
                self.assertEqual(failures,[])
                with self.assertRaises(json.JSONDecodeError):
                    queue.summarize(recover=True)

    def test_controls_are_never_queued(self):
        jobs = [(a,e) for group in queue.GROUPS for e in range(200,260) for a in group]
        self.assertEqual(len(jobs),360)
        self.assertEqual(len(set(jobs)),360)
        self.assertFalse({'D','E','DR','ER'} & {a for a,e in jobs})


if __name__ == '__main__':
    unittest.main()
