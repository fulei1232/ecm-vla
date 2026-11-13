"""Only new continuation bindings and cost reporting; accepted R9 checks are not rerun."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT)]
import fruit_r10_common as common
from fruit_r10_data_timing import summarize_episode_times


class ContinuationTests(unittest.TestCase):
    def test_only_process_local_output_bindings_change(self):
        support=common.activate()
        import fruit_r9_data as recorder
        import fruit_r9_generation as generation
        self.assertEqual(recorder.OUT,common.OUT)
        self.assertEqual(generation.OUT,common.OUT)
        self.assertEqual(generation.DATASET,common.DATASET)
        self.assertEqual(generation.RAW_CLEAN,common.RAW_CLEAN)
        self.assertEqual(support.OUT,common.OUT)

    def test_report_failed_attempts_without_biasing_success_cost(self):
        rows=[dict(task='count',N=8,purpose=purpose,success=success,process_wall_seconds=seconds,
                   validated_case_wall_seconds=seconds+1,physical_steps=100)
              for purpose,success,seconds in [('train_clean',True,10.),('train_dart',True,30.),('train_dart',False,5.)]]
        value=summarize_episode_times(rows);group=value['by_task_and_N']['count']
        self.assertEqual(value['attempts'],3);self.assertEqual(value['accepted'],2)
        self.assertEqual(group['all_attempts']['8']['process_wall_seconds']['mean'],15.)
        self.assertEqual(group['accepted']['8']['process_wall_seconds']['mean'],20.)
        self.assertEqual(group['DART_accepted']['8']['process_wall_seconds']['mean'],30.)
        self.assertEqual(value['sum_attempt_process_seconds'],45.)

    def test_parallel_stage_wall_is_not_sum_of_episode_times(self):
        saved=(common._phase_clock,common._phase_record)
        try:
            common._phase_clock=None;common._phase_record=None
            with patch.object(common,'proof'),patch.object(common.time,'monotonic',side_effect=[100.,110.,130.,140.]):
                common._data_phase({'state':'r10_generating_clean_DART'})
                common._data_phase({'state':'r10_building_shared_data'})
                common._data_phase({'state':'r10_computing_one_shared_normalization'})
                value=common.finish_data_timing()
            self.assertEqual(value['raw_generation_wall_seconds'],10.)
            self.assertEqual(value['feature_build_wall_seconds'],20.)
            self.assertEqual(value['normalization_and_delivery_wall_seconds'],10.)
            self.assertEqual(value['total_wall_seconds'],40.)
            self.assertTrue(value['completed'])
        finally:common._phase_clock,common._phase_record=saved


if __name__=='__main__':unittest.main()
