"""Revision-1 supervision-statistic boundary cases; no data construction changes."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from fruit_expert_audit_stats import delayed_supervision


class ExpertStatsTests(unittest.TestCase):
    def test_delayed_motion_and_history_only_anchor(self):
        states=np.zeros((24,7));actions=np.zeros((24,8));actions[8:,:7]=.05
        phases=['observe']*24;supervised=np.ones(24,dtype=bool)
        rows,total=delayed_supervision(states,actions,phases,supervised)
        self.assertEqual(rows['observe']['numerator'],1)
        self.assertEqual(total['denominator'],24)
        supervised[0]=False
        rows,total=delayed_supervision(states,actions,phases,supervised)
        self.assertEqual(total['numerator'],0)
        self.assertEqual(total['denominator'],23)

    def test_early_strict_threshold_and_current_state_reference(self):
        states=np.zeros((20,7));actions=np.zeros((20,8));actions[:8,:7]=.01;actions[8:,:7]=.05
        phases=['grasp']*20;supervised=np.zeros(20,dtype=bool);supervised[0]=True
        _,total=delayed_supervision(states,actions,phases,supervised)
        self.assertEqual(total['numerator'],0)
        states[0]=.5
        actions[:8,:7]=.5;actions[8:,:7]=.56
        _,total=delayed_supervision(states,actions,phases,supervised)
        self.assertEqual(total['numerator'],1)

    def test_last_action_padding_and_other_phase_group(self):
        states=np.zeros((3,7));actions=np.zeros((3,8));actions[-1,:7]=.06
        rows,total=delayed_supervision(states,actions,['reach','carry','pull'],np.ones(3,dtype=bool))
        self.assertEqual(rows['other']['denominator'],3)
        self.assertEqual(total['numerator'],0)


if __name__=='__main__': unittest.main()
