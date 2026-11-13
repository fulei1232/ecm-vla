"""Boundary checks for the common, protocol-facing failure metrics (CPU only)."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("failure_modes", SCRIPTS / "analyze_fruit_failure_modes.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)


class FailureModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "FruitCount" / "ep200"
        self.path.mkdir(parents=True)

    def fixture(self, steps=261, distance=0.06):
        result = {"task": "FruitCount", "episode_id": 200, "outcome": "timeout", "error": "",
                  "executed_steps": steps, "final_detached": 0, "final_in_basket": 0}
        gt = {"tcp_position": [0, 0, 0], "fruit_positions": [[1, 0, 0], [0, 1, 0]],
              "detached": [False, False], "in_basket": [False, False], "targets": [1, 0], "n_pick": 2}
        actions = [{"control_step": t, "action": [0]*6 + [-1.56, 1.0]}
                   for t in range(1, steps+1)]
        frames = [{"control_step": t, "gt": gt} for t in range(steps+1)]
        return result, actions, frames, [{"distance": [None, distance]}]

    def analyze(self, data):
        for name, value in zip(("result.json", "control_actions.jsonl", "raw_frames.jsonl", "event_signal.jsonl"), data):
            (self.path / name).write_text(json.dumps(value) if name.endswith(".json") else
                                          "".join(json.dumps(r)+"\n" for r in value))
        return fm.analyze_episode(self.path)

    def test_motion_and_freeze_boundaries(self):
        a = self.analyze(self.fixture(260))
        self.assertFalse(a["freeze_200"])
        self.assertEqual(a["longest_freeze_after_60"], 200)
        self.assertIsNone(a["start_idle"])
        b = self.analyze(self.fixture(261))
        self.assertTrue(b["freeze_200"])
        data = self.fixture(3)
        data[1][1]["action"][0] = 0.005  # Equality does not count as motion.
        data[1][2]["action"][0] = 0.011
        self.assertEqual(self.analyze(data)["start_idle"], 3)

    def test_near_miss_open_lower_closed_upper(self):
        self.assertFalse(self.analyze(self.fixture(distance=0.03))["near_miss"])
        self.assertTrue(self.analyze(self.fixture(distance=0.06))["near_miss"])
        self.assertFalse(self.analyze(self.fixture(distance=0.06001))["near_miss"])

    def test_proposed_wrist_and_joint_clip_exclude_gripper(self):
        data = self.fixture(1)
        action = data[1][0]
        action["action"][6] = 1.44  # Exactly center+3 is inside.
        action["proposed_action"] = action["action"][:]
        action["proposed_action"][6] = 6.29
        row = self.analyze(data)
        self.assertFalse(row["wrist_deviation_executed"])
        self.assertTrue(row["wrist_deviation_proposed"])
        self.assertFalse(row["wrist_over_limit"])
        self.assertEqual(row["guard_clip_rate"]["per_joint"][6]["numerator"], 1)
        action["proposed_action"] = action["action"][:]
        action["proposed_action"][7] = -1.0
        self.assertEqual(self.analyze(data)["guard_clip_rate"]["any_joint"]["numerator"], 0)

    def test_gt_target_order_and_count_exact(self):
        data = self.fixture(1)
        gt = data[2][-1]["gt"]
        gt["in_basket"] = [True, False]
        gt["detached"] = [True, False]
        gt["n_pick"] = 1
        data[0].update(final_in_basket=1, final_detached=1)
        row = self.analyze(data)
        self.assertFalse(row["first_target_in_basket"])
        self.assertEqual(row["correct_in_basket"], 1)
        self.assertTrue(row["count_exact"])
        self.assertFalse(row["engineering_error"])

    def test_onset_uses_32_steps_before_not_onset_snapshot(self):
        data = self.fixture(40)
        data[1][-1]["action"][6] = 6.29
        for frame in data[2]:
            frame["gt"] = dict(frame["gt"])
            if frame["control_step"] >= 8:
                frame["gt"]["detached"] = [True, False]
                frame["gt"]["fruit_positions"] = [[0.039, 0, 0], [1, 0, 0]]
        data[0]["final_detached"] = 1
        row = self.analyze(data)
        self.assertEqual(row["deviation_onset_context"]["snapshot_control_step"], 8)
        self.assertEqual(row["deviation_onset_context"]["category"], "carrying_fruit")

    def test_missing_signal_and_engineering_error_are_recorded(self):
        data = self.fixture(1)
        data[3].clear()
        row = self.analyze(data)
        self.assertTrue(row["engineering_error"])
        self.assertIsNone(row["near_miss"])
        self.assertEqual(fm.aggregate([row])["near_miss"]["denominator"], 0)

    def test_missing_result_is_reported_and_incomplete_raw_frames_rejected(self):
        self.analyze(self.fixture(1))
        (self.path / "result.json").unlink()
        self.assertTrue(fm.analyze_episode(self.path)["engineering_error"])
        data = self.fixture(1)
        data[2].pop()
        self.assertTrue(self.analyze(data)["engineering_error"])

    def test_leave_home_ignores_joint_jitter_and_requires_strict_tcp_crossing(self):
        data = self.fixture(3)
        for i, frame in enumerate(data[2]):
            frame["gt"] = dict(frame["gt"], tcp_position=[(0, .049, .05, .051)[i], 0, 0])
        row = self.analyze(data)
        self.assertEqual(row["leave_home_step"], 3)
        self.assertFalse(row["never_left_home"])
        self.assertIsNone(row["start_idle"])

    def test_never_leave_home_and_no_basket_entries_are_separate(self):
        row = self.analyze(self.fixture(3))
        self.assertIsNone(row["leave_home_step"])
        self.assertTrue(row["never_left_home"])
        self.assertEqual(row["post_release_resume_steps"], [])
        total = fm.aggregate([row])
        self.assertEqual(total["leave_home_step"]["never_left_home"]["numerator"], 1)
        self.assertIsNone(total["leave_home_step"]["median"])
        self.assertEqual(total["post_release_resume_steps"]["events"], 0)

    def test_basket_entry_anchor_censoring_and_simultaneous_entries(self):
        data = self.fixture(5)
        positions = [0, .1, .15, .151, .2, .2]
        inside = [[False, False], [True, False], [True, False], [True, False], [True, True], [True, True]]
        for i, frame in enumerate(data[2]):
            frame["gt"] = dict(frame["gt"], tcp_position=[positions[i], 0, 0],
                               in_basket=inside[i], detached=inside[i])
        data[0].update(final_detached=2, final_in_basket=2)
        row = self.analyze(data)
        first, last = row["post_release_resume_steps"]
        self.assertEqual(first["entry_control_step"], 1)
        self.assertEqual(first["resume_steps"], 2)
        self.assertEqual(first["new_fruit_ids"], [0])
        self.assertIsNone(last["resume_steps"])
        self.assertTrue(last["right_censored"])
        total = fm.aggregate([row])["post_release_resume_steps"]
        self.assertEqual(total["median"], 2)
        self.assertEqual(total["unresumed_at_end"]["numerator"], 1)
        self.assertEqual(total["unresumed_at_end"]["denominator"], 2)
        data[2][1]["gt"]["in_basket"] = [True, True]
        data[2][2]["gt"]["in_basket"] = [True, True]
        data[2][3]["gt"]["in_basket"] = [True, True]
        events = self.analyze(data)["post_release_resume_steps"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["new_fruit_ids"], [0, 1])

    def test_reentry_counts_again_and_initial_basket_is_not_new_entry(self):
        frames = [{"control_step": i, "gt": {"tcp_position": [0, 0, 0], "in_basket": [inside]}}
                  for i, inside in enumerate([True, False, True, False, True])]
        events = fm.tcp_motion_metrics(frames)["post_release_resume_steps"]
        self.assertEqual([e["entry_control_step"] for e in events], [2, 4])
        self.assertTrue(all(e["right_censored"] for e in events))

    def prefix_fixture(self):
        data = self.fixture(34)
        data[0].update(observe_hold_steps=30, total_step_budget=1300)
        for i, frame in enumerate(data[2]):
            frame.update(frame_id=i, exec_start_idx=30, policy_control_step=i-30, is_demo=i<30)
            frame["gt"] = dict(frame["gt"], tcp_position=[0 if i<=30 else .049 if i==31 else .051,0,0])
        for action in data[1][:30]: action["observe_hold"] = True
        return data

    def test_physical_history_prefix_retains_initial_clock_and_onset_snapshot(self):
        data = self.prefix_fixture()
        data[1][30]["action"][0] = .006
        data[1][30]["action"][6] = 6.29
        row = self.analyze(data)
        self.assertFalse(row["engineering_error"], row["errors"])
        self.assertEqual(row["control_steps"], 34)
        self.assertEqual(row["start_idle"], 31)
        self.assertEqual(row["leave_home_step"], 32)
        self.assertEqual(row["deviation_onset_context"]["snapshot_control_step"], 0)

    def test_physical_history_prefix_rejects_missing_frames_or_changed_holds(self):
        data = self.prefix_fixture();data[2].pop(0)
        self.assertTrue(self.analyze(data)["engineering_error"])
        data = self.prefix_fixture();data[1][2]["action"][7] = -1.
        self.assertTrue(self.analyze(data)["engineering_error"])

    def test_legacy_video_prefix_still_excluded(self):
        data = self.fixture(3)
        initial = data[2][0]
        data[2][:0] = [{**initial, "control_step": i, "is_demo": True} for i in (-2,-1)]
        row = self.analyze(data)
        self.assertFalse(row["engineering_error"], row["errors"])
        self.assertEqual(row["control_steps"], 3)


if __name__ == "__main__":
    unittest.main()
