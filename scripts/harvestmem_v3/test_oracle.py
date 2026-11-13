import unittest

from oracle import HarvestMemOracle


def fruit(y, *, attached=False, present=True, wind_detached=False, in_basket=False,
          x=0.4, z=0.5):
    return {"pos": [x, y, z], "attached": attached, "present": present,
            "wind_detached": wind_detached, "in_basket": in_basket}


def snapshot(timestep, fruits, *, basket=0, tcp=(0.4, 0.0, 0.5), gripper=(0.02, 0.02)):
    return {"timestep": timestep, "fruits": fruits, "n_in_basket": basket,
            "tcp_pos": list(tcp), "gripper_qpos": list(gripper)}


class OracleTest(unittest.TestCase):
    def make(self, n=2):
        return HarvestMemOracle(
            "count", n, [0, 1, 2], {0: (10, 20), 1: (30, 40), 2: (50, 60)},
            [True, True, True], [[0.4, 0.2, 0.5], [0.4, 0.8, 0.5], [0.4, 1.1, 0.5]],
        )

    def test_continuous_pull_uses_ten_frame_fallback(self):
        oracle = self.make()
        base = [fruit(0.2), fruit(0.8), fruit(1.1, present=False)]
        self.assertEqual(oracle.step(snapshot(30, base)).target_slot, 1)
        output = None
        for offset in range(1, 11):
            moving_y = 0.69 - 0.002 * offset
            moving = [fruit(0.2), fruit(moving_y, attached=True),
                      fruit(1.1, present=False)]
            output = oracle.step(snapshot(30 + offset, moving,
                                          tcp=(0.4, moving_y, 0.5)))
        self.assertEqual(output.phase, "place")
        self.assertIn("pick_place_timeout_fallback", output.events)

    def test_already_in_basket_advances_to_next_pick(self):
        oracle = self.make()
        base = [fruit(0.2), fruit(0.8), fruit(1.1, present=False)]
        oracle.step(snapshot(30, base))
        placed = [fruit(0.2), fruit(0.6, attached=True, in_basket=True), fruit(1.1, present=False)]
        output = oracle.step(snapshot(31, placed, basket=1, tcp=(0.4, 0.6, 0.5)))
        self.assertIn("basket_confirmed", output.events)
        for timestep in range(32, 35):
            oracle.step(snapshot(timestep, placed, basket=1, gripper=(0.04, 0.04)))
        output = oracle.step(snapshot(35, placed, basket=1, gripper=(0.04, 0.04)))
        self.assertEqual(output.phase, "pick")
        self.assertEqual(output.target_slot, 0)

    def test_partially_open_gripper_uses_twenty_frame_fallback(self):
        oracle = self.make()
        base = [fruit(0.2), fruit(0.8), fruit(1.1, present=False)]
        oracle.step(snapshot(30, base))
        placed = [fruit(0.2), fruit(0.6, attached=True, in_basket=True), fruit(1.1, present=False)]
        oracle.step(snapshot(31, placed, basket=1))
        output = None
        for timestep in range(32, 51):
            output = oracle.step(snapshot(timestep, placed, basket=1, gripper=(0.035, 0.035)))
        self.assertEqual(output.phase, "pick")
        self.assertIn("place_timeout_fallback", output.events)

    def test_new_higher_fruit_does_not_change_locked_target(self):
        oracle = self.make()
        first = [fruit(0.2), fruit(0.8), fruit(1.1, present=False)]
        self.assertEqual(oracle.step(snapshot(30, first)).target_slot, 1)
        replenished = [fruit(0.2), fruit(0.8), fruit(1.1)]
        self.assertEqual(oracle.step(snapshot(31, replenished)).target_slot, 1)

    def test_count_reopens_after_no_target_before_n(self):
        oracle = self.make(n=2)
        empty = [fruit(0.2, present=False), fruit(0.8, present=False), fruit(1.1, present=False)]
        self.assertEqual(oracle.step(snapshot(30, empty)).phase, "return")
        reopened = [fruit(0.2), fruit(0.8, present=False), fruit(1.1, present=False)]
        output = oracle.step(snapshot(31, reopened))
        self.assertEqual(output.phase, "pick")
        self.assertIn("reopen_after_no_target", output.events)

    def test_drop_event_reselects(self):
        oracle = self.make()
        base = [fruit(0.2), fruit(0.8), fruit(1.1, present=False)]
        oracle.step(snapshot(30, base))
        carrying = [fruit(0.2), fruit(0.6, attached=True), fruit(1.1, present=False)]
        oracle.step(snapshot(31, carrying, tcp=(0.4, 0.6, 0.5)))
        dropped = [fruit(0.2), fruit(-0.2, attached=True), fruit(1.1, present=False)]
        output = None
        for timestep in range(32, 43):
            output = oracle.step(snapshot(timestep, dropped, tcp=(0.4, 0.4, 0.5)))
        self.assertEqual(output.phase, "pick")
        self.assertIn("drop", output.events)

    def test_wrong_fruit_detached_enters_place(self):
        oracle = self.make()
        base = [fruit(0.2), fruit(0.8), fruit(1.1, present=False)]
        self.assertEqual(oracle.step(snapshot(30, base)).target_slot, 1)
        wrong = [fruit(0.2, attached=True), fruit(0.8), fruit(1.1, present=False)]
        output = oracle.step(snapshot(31, wrong))
        self.assertEqual(output.phase, "place")
        self.assertEqual(output.target_slot, 0)
        self.assertIn("wrong_fruit_detached", output.events)

    def test_place_stall_after_three_hundred_frames(self):
        oracle = self.make()
        base = [fruit(0.2), fruit(0.8), fruit(1.1, present=False)]
        oracle.step(snapshot(30, base))
        carrying = [fruit(0.2), fruit(0.6, attached=True), fruit(1.1, present=False)]
        oracle.step(snapshot(31, carrying, tcp=(0.4, 0.6, 0.5)))
        output = None
        for offset in range(1, 302):
            moving_y = -0.2 + 0.003 * offset
            stalled = [fruit(0.2), fruit(moving_y, attached=True), fruit(1.1, present=False)]
            output = oracle.step(snapshot(31 + offset, stalled,
                                          tcp=(0.4, moving_y + 0.2, 0.5)))
        self.assertEqual(output.phase, "pick")
        self.assertIn("oracle_stall", output.events)


if __name__ == "__main__":
    unittest.main()
