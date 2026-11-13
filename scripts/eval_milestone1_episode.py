#!/usr/bin/env python3
"""Reuse the audited official V-A evaluator, additionally record actual environment timebase."""
from pathlib import Path
import sys

from v_a_common import POLICY, SIM, atomic_json


def main():
    sys.path[:0] = [str(POLICY / "examples/robomme"), str(POLICY / "packages/openpi-client/src"), str(SIM / "src")]
    from env_runner import EnvRunner
    import eval_v_a_episode
    original_make = EnvRunner.make_env

    def make_env(self, episode_id):
        result = original_make(self, episode_id)
        env = self.env.unwrapped
        control_freq = getattr(env, "control_freq", None)
        sim_freq = getattr(env, "sim_freq", None)
        atomic_json(Path(self.video_save_dir) / "timebase.json", {
            "control_freq_hz": float(control_freq) if control_freq is not None else None,
            "sim_freq_hz": float(sim_freq) if sim_freq is not None else None,
            "source": "live env.unwrapped.control_freq / sim_freq",
            "observation_rate": "one recorded observation per executed control action; video prefix separately indexed",
            "capture_time_in_frame_record": None,
            "note": "control frequency does not establish demonstration playback timestamps"})
        return result

    EnvRunner.make_env = make_env
    return eval_v_a_episode.main()


if __name__ == "__main__":
    sys.exit(main())
