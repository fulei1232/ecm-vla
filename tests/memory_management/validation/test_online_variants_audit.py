"""CPU selection audit: replay every V-A condition on recorded official rollouts (plan §0A.3 gate 3)."""
from src.memory_management.validation.online_variants import CONDITIONS, EpisodeState, choose
from src.memory_management.validation.partition import partition


def test_selection_audit_on_recorded_rollouts(m3_episodes):
    totals = {c: 0 for c in CONDITIONS}
    for ep in m3_episodes:
        states = {c: EpisodeState(ep["task"], ep["episode_id"], 7, ep["exec_start_idx"]) for c in CONDITIONS}
        for d in ep["decisions"]:
            F = d["available_max_frame_id"]
            rows = [r for r in ep["rows"] if r["frame_id"] <= F]
            uniform = d["selected_frame_ids"]
            part = partition(F)
            logs = {c: choose(c, uniform, F, rows, states[c])[1] for c in CONDITIONS}
            for c, log in logs.items():
                assert log["n_frames"] == min(F + 1, 32)
                if c in ("D4B", "RNDB", "RNDC", "JITB"):
                    assert log["protected_present"], (c, ep["task"], F)
                totals[c] += 1
            assert logs["D0a"]["selected"] == uniform == logs["D0b"]["selected"]
            assert logs["D4L"]["pins"] == logs["D4B"]["pins"] == logs["RNDB"]["pins"] == logs["RNDC"]["pins"]
            assert logs["D4B"]["pin_overflow"] == logs["RNDB"]["pin_overflow"] == logs["RNDC"]["pin_overflow"]
            if not logs["D4B"]["pin_overflow"] and not logs["RNDB"].get("rndb_infeasible"):
                assert logs["RNDB"]["deleted_d0_frames"] == logs["D4B"]["deleted_d0_frames"]
                assert len(logs["RNDC"]["deleted_d0_frames"]) == len(logs["D4B"]["deleted_d0_frames"])
                for c in ("D4B", "RNDB", "RNDC", "JITB"):
                    if logs[c]["phase_borrow"] == 0:
                        assert logs[c]["n_demo_slots"] == logs[c]["n_demo_slots_d0"], (c, ep["task"], F)
            if part.full:
                assert len(logs["RNDB"]["added_frames"]) == len(logs["D4B"]["added_frames"])
    assert all(v > 100 for v in totals.values())
