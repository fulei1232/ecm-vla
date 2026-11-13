#!/usr/bin/env python3
"""Task-space view of run_fruit_diagnosis.py's replay: tcp position of the policy's predicted joint targets
versus the demonstration's targets (forward kinematics of the RM75-B model), and of holding the current joints.
Forward kinematics is checked against the recorded tcp of every decision frame first.

    SIM_PYTHON scripts/fruit_diagnosis_tcp.py --out results/fruit_diagnosis_20261027
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ECM = Path(__file__).resolve().parents[1]
TASKS = ("FruitCount", "FruitOccludedRipe")


def kinematics():
    import mplib
    sys.path.insert(0, str(ECM))
    from fruit_picking.agent import RM75BEG2
    planner = mplib.Planner(urdf=RM75BEG2.urdf_path, srdf=RM75BEG2.urdf_path.replace(".urdf", ".srdf"), move_group="tcp")
    model = planner.robot.get_pinocchio_model()
    tcp = list(model.get_link_names()).index("tcp")
    assert len(planner.robot.get_qpos()) == 9

    def fk(state8):
        model.compute_forward_kinematics(np.concatenate([state8[:7], [state8[7], state8[7]]]).astype(float))
        return np.asarray(model.get_link_pose(tcp))[:3]
    return fk


def frames_tcp(path, ids):
    want, got = set(int(i) for i in ids), {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["frame_id"] in want:
                got[row["frame_id"]] = row["gt"]["tcp_position"]
    return np.array([got[int(i)] for i in ids])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    fk = kinematics()
    result, fk_err = {}, []
    for split in ("train", "val"):
        for task in TASKS:
            err, hold, near = [], [], []
            files = sorted((a.out / "replay" / split / task).glob("ep*/replay.npz"))
            for f in files:
                z = np.load(f)
                state, pred, demo = z["state"], z["pred"], z["demo"]
                now = np.stack([fk(s) for s in state])
                fk_err.append(np.linalg.norm(now - frames_tcp(f.parent / "raw_frames.jsonl", z["t"]), axis=1))
                grip = np.where(np.isfinite(demo[..., 7]), state[:, None, 7].repeat(16, 1), np.nan)
                tp = np.full(demo.shape[:2] + (3,), np.nan)
                td = np.full_like(tp, np.nan)
                for i, h in zip(*np.nonzero(np.isfinite(demo[..., 0]))):
                    tp[i, h] = fk(np.append(pred[i, h, :7], grip[i, h]))
                    td[i, h] = fk(np.append(demo[i, h, :7], grip[i, h]))
                err.append(np.linalg.norm(tp - td, axis=-1))
                hold.append(np.linalg.norm(now[:, None] - td, axis=-1))
                near.append(z["distance"] <= 0.10)
            if not files:
                continue
            err, hold, near = (np.concatenate(x) for x in (err, hold, near))
            cm = lambda x: {f"h{h + 1}": round(100 * float(np.nanmedian(x[:, h])), 2) for h in (0, 7, 15)}
            p90 = lambda x: {f"h{h + 1}": round(100 * float(np.nanpercentile(x[:, h], 90)), 2) for h in (0, 7, 15)}
            result[f"{split}/{task}"] = {"decisions": len(err), "tcp_err_cm_median": cm(err), "tcp_err_cm_p90": p90(err),
                                         "hold_cm_median": cm(hold), "near_fruit_tcp_err_cm_median": cm(err[near]),
                                         "near_fruit_decisions": int(near.sum())}
    fk_err = np.concatenate(fk_err)
    out = {"fk_check_max_cm": round(100 * float(fk_err.max()), 4), "replay_tcp": result}
    (a.out / "tcp_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    assert fk_err.max() < 1e-3, "forward kinematics does not reproduce the recorded tcp"


if __name__ == "__main__":
    main()
