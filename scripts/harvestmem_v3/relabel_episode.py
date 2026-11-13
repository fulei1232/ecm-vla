#!/usr/bin/env python3
"""Relabel one successful R12-style replay with HarvestMem v3 subgoals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from subgoal import OBSERVE, PLACE, RETURN, pick_subgoal, project_world_to_pixel, quantize_unique, simple_subgoal


PICK_PHASES = {"reach", "approach", "grasp", "pull"}
PLACE_PHASES = {"carry", "release", "settle"}
RETURN_PHASES = {"finish", "done", "terminal_hold_target"}


def _write_h5_string(group, name: str, value: str) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=value)


def _target_slot(actor, actor_to_slot):
    if actor is None:
        return None
    actor = int(actor)
    if actor < 0 or actor >= len(actor_to_slot):
        return None
    return int(actor_to_slot[actor])


def _label_for_row(task, row, slot_xy, ripe, next_slots):
    phase = row["phase"]
    raw_subgoal = row.get("subgoal", "")
    slot = _target_slot(row.get("target"), next_slots["actor_to_slot"])
    if phase == "observe" or row["timestep"] < 30:
        return OBSERVE, None
    if phase in RETURN_PHASES or "return to rest" in raw_subgoal:
        return RETURN, None
    if phase in PLACE_PHASES:
        if "return to rest" in raw_subgoal:
            return RETURN, None
        return PLACE, slot
    if phase in PICK_PHASES:
        if slot is None:
            raise ValueError(f"pick frame without target slot at timestep {row['timestep']}")
        return pick_subgoal(task, slot_xy[slot], ripe[slot] if task == "occluded_ripe" else None), slot
    if phase == "inspect":
        # This is a one-frame internal recovery after a windfall. Keep it out of
        # the training vocabulary by attaching it to the next actual pick.
        next_slot = next_slots["inspect_next"].get(int(row["timestep"]))
        if next_slot is None:
            return RETURN, None
        return pick_subgoal(task, slot_xy[next_slot], ripe[next_slot] if task == "occluded_ripe" else None), next_slot
    raise ValueError(f"unsupported raw phase {phase!r} at timestep {row['timestep']}")


def _audit_transitions(rows, labels, slots):
    transitions = []
    for i in range(1, len(rows)):
        if labels[i][0] != labels[i - 1][0]:
            transitions.append((i, labels[i - 1][0], labels[i][0]))
    bad = []
    for i, previous, current in transitions:
        row = rows[i]
        previous_row = rows[i - 1]
        if previous == OBSERVE and current.startswith("pick"):
            if row["phase"] != "reach":
                bad.append({"index": i, "kind": "observe_to_pick", "phase": row["phase"]})
        elif previous.startswith("pick") and current == PLACE:
            actor = row.get("target")
            attached = actor is not None and bool(rows[i]["fruits"][int(actor)].get("attached", False))
            if row["phase"] != "carry" or not attached:
                bad.append({"index": i, "kind": "pick_to_place", "phase": row["phase"], "attached": attached})
        elif previous == PLACE and (current.startswith("pick") or current == RETURN):
            start = i - 1
            while start > 0 and labels[start - 1][0] == PLACE:
                start -= 1
            confirmed = max(int(x.get("n_in_basket", 0)) for x in rows[start:i + 1]) > int(rows[start].get("n_in_basket", 0))
            if not confirmed:
                bad.append({"index": i, "kind": "place_to_next", "phase": row["phase"], "confirmed": False})
            if current.startswith("pick") and row["phase"] != "reach":
                bad.append({"index": i, "kind": "place_to_pick_phase", "phase": row["phase"]})
            if current == RETURN:
                is_legacy_terminal = (row["phase"] == "terminal_hold_target" and i == len(rows) - 1)
                is_explicit_return = row["phase"] == "carry" and "return to rest" in row.get("subgoal", "")
                if not (is_legacy_terminal or is_explicit_return):
                    bad.append({"index": i, "kind": "place_to_return_phase", "phase": row["phase"], "subgoal": row.get("subgoal")})
        else:
            bad.append({"index": i, "kind": "unexpected_transition", "previous": previous, "current": current})
    return transitions, bad


def relabel(directory: Path) -> dict:
    directory = directory.resolve()
    result_path = directory / "result.json"
    gt_path = directory / "demo_gt.jsonl"
    h5_path = directory / "demo.h5"
    result = json.loads(result_path.read_text())
    rows = [json.loads(line) for line in gt_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty GT: {gt_path}")
    task = result["task"]
    actor_to_slot = [int(x) for x in result["slot_of"]]
    geometry = result["slot_geometry"]
    ripe = [bool(x["ripe"]) for x in geometry]
    with h5py.File(h5_path, "r+") as h5:
        ep = h5[f"episode_{result['episode']}"]
        intrinsic = ep["setup/front_camera_intrinsic"][()]
        extrinsic = ep["timestep_0/obs/front_camera_extrinsic"][()]
        projected = {
            slot: project_world_to_pixel(intrinsic, extrinsic, geometry[slot]["pos"])
            for slot in range(len(geometry))
        }
        slot_xy = quantize_unique(projected)

        inspect_next = {}
        for i, row in enumerate(rows):
            if row["phase"] != "inspect":
                continue
            for future in rows[i + 1:]:
                slot = _target_slot(future.get("target"), actor_to_slot)
                if future["phase"] in PICK_PHASES and slot is not None:
                    inspect_next[int(row["timestep"])] = slot
                    break
        context = {"actor_to_slot": actor_to_slot, "inspect_next": inspect_next}
        labels = [_label_for_row(task, row, slot_xy, ripe, context) for row in rows]
        # Occluded R12 terminates immediately after the final confirmed basket
        # entry; retain that task behavior while giving the terminal frame the
        # merged stop/hold label.
        if rows[-1]["phase"] == "terminal_hold_target" and labels[-1][0] == PLACE:
            labels[-1] = (RETURN, None)

        transitions, bad = _audit_transitions(rows, labels, slot_xy)
        if any(row["phase"] == "wait_for_green" for row in rows):
            bad.append({"kind": "wait_for_green_present"})
        if bad:
            raise ValueError(json.dumps({"directory": str(directory), "bad": bad[:20]}, ensure_ascii=False))

        for row, (grounded, slot) in zip(rows, labels):
            row["grounded_subgoal"] = grounded
            row["simple_subgoal"] = simple_subgoal(task, grounded)
            row["grounded_subgoal_online"] = grounded
            row["simple_subgoal_online"] = row["simple_subgoal"]
            row["target_slot"] = slot
            row["n_picked_so_far"] = int(row.get("n_in_basket", 0))
            info = ep[f"timestep_{row['timestep']}"]["info"]
            for name, value in (
                ("grounded_subgoal", grounded),
                ("simple_subgoal", row["simple_subgoal"]),
                ("grounded_subgoal_online", grounded),
                ("simple_subgoal_online", row["simple_subgoal_online"]),
                ("phase", row["phase"]),
            ):
                _write_h5_string(info, name, value)
            for name, value in (("target_slot", -1 if slot is None else int(slot)),
                                ("n_picked_so_far", row["n_picked_so_far"])):
                if name in info:
                    del info[name]
                info.create_dataset(name, data=value)
            if "is_subgoal_boundary" in info:
                del info["is_subgoal_boundary"]
            info.create_dataset("is_subgoal_boundary", data=(row["timestep"] == 0 or grounded != labels[row["timestep"] - 1][0]))
        gt_path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))

    counts = {}
    for grounded, _ in labels:
        counts[grounded] = counts.get(grounded, 0) + 1
    audit = {
        "passed": True,
        "task": task,
        "episode": result["episode"],
        "frames": len(rows),
        "slot_pixels": {str(k): list(v) for k, v in sorted(slot_xy.items())},
        "projection_source": "3d_slot_geometry_plus_front_camera_intrinsic_extrinsic_cv",
        "subgoal_frame_counts": counts,
        "transitions": [{"index": i, "from": a, "to": b} for i, a, b in transitions],
        "transition_mismatches": [],
        "wait_for_green_frames": 0,
        "inspect_frames_mapped_to_next_pick": len(inspect_next),
    }
    (directory / "harvestmem_subgoal_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    result.update({
        "harvestmem_v3_subgoals": True,
        "harvestmem_v3_subgoal_audit": str(directory / "harvestmem_subgoal_audit.json"),
        "harvestmem_v3_slot_pixels": {str(k): list(v) for k, v in sorted(slot_xy.items())},
        "harvestmem_v3_transition_count": len(transitions),
        "harvestmem_v3_transition_mismatches": 0,
        "harvestmem_v3_wait_for_green_frames": 0,
    })
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(relabel(args.dir), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
