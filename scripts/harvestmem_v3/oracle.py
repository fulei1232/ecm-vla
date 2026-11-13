#!/usr/bin/env python3
"""Causal HarvestMem v3 oracle and offline dev-replay validator."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

OBSERVE = "observe the fruit tree"
PLACE = "place the fruit in the basket"
RETURN = "return to rest and hold still"

PULL_COMPLETE_DISPLACEMENT = 0.105
PULL_IMMEDIATE_DISPLACEMENT = 0.15
PULL_TIMEOUT_FRAMES = 10
PULL_STOP_VELOCITY = 0.001
RELEASE_OPEN_THRESHOLD = 0.0392
RELEASE_OPEN_STREAK = 3
PLACE_CONFIRM_TIMEOUT = 20
DROP_DISTANCE = 0.05
DROP_STILL_VELOCITY = 0.002
DROP_STILL_FRAMES = 10
PLACE_STALL_FRAMES = 300

FALLBACK_EVENTS = {
    "pick_place_timeout_fallback",
    "pick_place_displacement_fallback",
    "place_timeout_fallback",
}


@dataclass(frozen=True)
class OracleOutput:
    subgoal: str
    phase: str
    target_slot: int | None
    events: tuple[str, ...] = ()

    @property
    def event(self) -> str | None:
        return self.events[0] if self.events else None


def _distance(a: Iterable[float], b: Iterable[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _collapse(values: Iterable[str]) -> list[str]:
    collapsed: list[str] = []
    for value in values:
        if not collapsed or value != collapsed[-1]:
            collapsed.append(value)
    return collapsed


class HarvestMemOracle:
    """Causal state machine shared by online evaluation and offline replay."""

    def __init__(self, task: str, n_pick: int, slot_of: list[int],
                 slot_xy: dict[int, tuple[int, int]], initial_ripe: list[bool],
                 slot_pos: list[list[float]] | None = None):
        if task not in {"count", "occluded_ripe"}:
            raise ValueError(f"unsupported task: {task}")
        self.task = task
        self.n_pick = int(n_pick)
        self.slot_of = [int(x) for x in slot_of]
        self.slot_xy = {int(k): (int(v[0]), int(v[1])) for k, v in slot_xy.items()}
        self.initial_ripe = [bool(x) for x in initial_ripe]
        self.slot_pos = slot_pos
        self.reset()

    def reset(self) -> None:
        self.mode = "observe"
        self.target_actor: int | None = None
        self.target_slot: int | None = None
        self.prev_attached: list[bool] | None = None
        self.prev_tcp_pos: list[float] | None = None
        self.prev_target_pos: list[float] | None = None
        self.detached_frames = 0
        self.place_frames = 0
        self.basket_confirmed = False
        self.basket_confirmed_frames = 0
        self.release_open_streak = 0
        self.place_ready = False
        self.drop_streak = 0
        self.return_absorbing = False
        self.event_counts: Counter[str] = Counter()

    def _emit(self, subgoal: str, phase: str, target_slot: int | None,
              events: Iterable[str] = ()) -> OracleOutput:
        event_tuple = tuple(events)
        self.event_counts.update(event_tuple)
        return OracleOutput(subgoal, phase, target_slot, event_tuple)

    def _slot(self, actor: int) -> int:
        if actor < 0 or actor >= len(self.slot_of):
            raise ValueError(f"actor index out of range: {actor}")
        return self.slot_of[actor]

    @staticmethod
    def _hanging(fruit: dict[str, Any]) -> bool:
        # Here attached=True means detached from the branch.
        return bool(fruit.get("present", False)) and not bool(fruit.get("attached", False)) \
            and not bool(fruit.get("wind_detached", False)) \
            and not bool(fruit.get("in_basket", False))

    def _pick_subgoal(self, slot: int) -> str:
        x, y = self.slot_xy[slot]
        if self.task == "occluded_ripe":
            return f"pick the red fruit at <{x}, {y}>"
        return f"pick the fruit at <{x}, {y}>"

    def _eligible(self, snapshot: dict[str, Any]) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        for actor, fruit in enumerate(snapshot["fruits"]):
            if not self._hanging(fruit):
                continue
            slot = self._slot(actor)
            if self.task == "occluded_ripe" and not self.initial_ripe[slot]:
                continue
            candidates.append((actor, slot))
        return candidates

    def _select_target(self, snapshot: dict[str, Any],
                       events: Iterable[str] = ()) -> OracleOutput:
        candidates = self._eligible(snapshot)
        if not candidates:
            self.mode = "return"
            self.target_actor = None
            self.target_slot = None
            self.return_absorbing = self.task != "count" or int(snapshot.get("n_in_basket", 0)) >= self.n_pick
            return self._emit(RETURN, "return", None, (*events, "no_eligible_target"))
        actor, slot = max(candidates, key=lambda pair: float(snapshot["fruits"][pair[0]]["pos"][1]))
        self.mode = "pick"
        self.return_absorbing = False
        self.target_actor = actor
        self.target_slot = slot
        self.prev_tcp_pos = None
        self.prev_target_pos = None
        self.detached_frames = 0
        return self._emit(self._pick_subgoal(slot), "pick", slot, (*events, "target_selected"))

    def _pick_output(self, events: Iterable[str] = ()) -> OracleOutput:
        if self.target_slot is None:
            raise RuntimeError("pick output requested without a target")
        return self._emit(self._pick_subgoal(self.target_slot), "pick", self.target_slot, events)

    def _stop(self, snapshot: dict[str, Any]) -> bool:
        if self.task == "count":
            return int(snapshot.get("n_in_basket", 0)) >= self.n_pick
        return not any(
            self._hanging(fruit) and self.initial_ripe[self._slot(actor)]
            for actor, fruit in enumerate(snapshot["fruits"])
        )

    def _enter_place(self, snapshot: dict[str, Any], actor: int,
                     events: Iterable[str]) -> OracleOutput:
        self.mode = "place"
        self.target_actor = actor
        self.target_slot = self._slot(actor)
        self.detached_frames = 0
        self.place_frames = 0
        self.basket_confirmed = False
        self.basket_confirmed_frames = 0
        self.release_open_streak = 0
        self.place_ready = False
        self.drop_streak = 0
        fruit = snapshot["fruits"][actor]
        self.prev_target_pos = [float(x) for x in fruit["pos"]]
        place_events = list(events)
        if bool(fruit.get("in_basket", False)):
            self.basket_confirmed = True
            self.basket_confirmed_frames = 1
            place_events.append("basket_confirmed")
        return self._emit(PLACE, "place", self.target_slot, place_events)

    def _advance_after_place(self, snapshot: dict[str, Any],
                             events: Iterable[str] = ()) -> OracleOutput:
        if self._stop(snapshot):
            self.mode = "return"
            self.target_actor = None
            self.target_slot = None
            return self._emit(RETURN, "return", None, events)
        self.target_actor = None
        self.target_slot = None
        return self._select_target(snapshot, events)

    def _new_robot_detachments(self, snapshot: dict[str, Any]) -> list[int]:
        if self.prev_attached is None:
            return []
        result = []
        for actor, fruit in enumerate(snapshot["fruits"]):
            previous = self.prev_attached[actor] if actor < len(self.prev_attached) else False
            if (bool(fruit.get("attached", False)) and not previous
                    and not bool(fruit.get("wind_detached", False))):
                result.append(actor)
        return result

    def step(self, snapshot: dict[str, Any]) -> OracleOutput:
        timestep = int(snapshot["timestep"])
        new_detachments = self._new_robot_detachments(snapshot)

        if timestep < 30:
            output = self._emit(OBSERVE, "observe", None)
        elif self.mode == "return":
            if (self.task == "count" and not self.return_absorbing
                    and int(snapshot.get("n_in_basket", 0)) < self.n_pick
                    and self._eligible(snapshot)):
                self.mode = "pick"
                output = self._select_target(snapshot, ("reopen_after_no_target",))
            else:
                output = self._emit(RETURN, "return", None)
        else:
            if self.mode == "observe":
                self.mode = "pick"
            if self.mode == "pick":
                output = self._step_pick(snapshot, new_detachments)
            elif self.mode == "place":
                output = self._step_place(snapshot)
            else:
                raise AssertionError(f"unknown oracle mode: {self.mode}")

        self.prev_attached = [bool(fruit.get("attached", False)) for fruit in snapshot["fruits"]]
        tcp = snapshot.get("tcp_pos")
        self.prev_tcp_pos = None if tcp is None else [float(x) for x in tcp]
        if self.target_actor is not None and self.target_actor < len(snapshot["fruits"]):
            pos = snapshot["fruits"][self.target_actor].get("pos")
            self.prev_target_pos = None if pos is None else [float(x) for x in pos]
        return output

    def _step_pick(self, snapshot: dict[str, Any], new_detachments: list[int]) -> OracleOutput:
        if self.target_actor is None:
            return self._select_target(snapshot)

        wrong = [actor for actor in new_detachments if actor != self.target_actor]
        if wrong:
            return self._enter_place(
                snapshot, wrong[0], ("wrong_fruit_detached", "robot_detached")
            )

        fruit = snapshot["fruits"][self.target_actor]
        if bool(fruit.get("wind_detached", False)) or not bool(fruit.get("present", False)):
            self.target_actor = None
            self.target_slot = None
            return self._select_target(snapshot)
        if not bool(fruit.get("attached", False)):
            self.detached_frames = 0
            return self._pick_output()

        slot_pos = None if self.slot_pos is None else self.slot_pos[self.target_slot]
        displacement = 0.0 if slot_pos is None else _distance(fruit["pos"], slot_pos)
        tcp = snapshot.get("tcp_pos")
        tcp_velocity = 0.0 if tcp is None or self.prev_tcp_pos is None else _distance(tcp, self.prev_tcp_pos)

        if slot_pos is None or displacement >= PULL_COMPLETE_DISPLACEMENT:
            self.detached_frames += 1
        else:
            self.detached_frames = 0

        if slot_pos is None or (displacement >= PULL_COMPLETE_DISPLACEMENT
                                and tcp_velocity <= PULL_STOP_VELOCITY):
            return self._enter_place(snapshot, self.target_actor, ("robot_detached",))
        if displacement >= PULL_IMMEDIATE_DISPLACEMENT:
            return self._enter_place(
                snapshot, self.target_actor,
                ("robot_detached", "pick_place_displacement_fallback"),
            )
        if self.detached_frames >= PULL_TIMEOUT_FRAMES:
            return self._enter_place(
                snapshot, self.target_actor,
                ("robot_detached", "pick_place_timeout_fallback"),
            )
        return self._pick_output()

    def _step_place(self, snapshot: dict[str, Any]) -> OracleOutput:
        if self.target_actor is None:
            return self._select_target(snapshot, ("oracle_stall",))
        if self.place_ready:
            return self._advance_after_place(snapshot)

        self.place_frames += 1
        fruit = snapshot["fruits"][self.target_actor]
        events: list[str] = []
        if bool(fruit.get("in_basket", False)) and not self.basket_confirmed:
            self.basket_confirmed = True
            self.basket_confirmed_frames = 0
            events.append("basket_confirmed")

        if self.basket_confirmed and self.task == "occluded_ripe" and self._stop(snapshot):
            self.mode = "return"
            self.target_actor = None
            self.target_slot = None
            return self._emit(RETURN, "return", None, events)

        if self.basket_confirmed:
            self.basket_confirmed_frames += 1
            gripper = snapshot.get("gripper_qpos", [])
            if len(gripper) >= 2 and min(float(x) for x in gripper[:2]) >= RELEASE_OPEN_THRESHOLD:
                self.release_open_streak += 1
            else:
                self.release_open_streak = 0
            if self.release_open_streak >= RELEASE_OPEN_STREAK:
                self.place_ready = True
                return self._emit(PLACE, "place", self.target_slot, events)
            if self.basket_confirmed_frames >= PLACE_CONFIRM_TIMEOUT:
                return self._advance_after_place(snapshot, (*events, "place_timeout_fallback"))
            return self._emit(PLACE, "place", self.target_slot, events)

        tcp = snapshot.get("tcp_pos")
        pos = fruit.get("pos")
        far_from_tcp = tcp is not None and pos is not None and _distance(tcp, pos) > DROP_DISTANCE
        movement = None if pos is None or self.prev_target_pos is None else _distance(pos, self.prev_target_pos)
        if far_from_tcp and movement is not None and movement < DROP_STILL_VELOCITY:
            self.drop_streak += 1
        else:
            self.drop_streak = 0
        if self.drop_streak >= DROP_STILL_FRAMES:
            self.target_actor = None
            self.target_slot = None
            return self._select_target(snapshot, ("drop",))
        if self.place_frames > PLACE_STALL_FRAMES and far_from_tcp:
            self.target_actor = None
            self.target_slot = None
            return self._select_target(snapshot, ("oracle_stall",))
        return self._emit(PLACE, "place", self.target_slot)


def _successful_attempt(root: Path, episode: int) -> Path:
    for result_path in sorted(root.glob(f"ep{episode}_attempt*/result.json")):
        if json.loads(result_path.read_text()).get("success"):
            return result_path.parent
    raise FileNotFoundError(f"no successful attempt for episode {episode} under {root}")


def _make_oracle(result: dict[str, Any]) -> HarvestMemOracle:
    slot_xy = {int(k): tuple(v) for k, v in result["harvestmem_v3_slot_pixels"].items()}
    ripe = [bool(x["ripe"]) for x in result["slot_geometry"]]
    slot_pos = [list(x["pos"]) for x in result["slot_geometry"]]
    return HarvestMemOracle(result["task"], result["N"], result["slot_of"], slot_xy, ripe, slot_pos)


def validate_replay(directory: Path) -> dict[str, Any]:
    result = json.loads((directory / "result.json").read_text())
    rows = [json.loads(line) for line in (directory / "demo_gt.jsonl").read_text().splitlines() if line.strip()]
    oracle = _make_oracle(result)
    mismatches: list[dict[str, Any]] = []
    predictions: list[OracleOutput] = []
    for index, row in enumerate(rows):
        prediction = oracle.step(row)
        predictions.append(prediction)
        expected = row["grounded_subgoal"]
        if prediction.subgoal != expected:
            mismatches.append({"index": index, "timestep": row["timestep"],
                               "predicted": prediction.subgoal, "expected": expected,
                               "predicted_phase": prediction.phase, "raw_phase": row.get("phase")})

    expected_sequence = _collapse(row["grounded_subgoal"] for row in rows)
    predicted_sequence = _collapse(prediction.subgoal for prediction in predictions)
    expected_picks = [value for value in expected_sequence if value.startswith("pick")]
    selected_picks = [prediction.subgoal for prediction in predictions
                      if "target_selected" in prediction.events]
    target_errors = sum(a != b for a, b in zip(expected_picks, selected_picks)) \
        + abs(len(expected_picks) - len(selected_picks))
    sequence_match = predicted_sequence == expected_sequence

    transitions = [i for i in range(1, len(rows))
                   if rows[i]["grounded_subgoal"] != rows[i - 1]["grounded_subgoal"]]
    boundary_mismatches = 0
    max_boundary_distance = 0
    for mismatch in mismatches:
        distance = min((abs(mismatch["index"] - transition) for transition in transitions),
                       default=len(rows))
        mismatch["nearest_transition_distance"] = distance
        max_boundary_distance = max(max_boundary_distance, distance)
        if distance <= 5:
            boundary_mismatches += 1

    return {
        "task": result["task"], "episode": result["episode"], "N": result["N"],
        "frames": len(rows), "matching_frames": len(rows) - len(mismatches),
        "frame_accuracy": (len(rows) - len(mismatches)) / len(rows),
        "mismatches": mismatches, "boundary_mismatches": boundary_mismatches,
        "max_boundary_distance": max_boundary_distance,
        "target_errors": target_errors,
        "count_errors": 0 if sequence_match else 1,
        "sequence_match": sequence_match,
        "expected_sequence": expected_sequence,
        "predicted_sequence": predicted_sequence,
        "event_counts": dict(oracle.event_counts),
    }


def validate_dev(root: Path, episodes: range = range(200, 250)) -> dict[str, Any]:
    by_task = {
        task: [validate_replay(_successful_attempt(root / task, episode)) for episode in episodes]
        for task in ("count", "occluded_ripe")
    }
    summary: dict[str, Any] = {"by_task": {}, "passed": True}
    for task, rows in by_task.items():
        frames = sum(row["frames"] for row in rows)
        matches = sum(row["matching_frames"] for row in rows)
        event_counts: Counter[str] = Counter()
        for row in rows:
            event_counts.update(row["event_counts"])
        task_summary = {
            "episodes": len(rows), "frames": frames, "matching_frames": matches,
            "frame_accuracy": matches / frames if frames else 0.0,
            "target_errors": sum(row["target_errors"] for row in rows),
            "count_errors": sum(row["count_errors"] for row in rows),
            "sequence_matches": sum(bool(row["sequence_match"]) for row in rows),
            "boundary_mismatches": sum(row["boundary_mismatches"] for row in rows),
            "max_boundary_distance": max(row["max_boundary_distance"] for row in rows),
            "event_counts": dict(event_counts),
            "fallback_event_counts": {event: event_counts[event] for event in sorted(FALLBACK_EVENTS)},
            "by_N": {}, "episodes_detail": rows,
        }
        for n in sorted({int(row["N"]) for row in rows}):
            subset = [row for row in rows if int(row["N"]) == n]
            f = sum(row["frames"] for row in subset)
            m = sum(row["matching_frames"] for row in subset)
            task_summary["by_N"][str(n)] = {
                "episodes": len(subset), "frames": f, "frame_accuracy": m / f if f else 0.0,
                "target_errors": sum(row["target_errors"] for row in subset),
                "count_errors": sum(row["count_errors"] for row in subset),
                "sequence_matches": sum(bool(row["sequence_match"]) for row in subset),
            }
        summary["by_task"][task] = task_summary
        if (task_summary["frame_accuracy"] < 0.99
                or task_summary["target_errors"]
                or task_summary["count_errors"]
                or task_summary["sequence_matches"] != task_summary["episodes"]
                or task_summary["boundary_mismatches"] != frames - matches):
            summary["passed"] = False
    summary["sequence_matches"] = sum(value["sequence_matches"] for value in summary["by_task"].values())
    summary["episodes"] = sum(value["episodes"] for value in summary["by_task"].values())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = validate_dev(args.dev_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"passed": summary["passed"],
                      "sequence_matches": f"{summary['sequence_matches']}/{summary['episodes']}",
                      "accuracy": {k: v["frame_accuracy"] for k, v in summary["by_task"].items()}},
                     ensure_ascii=False), flush=True)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
