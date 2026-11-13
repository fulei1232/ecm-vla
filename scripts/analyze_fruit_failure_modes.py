#!/usr/bin/env python3
"""Read archived rollout logs using HANDOFF_lowlevel_rescue_20261102 §5.

Example: --episodes-root results/.../episodes/K0 --pattern 'ep{e}_seed{s}'
         --episodes 200-219 --seeds 7 --out results/.../baseline
Only logs are read; no simulator, model, layout or image shards are opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import statistics

from v_a_common import atomic_json, sha256, timestamp

TASKS = ("FruitCount", "FruitOccludedRipe")
CONTEXTS = ("carrying_fruit", "closed_empty", "open_after_basket", "open_before_first_detach")


def jsonl(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def numbers(value):
    if isinstance(value, list):
        for item in value:
            yield from numbers(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield float(value)


def motion_metrics(actions):
    """Steps use the log's one-based control_step; freeze counts steps >60."""
    first, run, longest = None, 0, 0
    for prev, cur in zip(actions, actions[1:]):
        moving = max(abs(x-y) for x, y in zip(cur["action"][:7], prev["action"][:7])) > 0.005
        step = cur["control_step"]
        if moving and first is None:
            first = step
        if step > 60:
            run = 0 if moving else run + 1
            longest = max(longest, run)
    return first, longest


def tcp_motion_metrics(frames):
    """Revision 1 §9.5, using the control-frame clock and strictly >5 cm.

    Basket entry is a false->true transition of any fruit's in_basket flag.
    Simultaneous entries are one event with all fruit ids; re-entry counts again.
    Each event has its own TCP anchor, including the last entry of a successful
    episode. Unfinished resumptions are right-censored, never filled with zero.
    """
    initial = frames[0]["gt"]["tcp_position"]
    leave = next((r["control_step"] for r in frames
                  if math.dist(initial, r["gt"]["tcp_position"]) > 0.05), None)
    events = []
    for index, (prev, cur) in enumerate(zip(frames, frames[1:]), start=1):
        before, after = prev["gt"]["in_basket"], cur["gt"]["in_basket"]
        if len(before) != len(after):
            raise ValueError("basket flag count changed within episode")
        entered = [i for i, (a, b) in enumerate(zip(before, after)) if not a and b]
        if not entered:
            continue
        anchor = cur["gt"]["tcp_position"]
        resumed = next((r["control_step"] for r in frames[index+1:]
                        if math.dist(anchor, r["gt"]["tcp_position"]) > 0.05), None)
        events.append({"entry_control_step": cur["control_step"], "new_fruit_ids": entered,
                       "anchor_tcp_position": anchor, "resume_control_step": resumed,
                       "resume_steps": resumed-cur["control_step"] if resumed is not None else None,
                       "right_censored": resumed is None,
                       "observed_steps_after_entry": frames[-1]["control_step"]-cur["control_step"]})
    return {"leave_home_step": leave, "never_left_home": leave is None,
            "post_release_resume_steps": events}


def onset_context(actions, frames, onset):
    # Snapshot at control step onset-32 (raw frames are post-action, with an initial t=0 frame).
    step = max(0, onset - 32)
    preceding = [r for r in frames if r["control_step"] <= step]
    if not preceding:
        raise ValueError("no raw snapshot for deviation onset")
    frame = preceding[-1]
    gt = frame["gt"]
    tcp = gt["tcp_position"]
    carried = any(d and not inside and math.dist(pos, tcp) < 0.04
                  for d, inside, pos in zip(gt["detached"], gt["in_basket"], gt["fruit_positions"]))
    earlier = [r for r in actions if r["control_step"] <= frame["control_step"]]
    # Initial open gripper: legacy action convention is <0 closed, >=0 open.
    grip = earlier[-1]["action"][7] if earlier else 1.0
    context = ("carrying_fruit" if carried else "closed_empty" if grip < 0 else
               "open_after_basket" if any(gt["in_basket"]) else "open_before_first_detach")
    return {"category": context, "requested_control_step": step,
            "snapshot_control_step": frame["control_step"], "gripper_command": grip,
            "detached": sum(gt["detached"]), "in_basket": sum(gt["in_basket"])}


def physical_frames(result, actions, recorded):
    """History-only observations can still belong to physical control time.

    Rescue stage3's first30 frames accompany actual holds counted in the
    episode budget. Older video demonstrations precede physical time and
    retain the existing exclusion rule. Require the explicit stage3 contract
    and verify every clock/hold instead of silently accepting missing frames.
    """
    prefix = result.get("observe_hold_steps", 0)
    if not prefix:
        return [r for r in recorded if not r.get("is_demo", False)]
    if type(prefix) is not int or prefix != 30 or result.get("total_step_budget") != 1300:
        raise ValueError("invalid physical observe-prefix contract")
    if len(actions) < prefix or len(actions) > 1300:
        raise ValueError("physical-prefix control budget mismatch")
    for index, frame in enumerate(recorded):
        if (frame.get("frame_id") != index or frame.get("control_step") != index
                or frame.get("exec_start_idx") != prefix
                or frame.get("policy_control_step") != index-prefix
                or frame.get("is_demo") != (index < prefix)):
            raise ValueError("physical-prefix frame clock/flags mismatch")
    holds = actions[:prefix]
    first = holds[0]["action"]
    if (len(first) != 8 or first[7] != 1.0
            or any(not a.get("observe_hold") or a["action"] != first for a in holds)):
        raise ValueError("physical-prefix actions are not fixed open-gripper holds")
    return recorded


def analyze_episode(directory, q7_center=-1.56):
    try:
        result = json.loads((directory / "result.json").read_text())
    except (OSError, ValueError) as exc:
        result = {"outcome": "missing_or_invalid_result", "error": str(exc)}
    row = {"directory": str(directory.resolve()), "task": result.get("task", directory.parent.name),
           "episode_id": result.get("episode_id"), "model_seed": result.get("model_seed"),
           "outcome": result.get("outcome"), "errors": [],
           "sources": {name: str((directory / name).resolve()) for name in
                       ("result.json", "control_actions.jsonl", "raw_frames.jsonl", "event_signal.jsonl")}}
    if result.get("error") or result.get("outcome") not in ("success", "fail", "timeout"):
        row["errors"].append(result.get("error") or "invalid rollout outcome")
    for metric in ("correct_in_basket", "wrong_in_basket", "first_target_in_basket", "never_detached",
                   "near_miss", "wrist_over_limit", "wrist_deviation_executed", "wrist_deviation_proposed",
                   "deviation_onset_context", "start_idle", "freeze_200", "longest_freeze_after_60",
                   "basket_unsuccessful", "count_exact", "min_target_distance", "guard_clip_rate",
                   "leave_home_step", "never_left_home", "post_release_resume_steps"):
        row[metric] = None
    try:
        actions = list(jsonl(directory / "control_actions.jsonl"))
        frames = physical_frames(result, actions, list(jsonl(directory / "raw_frames.jsonl")))
        if not actions or not frames:
            raise ValueError("empty actions or frames")
        if [r["control_step"] for r in actions] != list(range(1, len(actions)+1)):
            raise ValueError("noncontiguous control steps")
        if [r["control_step"] for r in frames] != list(range(len(actions)+1)):
            raise ValueError("raw frame control steps disagree with actions")
        row.update(tcp_motion_metrics(frames))
        for action in actions:
            for key in ("action", "proposed_action"):
                if key in action and (len(action[key]) != 8 or not all(math.isfinite(x) for x in action[key])):
                    raise ValueError(f"invalid {key}")
        if result.get("executed_steps", len(actions)) != len(actions):
            raise ValueError("executed_steps disagrees with actions")
        gt = frames[-1]["gt"]
        targets, inside = set(gt["targets"]), gt["in_basket"]
        detached = sum(gt["detached"])
        if result.get("final_detached", detached) != detached or result.get("final_in_basket", sum(inside)) != sum(inside):
            raise ValueError("result final counts disagree with raw gt")
        row.update(correct_in_basket=sum(bool(inside[i]) for i in targets),
                   wrong_in_basket=sum(bool(v) for i, v in enumerate(inside) if i not in targets),
                   first_target_in_basket=bool(inside[gt["targets"][0]]), n_targets=len(targets),
                   final_detached=detached, final_in_basket=sum(inside), never_detached=detached == 0,
                   n_pick=gt["n_pick"], basket_unsuccessful=any(inside) and row["outcome"] != "success",
                   count_exact=(sum(inside) == gt["n_pick"]) if row["task"] == "FruitCount" else None,
                   control_steps=len(actions), proposed_fallback_steps=sum("proposed_action" not in a for a in actions))
        row["wrist_over_limit"] = any(abs(a["action"][6]) > 6.28 for a in actions)
        for key, field in (("action", "executed"), ("proposed_action", "proposed")):
            onset = next((a["control_step"] for a in actions
                          if abs(a.get(key, a["action"])[6] - q7_center) > 3), None)
            row[f"wrist_deviation_{field}"] = onset is not None
            row[f"wrist_deviation_onset_{field}"] = onset
            if field == "executed" and onset is not None:
                row["deviation_onset_context"] = onset_context(actions, frames, onset)
        row["start_idle"], row["longest_freeze_after_60"] = motion_metrics(actions)
        row["start_idle_right_censored"] = row["start_idle"] is None
        row["freeze_200"] = row["longest_freeze_after_60"] > 200
        # A changed arm target counts as a joint clip. Gripper is deliberately excluded.
        clips = [sum(abs(a["action"][j] - a.get("proposed_action", a["action"])[j]) > 1e-12
                     for a in actions) for j in range(7)]
        any_clip = sum(any(abs(a["action"][j] - a.get("proposed_action", a["action"])[j]) > 1e-12
                           for j in range(7)) for a in actions)
        row["guard_clip_rate"] = {"any_joint": ratio(any_clip, len(actions)),
                                  "per_joint": [ratio(x, len(actions)) for x in clips]}
        distances = [x for event in jsonl(directory / "event_signal.jsonl")
                     for x in numbers(event.get("distance"))]
        if not distances:
            raise ValueError("event signal has no finite distances")
        row["min_target_distance"] = min(distances)
        row["near_miss"] = row["never_detached"] and 0.03 < min(distances) <= 0.06
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        row["errors"].append(f"{type(exc).__name__}: {exc}")
    row["engineering_error"] = bool(row["errors"])
    return row


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "fraction": numerator / denominator if denominator else None}


def aggregate(rows):
    out = {"episodes": len(rows)}
    for outcome in ("success", "fail", "timeout"):
        out[outcome] = ratio(sum(r["outcome"] == outcome for r in rows), len(rows))
    for name in ("engineering_error", "first_target_in_basket", "never_detached", "near_miss", "wrist_over_limit",
                 "wrist_deviation_executed", "wrist_deviation_proposed", "freeze_200", "count_exact"):
        known = [r[name] for r in rows if r[name] is not None]
        out[name] = ratio(sum(known), len(known))
    for name in ("correct_in_basket", "wrong_in_basket"):
        known = [r[name] for r in rows if r[name] is not None]
        out[name] = {"total_fruits": sum(known), "episodes_with_data": len(known)}
    subset = [r for r in rows if r["basket_unsuccessful"] and r["freeze_200"] is not None]
    out["freeze_200_basket_unsuccessful"] = ratio(sum(r["freeze_200"] for r in subset), len(subset))
    vals = [r["start_idle"] for r in rows if r["start_idle"] is not None]
    out["start_idle"] = {"median": statistics.median(vals) if vals else None, "observed": len(vals),
                         "right_censored": sum(r.get("start_idle_right_censored", False) for r in rows)}
    tcp_rows = [r for r in rows if r.get("never_left_home") is not None]
    left = [r["leave_home_step"] for r in tcp_rows if r["leave_home_step"] is not None]
    out["leave_home_step"] = {"median": statistics.median(left) if left else None,
                              "observed": len(left), "episodes_with_data": len(tcp_rows),
                              "never_left_home": ratio(sum(r["never_left_home"] for r in tcp_rows), len(tcp_rows)),
                              "median_convention": "observed crossings only; never-left episodes counted separately"}
    event_rows = [r for r in rows if r.get("post_release_resume_steps") is not None]
    events = [event for r in event_rows for event in r["post_release_resume_steps"]]
    resumed = [event["resume_steps"] for event in events if event["resume_steps"] is not None]
    out["post_release_resume_steps"] = {
        "median": statistics.median(resumed) if resumed else None, "events": len(events),
        "observed_resumptions": len(resumed),
        "unresumed_at_end": ratio(sum(event["right_censored"] for event in events), len(events)),
        "episodes_with_entries": ratio(sum(bool(r["post_release_resume_steps"]) for r in event_rows), len(event_rows)),
        "median_convention": "observed resumptions only; right-censored events counted separately"}
    context = Counter(r["deviation_onset_context"]["category"] for r in rows if r["deviation_onset_context"])
    out["deviation_onset_context"] = {c: ratio(context[c], sum(context.values())) for c in CONTEXTS}
    guards = [r["guard_clip_rate"] for r in rows if r["guard_clip_rate"] is not None]
    out["guard_clip_rate"] = {"any_joint": ratio(sum(g["any_joint"]["numerator"] for g in guards),
                                                sum(g["any_joint"]["denominator"] for g in guards)),
                              "per_joint": [ratio(sum(g["per_joint"][j]["numerator"] for g in guards),
                                                  sum(g["per_joint"][j]["denominator"] for g in guards))
                                            for j in range(7)]}
    return out


def parse_range(text):
    values = set()
    for part in text.split(","):
        a, _, b = part.partition("-")
        values.update(range(int(a), int(b or a)+1))
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes-root", type=Path, required=True)
    p.add_argument("--pattern", choices=("ep{e}", "ep{e}_seed{s}"), required=True)
    p.add_argument("--episodes", help="inclusive, e.g. 200-219")
    p.add_argument("--seeds", help="inclusive, e.g. 7-9")
    p.add_argument("--q7-center", type=float, default=-1.56)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root, out = a.episodes_root.resolve(), a.out.resolve()
    # Protect every input tree, including sealed archived result directories.
    if root == out or root in out.parents:
        p.error("output must be outside the input episodes tree")
    if not root.is_dir() or not math.isfinite(a.q7_center):
        p.error("invalid input root or q7 center")
    if a.seeds and a.pattern == "ep{e}":
        p.error("seed filter requires ep{e}_seed{s}")
    episodes = parse_range(a.episodes) if a.episodes else None
    seeds = parse_range(a.seeds) if a.seeds else None
    regex = re.compile(r"ep(?P<e>\d+)" + (r"_seed(?P<s>\d+)" if "{s}" in a.pattern else ""))
    rows, identities = [], set()
    for task in TASKS:
        for directory in sorted((root / task).glob("ep*")):
            match = regex.fullmatch(directory.name)
            if not directory.is_dir() or not match:
                continue
            ep, seed = int(match["e"]), int(match["s"]) if "{s}" in a.pattern else None
            if (episodes is not None and ep not in episodes) or (seeds is not None and seed not in seeds):
                continue
            identities.add((task, ep, seed))
            try:
                row = analyze_episode(directory, a.q7_center)
            except (OSError, ValueError) as exc:
                p.error(f"cannot analyze {directory}: {exc}")
            row.update(task=task, episode_id=ep, model_seed=seed)
            rows.append(row)
    if not rows:
        p.error("no matching episodes")
    missing = []
    if episodes is not None and (a.pattern == "ep{e}" or seeds is not None):
        expected = {(task, ep, seed) for task in TASKS for ep in episodes
                    for seed in (seeds if seeds is not None else [None])}
        missing = sorted(expected - identities)
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    with (out / "episodes.csv").open("w") as f:
        keys = sorted(set().union(*(r.keys() for r in rows)))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    summary = {"schema": "fruit-failure-modes-v2", "generated_utc": timestamp(),
               "input": {"root": str(root), "pattern": a.pattern, "episodes": a.episodes, "seeds": a.seeds},
               "definitions": {"q7_center": a.q7_center, "wrist_limit": 6.28, "wrist_deviation": 3,
                               "motion_threshold": 0.005, "freeze_after_step": 60, "freeze_strictly_over": 200,
                               "start_idle_index": "one-based control_step of first moving action; no motion = null",
                               "context": "post-action snapshot at max(0,onset_control_step-32); gripper from command",
                               "leave_home_step": "first control step with TCP >0.05m from frame0; null and never-left count if absent",
                               "post_release_resume_steps": "per basket-entry transition, first subsequent TCP >0.05m from entry position; simultaneous fruit entries one event, re-entries included, unresumed entries right-censored",
                               "legacy_proposed_action": "falls back to executed action",
                               "physical_observe_prefix": "explicit observe_hold_steps=30 keeps physical frames0..30 and hold actions in the1300-step clock; history-only flags do not remove physical time; legacy pre-execution video frames remain excluded",
                               "open_before_first_detach": "fourth residual class; includes open after detached loss",
                               "missing_metric": "null; excluded only from its own metric denominator"},
               "script_sha256": sha256(Path(__file__)), "missing_expected_episodes": missing,
               "all": aggregate(rows), "by_task": {t: aggregate([r for r in rows if r["task"] == t]) for t in TASKS},
               "failed_runs": [{"directory": r["directory"], "errors": r["errors"]} for r in rows if r["engineering_error"]]}
    atomic_json(out / "summary.json", summary)
    print(json.dumps({"out": str(out), "episodes": len(rows), "errors": len(summary["failed_runs"]),
                      "missing": len(missing)}, ensure_ascii=False), flush=True)
    return int(bool(summary["failed_runs"] or missing))


if __name__ == "__main__":
    raise SystemExit(main())
