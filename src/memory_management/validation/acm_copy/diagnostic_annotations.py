"""Causal simulator annotations with explicit visual-evidence requirements.

The simulator state identifies what happened. Segmentation-derived visible-pixel
counts decide whether that event is usable image evidence. All frame indexes are
from the original trajectory and never refer to a future intervention frame.
"""
import json
import math
import re
from pathlib import Path

VERSION = "candidate-visual-support-v2"
MIN_VISIBLE_PIXELS = 8
MIN_TARGET_PIXELS = 32
MIN_CONTAINER_PIXELS = 128
MIN_VISIBILITY_RATIO = 0.75
MAX_ABSENT_PIXELS = 7
EVENT_RECENCY_FRAMES = 64
BIN_EVENT_CLIP = 3
VIDEO_EVENT_CLIP = 2
BIN_CONTAINER = "board_with_hole"


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def integer(value, default=0):
    while isinstance(value, list) and value:
        value = value[0]
    return int(value) if value is not None else default


def group(rows, first, last):
    return [r["frame_id"] for r in rows if first <= r["frame_id"] <= last]


def contiguous(ids):
    result = []
    for i in sorted(set(ids)):
        if not result or i != result[-1][-1] + 1:
            result.append([i])
        else:
            result[-1].append(i)
    return result


def _position(item):
    value = item.get("position")
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, list) or len(value) < 2:
        return None
    return tuple(float(x) for x in value[:3])


def _actors(row):
    return {item.get("name"): _position(item) for item in row.get("gt", {}).get("all_cubes", [])
            if item.get("name")}


def _in_deletion_region(position):
    return position is not None and abs(position[0]) >= 5.0 and abs(position[1]) >= 5.0


def resolve_binfill_actor(rows, transition_frame, color):
    """Return the exact cube actor newly moved to the task's deletion region."""
    by_id = {row["frame_id"]: row for row in rows}
    if transition_frame not in by_id or transition_frame - 1 not in by_id:
        return None
    before, after = _actors(by_id[transition_frame - 1]), _actors(by_id[transition_frame])
    pattern = re.compile(rf"^cube_{re.escape(color)}_[0-9]+$", re.IGNORECASE)
    moved = [name for name, position in after.items()
             if pattern.match(name) and _in_deletion_region(position)
             and name in before and not _in_deletion_region(before[name])]
    if len(moved) == 1:
        return moved[0]
    if not moved:
        return None
    # Multiple simultaneous deletions are not expected. Pick the largest verified jump
    # deterministically, while leaving the ambiguity visible in annotation metadata.
    return max(moved, key=lambda name: (math.dist(before[name], after[name]), name))


def visible_pixels(row, actor):
    return integer(row.get("gt", {}).get("front_visible_pixels", {}).get(actor, 0))


def _peak(rows, actor):
    return max((visible_pixels(row, actor) for row in rows), default=0)


def _visible(row, actor, peak, minimum):
    pixels = visible_pixels(row, actor)
    required = max(minimum, math.ceil(peak * MIN_VISIBILITY_RATIO))
    return pixels >= required


def _eligible_joint(rows, target, container, target_peak=None, container_peak=None):
    target_peak = _peak(rows, target) if target_peak is None else target_peak
    container_peak = _peak(rows, container) if container_peak is None else container_peak
    return [row["frame_id"] for row in rows
            if _visible(row, target, target_peak, MIN_TARGET_PIXELS)
            and _visible(row, container, container_peak, MIN_CONTAINER_PIXELS)]


def _eligible_after(rows, target, container, container_peak):
    return [row["frame_id"] for row in rows
            if visible_pixels(row, target) <= MAX_ABSENT_PIXELS
            and _visible(row, container, container_peak, MIN_CONTAINER_PIXELS)]


def _robot_pixels(row):
    pixels = row.get("gt", {}).get("front_visible_pixels", {})
    return sum(integer(value) for name, value in pixels.items() if name.startswith("panda_"))


def _visibility_audit(rows, target, container, target_peak, container_peak, role):
    audit = []
    for row in rows:
        target_pixels = visible_pixels(row, target)
        container_pixels = visible_pixels(row, container)
        reasons = []
        if role == "before" and not _visible(row, target, target_peak, MIN_TARGET_PIXELS):
            reasons.append("target_occluded_or_too_small")
        if role == "after" and target_pixels > MAX_ABSENT_PIXELS:
            reasons.append("target_disappearance_not_established")
        if not _visible(row, container, container_peak, MIN_CONTAINER_PIXELS):
            reasons.append("container_occluded_or_too_small")
        audit.append({
            "frame_id": row["frame_id"], "role": role,
            "target_pixels": target_pixels, "container_pixels": container_pixels,
            "robot_pixels": _robot_pixels(row), "eligible": not reasons,
            "rejection_reasons": reasons,
        })
    return audit


def required_support(annotation, group_index):
    support = annotation.get("required_group_support", [])
    return int(support[group_index]) if group_index < len(support) else 1


def consecutive_windows(group_ids, length):
    return [run[i:i + length] for run in contiguous(group_ids)
            for i in range(max(0, len(run) - length + 1))]


def annotation_sufficient(annotation):
    return bool(annotation.get("required_groups")) and all(
        consecutive_windows(group_ids, required_support(annotation, group_index))
        for group_index, group_ids in enumerate(annotation["required_groups"])
    )


def annotate(task, rows, available_max=None):
    if available_max is not None:
        rows = [r for r in rows if r["frame_id"] <= available_max]
    if not rows:
        return []
    last, max_id = rows[-1]["gt"], rows[-1]["frame_id"]
    annotations = []

    def add(ident, kind, groups, seen, needed, reason, objects=(), active=True,
            quality="unreviewed", supports=None, roles=None, group_objects=None, checks=None,
            metadata=None):
        supports = list(supports or [1] * len(groups))
        annotations.append({
            "annotation_id": ident, "task": task, "kind": kind,
            "required_groups": groups, "required_group_support": supports,
            "group_roles": list(roles or ["evidence"] * len(groups)),
            "group_objects": list(group_objects or [list(objects)] * len(groups)),
            "t_seen": seen, "t_need_lower_bound": needed, "active": active,
            "reason": reason, "objects": list(objects), "visual_checks": list(checks or []),
            "metadata": dict(metadata or {}), "source": "sim_GT+front_segmentation",
            "quality": quality, "schema": VERSION,
        })

    if task == "BinFill":
        previous = {c: 0 for c in ("red", "blue", "green")}
        for row_index, row in enumerate(rows):
            frame_id, gt = row["frame_id"], row["gt"]
            for color in previous:
                count = integer(gt.get(color + "_cubes_in_bin", 0))
                if count > previous[color]:
                    actor = resolve_binfill_actor(rows, frame_id, color)
                    before_rows = rows[max(0, row_index - 8):row_index]
                    after_rows = rows[row_index:min(len(rows), row_index + 6)]
                    before_peak = _peak(before_rows, actor) if actor else 0
                    container_peak = _peak(before_rows + after_rows, BIN_CONTAINER)
                    before = (_eligible_joint(before_rows, actor, BIN_CONTAINER,
                                               before_peak, container_peak) if actor else [])
                    after = (_eligible_after(after_rows, actor, BIN_CONTAINER, container_peak)
                             if actor else [])
                    age = max_id - frame_id
                    checks = [
                        {"role": "before", "target": actor, "container": BIN_CONTAINER,
                         "target_peak_pixels": before_peak, "container_peak_pixels": container_peak,
                         "min_visibility_ratio": MIN_VISIBILITY_RATIO,
                         "min_target_pixels": MIN_TARGET_PIXELS,
                         "min_container_pixels": MIN_CONTAINER_PIXELS},
                        {"role": "after", "target": actor, "container": BIN_CONTAINER,
                         "target_max_absent_pixels": MAX_ABSENT_PIXELS,
                         "min_visibility_ratio": MIN_VISIBILITY_RATIO,
                         "min_container_pixels": MIN_CONTAINER_PIXELS},
                    ]
                    add(
                        f"drop_{color}_{count}",
                        "event" if age <= EVENT_RECENCY_FRAMES else "retention",
                        [before, after], frame_id, frame_id + BIN_EVENT_CLIP - 1,
                        "exact dropped actor and bin must be jointly visible before the count "
                        "transition; bin visibility and actor disappearance must persist after it",
                        [x for x in (actor, BIN_CONTAINER) if x],
                        quality="segmentation_visible_consecutive_clip",
                        supports=[BIN_EVENT_CLIP, BIN_EVENT_CLIP], roles=["before", "after"],
                        group_objects=[[x for x in (actor, BIN_CONTAINER) if x], [BIN_CONTAINER]],
                        checks=checks,
                        metadata={"color": color, "count": count, "target_actor": actor,
                                  "container_actor": BIN_CONTAINER, "transition_frame": frame_id,
                                  "actor_resolved": actor is not None,
                                  "frame_visibility_audit": (
                                      _visibility_audit(before_rows, actor, BIN_CONTAINER,
                                                        before_peak, container_peak, "before")
                                      + _visibility_audit(after_rows, actor, BIN_CONTAINER,
                                                          before_peak, container_peak, "after")
                                      if actor else [])},
                    )
                previous[color] = count
    elif task == "PickHighlight":
        names = last.get("target_cube_names", [])
        counts = last.get("target_cube_pickup_counts", {})
        for name in names:
            lit = [r["frame_id"] for r in rows if name in r["gt"].get("highlighted_targets", [])
                   and visible_pixels(r, name) >= MIN_VISIBLE_PIXELS]
            if not lit:
                add(f"highlight_{name}", "retention", [[]], 10, 10,
                    "no visibly highlighted target frame captured yet", [name],
                    active=counts.get(name, 0) < 1)
                continue
            ended = name not in last.get("highlighted_targets", [])
            add(f"highlight_{name}", "retention" if ended else "event", [lit], lit[0], lit[0],
                "visible object during original highlight window; needed until its first successful pickup",
                [name], active=counts.get(name, 0) < 1,
                quality="segmentation_visible_visual_unreviewed")
    elif task == "VideoUnmaskSwap":
        demo = [r for r in rows if r["is_demo"]]
        bins = last.get("selected_bins", [])
        colors = last.get("color_names", [])
        for i, (bin_actor, color) in enumerate(zip(bins, colors)):
            bin_name = bin_actor["name"]
            names = sorted({name for r in demo for name in r["gt"].get("front_visible_pixels", {})
                            if "cube" in name.lower() and color.lower() in name.lower()})
            target = max(names, key=lambda name: _peak(demo, name), default=None)
            target_visible = ([r["frame_id"] for r in demo
                               if visible_pixels(r, target) >= MIN_VISIBLE_PIXELS] if target else [])
            bin_visible = [r["frame_id"] for r in demo
                           if visible_pixels(r, bin_name) >= MIN_CONTAINER_PIXELS]
            if target_visible and bin_visible:
                last_target, first_bin = max(target_visible), min(bin_visible)
                before = [idx for idx in target_visible if last_target - 5 <= idx <= last_target]
                after = [idx for idx in bin_visible if first_bin <= idx <= first_bin + 5]
            else:
                before, after = [], []
            add(
                f"reveal_{i}_{color}", "retention", [before, after],
                before[0] if before else None, rows[0]["exec_start_idx"],
                "short consecutive clips on both sides of the unmask boundary establish "
                "the colored target-to-container correspondence",
                [x for x in (target, bin_name) if x],
                quality="segmentation_visible_consecutive_correspondence",
                supports=[VIDEO_EVENT_CLIP, VIDEO_EVENT_CLIP], roles=["target_before", "container_after"],
                group_objects=[[target] if target else [], [bin_name]],
                checks=[
                    {"role": "target_before", "target": target,
                     "min_target_pixels": MIN_VISIBLE_PIXELS},
                    {"role": "container_after", "container": bin_name,
                     "min_container_pixels": MIN_CONTAINER_PIXELS},
                ],
                metadata={"color": color, "target_actor": target, "container_actor": bin_name},
            )
        schedule = last.get("swap_schedule", [])
        for k, (a, b, start, end) in enumerate(schedule):
            phases = []
            for lo, hi in ((start - 3, start + 3),
                           ((start + end) // 2 - 3, (start + end) // 2 + 3),
                           (end - 3, end + 3)):
                candidates = [r for r in demo if lo <= integer(r["gt"].get("elapsed_steps")) - 1 <= hi]
                a_peak, b_peak = _peak(candidates, a), _peak(candidates, b)
                phases.append([r["frame_id"] for r in candidates
                               if _visible(r, a, a_peak, MIN_CONTAINER_PIXELS)
                               and _visible(r, b, b_peak, MIN_CONTAINER_PIXELS)])
            add(f"swap_{k}", "event", phases, phases[0][0] if phases and phases[0] else None,
                rows[0]["exec_start_idx"],
                "swap start/mid/end clips with both containers visibly distinguishable",
                [a, b], quality="segmentation_visible_consecutive_trajectory",
                supports=[VIDEO_EVENT_CLIP] * len(phases),
                roles=["swap_start", "swap_middle", "swap_end"],
                group_objects=[[a, b]] * len(phases))
    elif task == "PatternLock":
        demo = [r for r in rows if r["is_demo"]]
        runs = []
        for row in demo:
            phase = row["gt"].get("current_task_index")
            if not runs or runs[-1][0] != phase:
                runs.append([phase, []])
            runs[-1][1].append(row["frame_id"])
        for phase, ids in runs:
            label = rows[ids[0]]["gt"].get("task_phase", "")
            if not str(label).startswith("move "):
                continue
            n = len(ids)
            groups = [ids[:max(1, n // 4)], ids[n // 3:max(n // 3 + 1, 2 * n // 3)],
                      ids[max(0, 3 * n // 4):]]
            add(f"path_phase_{phase}", "retention", groups, ids[0], rows[0]["exec_start_idx"],
                f"demonstrated {label}; original chronological segment, three phase groups",
                quality="trajectory_GT_visual_unreviewed")
    return annotations


def active_annotations(task, rows, max_id, kinds):
    return [a for a in annotate(task, rows, max_id)
            if a["active"] and a["kind"] in kinds]


def active_groups(task, rows, max_id, kind):
    return [(annotation, group_ids)
            for annotation in active_annotations(task, rows, max_id, kind)
            if annotation_sufficient(annotation)
            for group_ids in annotation["required_groups"]]


def covered(annotation, selected, max_id):
    groups = [[i for i in group_ids if i <= max_id]
              for group_ids in annotation["required_groups"]]
    raw = bool(groups) and all(
        consecutive_windows(group_ids, required_support(annotation, group_index))
        for group_index, group_ids in enumerate(groups)
    )
    selected = set(selected)
    selected_coverage = all(
        any(set(window).issubset(selected)
            for window in consecutive_windows(group_ids, required_support(annotation, group_index)))
        for group_index, group_ids in enumerate(groups)
    )
    return raw, raw and selected_coverage
