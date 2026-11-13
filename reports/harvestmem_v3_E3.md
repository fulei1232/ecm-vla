# HarvestMem v3 E3 Report

Date: 2026-11-12

## SG Sensitivity Probe v2

The probe ran before the closed-loop interpretation and used fixed 16-step
boundaries from dev expert episodes 200-203 for both tasks. Every variant
reset the SG service, loaded the identical history buffer, and then inferred
with only the prompt changed. There were 70 probe points: A/new-pick 31,
B/in-progress-pick 35, and C/Count-return 4.

The action-change metric is mean absolute 7-joint action change divided by the
same block's expert-vs-idle error. Type swaps also report the last-action joint
distance to the rest pose.

| Point type | Other slot | Same +4 px | Same -4 px | Type swap | No subgoal |
|---|---:|---:|---:|---:|---:|
| A: new pick | 0.0285 | 0.0049 | 0.0041 | 0.0331 | 0.0904 |
| B: pick in progress | 0.0620 | 0.0286 | 0.0185 | 0.1237 | 0.2442 |
| C: Count return | 0.0000 | 0.0000 | 0.0000 | 70.1638 | 116.1579 |

For type swaps, A ended farther from rest in 27/31 points and closer in 4/31;
B ended farther in 20/35 and closer in 15/35. The mean rest distances for
A were 1.266 original vs 1.286 swapped; for B, 1.722 original vs 1.705
swapped. C was 0.01140 original vs 0.01145 swapped. This probe is diagnostic
only and has no pass/fail gate.

Probe output: `results/harvestmem_v3_E3/sensitivity_v2.json`.

## Closed-loop evaluation

Ran A = SG + oracle, D = K0, and E = NM with `memory=none` on dev episodes
200-239 for Count and Occluded ripe: 240 episodes total, 80 per arm. The
queue used 3/3/2 GPU lanes, at most one client per lane, 30-minute episode
timeouts, resumable outputs, and rebuilt `SUMMARY.md` after each completed
episode. All final 240 results are non-engineering results; six initial
layout self-check errors were corrected and rerun.

| Arm | Task | Episodes | Success | Rate | Mean steps | RETURN follow | Target follow |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Count | 40 | 0 | 0.0% | 1110.3 | 12.5% | 0.0% |
| A | Occluded ripe | 40 | 2 | 5.0% | 453.8 | 0.0% | 0.0% |
| D | Count | 40 | 0 | 0.0% | 917.6 | 22.5% | n/a |
| D | Occluded ripe | 40 | 0 | 0.0% | 556.8 | 0.0% | n/a |
| E | Count | 40 | 0 | 0.0% | 995.4 | 25.0% | n/a |
| E | Occluded ripe | 40 | 3 | 7.5% | 432.2 | 0.0% | n/a |

Count strata were identical across arms: A, D, and E each achieved 0/9 for
N=2-4, 0/16 for N=5-7, and 0/14 for N=8-10. The denominator is 39 because
one Count episode had an initial engineering self-check error and was rerun;
the final episode set is still 40 per arm, with the summary's N strata only
including episodes carrying a valid N field.

## Failure categories

Failures are multi-label, so counts can exceed 40 per row. The dominant
failure classes were low-level execution issues: dropped fruit, TCP stuck for
300 frames, never pulling a fruit, robot drops, and Count completion/overpick
errors. Occluded runs also frequently picked initially green fruit.

The complete per-arm/per-task table is in
`results/harvestmem_v3_E3/SUMMARY.md`. Representative counts:

- A Count: dropped outside basket 24, TCP stuck 24, never pulled 14,
  overpick 11.
- A Occluded: initially green picked 26, dropped outside basket 16.
- D Count: never pulled 22, TCP stuck 19, dropped outside basket 16.
- D Occluded: initially green picked 28, dropped outside basket 11.
- E Count: dropped outside basket 30, TCP stuck 23, overpick 14.
- E Occluded: initially green picked 28, dropped outside basket 7.

## Oracle events

A emitted `target_selected` 158 times on Count and 62 times on Occluded;
`wrong_fruit_detached` occurred 11 and 43 times respectively. Count also
used `pick_place_displacement_fallback` 47 times and
`pick_place_timeout_fallback` 37 times. Occluded used those fallbacks 9 and
1 times. No `oracle_stall` or `reopen_after_no_target` events were observed.
The full event table is in `SUMMARY.md`.

## Typical failure videos

| Arm/task | Three representative videos |
|---|---|
| A Count | `results/harvestmem_v3_E3/A/count/ep200/front_camera.mp4`; `results/harvestmem_v3_E3/A/count/ep201/front_camera.mp4`; `results/harvestmem_v3_E3/A/count/ep202/front_camera.mp4` |
| A Occluded | `results/harvestmem_v3_E3/A/occluded_ripe/ep200/front_camera.mp4`; `results/harvestmem_v3_E3/A/occluded_ripe/ep201/front_camera.mp4`; `results/harvestmem_v3_E3/A/occluded_ripe/ep202/front_camera.mp4` |
| D Count | `results/harvestmem_v3_E3/D/count/ep200/front_camera.mp4`; `results/harvestmem_v3_E3/D/count/ep201/front_camera.mp4`; `results/harvestmem_v3_E3/D/count/ep202/front_camera.mp4` |
| D Occluded | `results/harvestmem_v3_E3/D/occluded_ripe/ep200/front_camera.mp4`; `results/harvestmem_v3_E3/D/occluded_ripe/ep201/front_camera.mp4`; `results/harvestmem_v3_E3/D/occluded_ripe/ep202/front_camera.mp4` |
| E Count | `results/harvestmem_v3_E3/E/count/ep200/front_camera.mp4`; `results/harvestmem_v3_E3/E/count/ep201/front_camera.mp4`; `results/harvestmem_v3_E3/E/count/ep202/front_camera.mp4` |
| E Occluded | `results/harvestmem_v3_E3/E/occluded_ripe/ep200/front_camera.mp4`; `results/harvestmem_v3_E3/E/occluded_ripe/ep202/front_camera.mp4`; `results/harvestmem_v3_E3/E/occluded_ripe/ep203/front_camera.mp4` |

## Fixed decision

A is below 40% on both tasks: Count 0/40 and Occluded ripe 2/40. Under the
pre-registered rule, this is a low-level execution failure. The observed
target-follow rate is 0%, and RETURN follow is only 12.5% on Count and 0% on
Occluded. The appropriate conclusion is to report the operation failures and
stop; no training or tracker changes are made here.

E3 is complete. E4/C-arm work is not started.
