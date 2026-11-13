# HarvestMem v3 E4a Report

Date: 2026-11-12

## Conclusion

E4a completed on Count dev 200-259 for six arms, 360 episodes total. The
forced-stop intervention passed its engineering gate: every forced-stop
trigger had exactly N fruits, ended with `finish_signal` within 300 steps, and
had zero overpick. DR reached 26.7%, exceeding the preregistered 20% target
and outperforming D's 0%. The structural stopping intervention is effective.

## E4-0 correction

E3 was reprocessed with `scripts/harvestmem_v3/decision_metrics.py`. The
resulting correction is [harvestmem_v3_E3_erratum.md](harvestmem_v3_E3_erratum.md).
It identifies the missed start failures and separates decision failures from
execution failures without changing the original E3 report or results.

## Smoke tests

- Debug K0 `--debug-stop-at-step 200`: triggered at step 200, arrived at rest
  at step 430, and ended at step 472 with `finish_signal`; N was intentionally
  not reached, so this was the expected debug failure.
- Real forced-stop K0 ep220: success; trigger N=5, trigger step 823, end step
  893, no overpick, no primitive timeout.
- Real forced-stop K0 ep235: natural start failure before trigger.
- Real forced-stop K0 ep206: execution `robot_drop` before trigger.
- Stop switch disabled K0 ep200: all actions were policy actions and no
  stop-primitive fields appeared.

Smoke outputs: `results/harvestmem_v3_E4a/smoke/`.

## Formal results

| Arm | Checkpoint/input | Episodes | Success | Rate | Wilson 95% interval | Stop triggers | Overpick |
|---|---|---:|---:|---:|---:|---:|---:|
| A | SG + oracle, no stop | 60 | 0 | 0.0% | [0.0%, 6.0%] | 0 | 16 |
| AR | SG + oracle + stop | 60 | 16 | 26.7% | [17.1%, 39.0%] | 16 | 0 |
| D | K0, no stop | 60 | 0 | 0.0% | [0.0%, 6.0%] | 0 | 15 |
| DR | K0 + stop | 60 | 16 | 26.7% | [17.1%, 39.0%] | 16 | 0 |
| E | NM, no stop | 60 | 0 | 0.0% | [0.0%, 6.0%] | 0 | 21 |
| ER | NM + stop | 60 | 23 | 38.3% | [27.1%, 51.0%] | 23 | 0 |

Forced-stop engineering gate:

- AR/DR/ER overpick: `0`.
- Triggered episodes with `stop_trigger_n != N`: `0`.
- Triggered episodes not ending in `finish_signal`: `0`.
- Triggered episodes taking over 300 steps to end: `0`.

## Decision and execution indicators

The E3 erratum's CPU metrics identified Count start failures as decision
failures. In E4a, no-stop arms still show the same sparse stop/start issue:

- A: 12 start failures; AR: 12.
- D: 33 start failures; DR: 31.
- E: 8 start failures; ER: 8.

Forced stop removes the post-N overpick/budget path whenever it triggers. The
remaining dominant errors are robot drops, budget exhaustion, and start
failures before N is reached. Full per-episode data is in
`results/harvestmem_v3_E4a/rows.json`; the live summary is
`results/harvestmem_v3_E4a/SUMMARY.md`.

## Paired D vs DR table

Episodes are paired by episode number:

| Outcome | Count |
|---|---:|
| Both successful | 0 |
| DR only successful | 16 |
| D only successful | 0 |
| Both failed | 44 |

## Representative videos

DR successes:

- `results/harvestmem_v3_E4a/DR/count/ep201/front_camera.mp4`
- `results/harvestmem_v3_E4a/DR/count/ep206/front_camera.mp4`
- `results/harvestmem_v3_E4a/DR/count/ep210/front_camera.mp4`

DR execution failures:

- `results/harvestmem_v3_E4a/DR/count/ep205/front_camera.mp4`
- `results/harvestmem_v3_E4a/DR/count/ep215/front_camera.mp4`
- `results/harvestmem_v3_E4a/DR/count/ep219/front_camera.mp4`

DR start failures:

- `results/harvestmem_v3_E4a/DR/count/ep202/front_camera.mp4`
- `results/harvestmem_v3_E4a/DR/count/ep203/front_camera.mp4`
- `results/harvestmem_v3_E4a/DR/count/ep204/front_camera.mp4`

## E4b status

E4b was not implemented or run. Its design is documented in
`reports/harvestmem_v3_E4b_design.md`.

E4a is complete and stops here for review.
