# HarvestMem v3 E1b Oracle Robustness Report

Date: 2026-11-12

## Conclusion

E1b passed. The oracle now has bounded closed-loop recovery paths, preserves
target locking, and emits all events needed by E2/E3. The CPU replay gate
still passes: both tasks exceed 99% frame agreement, target errors are zero,
and all 100 collapsed subgoal sequences exactly match GT.

## Changes

- Replaced the old negative-count placeholder with an exact comparison of the
  predicted and GT subgoal sequences after consecutive duplicates are
  collapsed.
- Kept the primary PICK to PLACE condition: target displacement at least
  0.105 m and TCP pull velocity at most 0.001 m/frame.
- Added PICK fallbacks after the fruit has crossed 0.105 m: switch within 10
  frames, or immediately at 0.15 m displacement.
- Locked the selected target until robot detachment, wind detachment, or
  disappearance. A newly appearing higher fruit cannot replace it.
- Changed basket confirmation to the target fruit's own `in_basket=True`.
  A fruit already in the basket when PLACE begins is confirmed immediately.
- Kept the three-frame open-gripper transition and added a 20-frame
  post-confirmation fallback.
- Added wrong-fruit takeover, dropped-fruit recovery, and a 300-frame PLACE
  stall guard.
- `prev_tcp_pos` resets on each new target and is updated every frame.
- `OracleOutput.events` exposes `target_selected`, `robot_detached`,
  `basket_confirmed`, `drop`, `wrong_fruit_detached`, `oracle_stall`, and all
  fallback events for `subgoal_timeline.jsonl`.

The environment naming trap remains explicit: `attached=True` means the fruit
has detached from the branch.

## Synthetic tests

Seven CPU tests pass:

1. Continuous movement after detachment switches to PLACE within 10 frames.
2. Entering PLACE with the fruit already in the basket advances normally.
3. A gripper stalled at 0.035 advances through the 20-frame fallback.
4. A newly replenished higher fruit does not change the locked target.
5. A stationary dropped fruit emits `drop` and reselects.
6. Detaching a non-target fruit emits `wrong_fruit_detached` and tracks it.
7. PLACE beyond 300 frames emits `oracle_stall` and reselects.

Command:

```bash
python3 scripts/harvestmem_v3/test_oracle.py
```

Result: `Ran 7 tests ... OK`.

## Offline gate

Command:

```bash
python3 scripts/harvestmem_v3/oracle.py \
  --dev-root results/harvestmem_v3_S1/replays/dev_expert \
  --out results/harvestmem_v3_E1b_oracle_validation.json
```

| Task | Episodes | Frames | Accuracy | Target errors | Folded sequence matches | Max timing offset |
|---|---:|---:|---:|---:|---:|---:|
| Count | 50 | 49,242 | 99.3319% | 0 | 50/50 | 1 frame |
| Occluded ripe | 50 | 18,679 | 99.3522% | 0 | 50/50 | 1 frame |

Overall collapsed-sequence result: **100/100 exact matches**. The validator's
`count_errors` is now the number of episodes whose collapsed full sequence
differs from GT; it is zero for both tasks.

## Expert-replay event counts

| Event | Count | Occluded ripe |
|---|---:|---:|
| `target_selected` | 312 | 129 |
| `robot_detached` | 312 | 129 |
| `basket_confirmed` | 312 | 129 |
| `pick_place_timeout_fallback` | 0 | 0 |
| `pick_place_displacement_fallback` | 0 | 0 |
| `place_timeout_fallback` | 0 | 0 |
| `drop` | 0 | 0 |
| `wrong_fruit_detached` | 0 | 0 |
| `oracle_stall` | 0 | 0 |

The fallback and recovery conditions did not trigger on successful expert
replays, so they improve bounded closed-loop behavior without changing the
offline expert path.

## NM status

NM formal training completed normally at
`2026-11-11T23:10:35.204548+00:00`; `formal_finished.json` reports
`passed: true`. The final loss is 0.0024 and the checkpoint is:

`runs/harvestmem_v3_S2_nm_20261111/nm/ckpt/mme_vla_suite/nm/11999`

E2 has not been started. E1b stops here for review.
