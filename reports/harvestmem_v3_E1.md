# HarvestMem v3 E1 Oracle Report

Date: 2026-11-11

## Conclusion

E1 passed. The shared causal oracle reached at least 99% frame agreement on
both tasks, with zero target-selection errors and zero count errors. All
remaining mismatches were transition-timing differences within one frame,
which is inside the allowed +/-5-frame window.

## What ran

Implemented the shared CPU oracle in
`scripts/harvestmem_v3/oracle.py`. The same `HarvestMemOracle.step()` function
is intended for online snapshots and offline GT rows. It handles observe,
expert y-order target selection, Count/Occluded stopping, wind invalidation,
robot detachment, basket confirmation, return, and the permitted dropped-fruit
reselection path.

The offline validator used the successful attempt for each episode in:
`results/harvestmem_v3_S1/replays/dev_expert/{count,occluded_ripe}/`, episodes
200-249. No GPU was used. Unit tests are in
`scripts/harvestmem_v3/test_oracle.py`.

Commands:

```bash
python3 scripts/harvestmem_v3/test_oracle.py
python3 scripts/harvestmem_v3/oracle.py \
  --dev-root results/harvestmem_v3_S1/replays/dev_expert \
  --out results/harvestmem_v3_E1_oracle_validation.json
```

Source revision recorded for the surrounding work: commit
`414ef44b9afb7a84e6cec9573382700cba4b3151`; the worktree contains uncommitted
changes.

## Alignment results

| Task | Episodes | Frames | Frame accuracy | Target errors | Folded sequence errors | Transition mismatches |
|---|---:|---:|---:|---:|---:|---:|
| Count | 50 | 49,242 | 99.3319% | 0 | 0 | 329 |
| Occluded ripe | 50 | 18,679 | 99.3522% | 0 | 0 | 121 |

All 450 transition mismatches were within one frame of an expected label
transition. The mismatch types were carry/pick, settle/pick, pull/place, and
terminal-hold/place timing; there were no wrong fruit selections or collapsed
sequence errors.

### Accuracy by N

Count:

| N | Episodes | Frames | Accuracy |
|---:|---:|---:|---:|
| 2 | 3 | 1,063 | 99.6237% |
| 3 | 7 | 3,439 | 99.3894% |
| 4 | 4 | 2,746 | 99.4173% |
| 5 | 4 | 3,133 | 99.3616% |
| 6 | 10 | 9,406 | 99.2345% |
| 7 | 7 | 7,831 | 99.2849% |
| 8 | 3 | 3,502 | 99.5431% |
| 9 | 4 | 5,885 | 99.0824% |
| 10 | 8 | 12,237 | 99.4280% |

Occluded ripe:

| N | Episodes | Frames | Accuracy |
|---:|---:|---:|---:|
| 2 | 21 | 6,260 | 99.3450% |
| 3 | 29 | 12,419 | 99.3558% |

## State-machine adjustment

The first implementation switched PICK to PLACE as soon as the simulator's
`attached=True` flag appeared. That flag changes at grasp detachment, while
the training label changes when the carry motion begins. Following the task
list allowance, the oracle now requires the attached fruit to have moved at
least 0.105 m from its original slot and the TCP pull velocity to be at most
0.001 m per frame. PLACE uses the target fruit's own `in_basket=True` signal
and advances after three consecutive open-gripper frames. E1b added bounded
fallbacks and event reporting without changing the expert-replay sequence.
The transition offset remains at most one frame.

The environment naming trap is handled explicitly: `attached=True` means the
fruit has left the branch; hanging is `present and not attached`.

## Gate and next step

| Gate | Result |
|---|---|
| Count frame accuracy >= 99% | Pass: 99.3319% |
| Occluded frame accuracy >= 99% | Pass: 99.3522% |
| Wrong target selections | Pass: 0 |
| Folded sequence errors | Pass: 0; 100/100 exact matches |
| Unit tests | Pass: 7/7 |
| CPU-only E1 execution | Pass |

E1 and E1b are complete and ready for review. E2 has not been started. NM
formal training completed successfully at `2026-11-11T23:10:35.204548+00:00`;
its final checkpoint is
`runs/harvestmem_v3_S2_nm_20261111/nm/ckpt/mme_vla_suite/nm/11999`.
