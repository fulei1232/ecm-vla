# HarvestMem v3 E2 Report

Date: 2026-11-12

## Conclusion

E2 completed. The oracle recheck passed, the SG/K0/NM services loaded their
specified checkpoints and matching normalization statistics, all requested
200-step smoke runs completed without engineering errors, and the combined
train-replay open-loop ratio was below 0.3 for all three arms. E3 was not
started.

## Oracle recheck

The Count no-target RETURN fix was added: before `n_in_basket >= N`, RETURN is
non-absorbing and reopens when a hanging target appears, emitting
`reopen_after_no_target`. Occluded behavior is unchanged.

Command:

```bash
python3 scripts/harvestmem_v3/test_oracle.py
python3 scripts/harvestmem_v3/oracle.py \
  --dev-root results/harvestmem_v3_S1/replays/dev_expert \
  --out results/harvestmem_v3_E2_oracle_recheck.json
```

- Synthetic tests: 8/8 passed, including the reopen case.
- Count: 99.3319% frame accuracy, 0 target errors, 0 sequence errors.
- Occluded ripe: 99.3522% frame accuracy, 0 target errors, 0 sequence errors.
- Folded subgoal sequences: 100/100 exact matches.

## Services and self-checks

All three checkpoint `norm_stats.json` files compare byte-for-byte with
`data/harvestmem_v3_dataset_20261110/assets/robomme/norm_stats.json`.

Each smoke episode checked:

- `FruitPickTaskV2-v0`, `rm75b_eg2`, task revision 9;
- `no_lamp_wait=True`;
- seed `4200000 + 16 * episode`;
- Count prompt has no work-light prefix;
- live `N` and projected slot pixels match the dev expert record;
- train-clean Count ep0 instruction matches `data/harvestmem_v3_dataset_20261110/data/0.pkl` exactly;
- checkpoint path contains `harvestmem_v3_S2`.

The live slot pixels for Count ep200 were `{0: [164, 135], 1: [150, 98],
2: [119, 118], 3: [67, 114]}`, matching the dev expert record.

## Open-loop check

The checker used the same 30-frame prefix, 16-step action chunks, `add_buffer`
protocol, prompts, and memory mode as closed-loop evaluation. The reported
ratio below is the aggregate MAE over both train-clean ep0 replays, weighted by
the number of compared action elements. Dev values are included for diagnosis.

| Arm | Train Count | Train Occluded | Combined train ratio | Gate |
|---|---:|---:|---:|---|
| SG | 0.2105 | 0.3288 | 0.2738 | Pass |
| K0 | 0.1854 | 0.3647 | 0.2813 | Pass |
| NM | 0.2126 | 0.2904 | 0.2543 | Report only |

Open-loop result files:

- `results/harvestmem_v3_E2/open_loop_sg_final.json`
- `results/harvestmem_v3_E2/open_loop_k0.json`
- `results/harvestmem_v3_E2/open_loop_nm.json`

The SG sensitivity probe was run with a fresh policy reset for each variant.
Mean absolute action changes were:

| Replay | Other fruit coordinate | Same fruit +4 px | Same fruit -4 px |
|---|---:|---:|---:|
| Train Count ep0 | 0.000197 | 0.000181 | 0.000284 |
| Train Occluded ep0 | 0.001148 | 0.000439 | 0.001127 |
| Dev Count ep200 | 0.000226 | 0.000139 | 0.000128 |
| Dev Count ep201 | 0.000560 | 0.000354 | 0.000317 |
| Dev Occluded ep200 | 0.002404 | 0.000511 | 0.000945 |
| Dev Occluded ep201 | 0.040835 | 0.019691 | 0.005295 |

This is diagnostic only; no sensitivity gate was applied.

## Smoke runs and visual inspection

| Arm/task | Result | Output |
|---|---|---|
| SG Count ep200, 200 steps | Completed, no engineering error | `results/harvestmem_v3_E2/smoke/sg_count_ep200/` |
| SG Occluded ep200, 200 steps | Completed, no engineering error | `results/harvestmem_v3_E2/smoke/sg_occluded_ep200/` |
| K0 Count ep200, 200 steps | Completed, no engineering error | `results/harvestmem_v3_E2/smoke/k0_count_ep200/` |
| NM Count ep200, 200 steps | Completed, no engineering error | `results/harvestmem_v3_E2/smoke/nm_count_ep200/` |

Each output contains `result.json`, `trace.jsonl`, `actions.jsonl`,
`subgoal_timeline.jsonl`, `first_front.png`, `first_wrist.png`,
`first_front_wrist.png`, and `front_camera.mp4`. SG timelines contain
`target_selected` at timestep 30. The inspected Count and Occluded first-front
frames show the fruit-tree scene; the front camera is a tree view and does not
include the basket or RM75 body in that particular view. The wrist/front
artifacts are retained for the corresponding robot-side visual check.

## Files added

- `scripts/harvestmem_v3/eval_episode.py`
- `scripts/harvestmem_v3/open_loop.py`
- `results/harvestmem_v3_E2_oracle_recheck.json`
- `results/harvestmem_v3_E2/`

E2 is complete. Stop for review; do not start E3.
