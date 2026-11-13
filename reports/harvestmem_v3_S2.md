# HarvestMem v3 S2 Training Report

Date: 2026-11-11

## Conclusion

S2 training completed successfully for SG, K0, and NM. SG/K0 ran together;
NM completed later in a separate formal run. This report records the training
facts and the known deviations from the intended recipe. The
previous `harvestmem_v3_eval_20261111` rollout is not evidence about these
models on fruit picking; see its `INVALID.md`.

## What ran

Driver command:

```bash
python3 -u scripts/harvestmem_v3/run_s2.py formal \
  --run-dir runs/harvestmem_v3_S2_20261111 \
  --arms sg k0
```

The driver launched two parallel four-device trainings, each for 12,000
steps, with batch size 64, seed 42, cosine learning rate
`2.5e-5 -> 2.5e-6`, warmup 500 steps, and EMA decay 0.999. The formal run
started at `2026-11-11T03:34:28.614565+00:00` UTC and finished at
`2026-11-11T12:23:59.306543+00:00` UTC. The completion marker reports
`passed: true` and no errors.

The recorded source revision was
`414ef44b9afb7a84e6cec9573382700cba4b3151`. The worktree had uncommitted
changes at execution time.

The three-step smoke runs for SG, K0, and NM all passed. Their logs are under
`runs/harvestmem_v3_S2_smoke_20261110/{sg,k0,nm}/logs/`; the smoke driver
finished without errors. This directory is retained as the preflight evidence
for the later NM formal launch.

NM formal command:

```bash
PYTHONPATH=scripts python3 -u scripts/harvestmem_v3/run_s2.py formal \
  --run-dir /media/fulei/fuleivla/ecm_vla/runs/harvestmem_v3_S2_nm_20261111 \
  --arms nm
```

The NM completion marker reports `passed: true` with no errors and finished at
`2026-11-11T23:10:35.204548+00:00` UTC.

## Results

| Arm | First loss (step 0) | Loss (step 1000) | Last logged loss (step 11950) | Training elapsed | Final checkpoint |
|---|---:|---:|---:|---:|---|
| SG | 0.1495 | 0.0080 | 0.0023 | 8:48:03 | `/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/sg/ckpt/mme_vla_suite/sg/11999` |
| K0 | 0.1495 | 0.0080 | 0.0023 | 8:30:09 | `/media/fulei/fuleivla/acm_vla/vendor/robomme_policy_learning/runs/harvestmem_v3_S2_20261111/k0/ckpt/mme_vla_suite/k0/11999` |
| NM | 0.1495 | 0.0081 | 0.0024 | 8:46:33 | `/media/fulei/fuleivla/ecm_vla/runs/harvestmem_v3_S2_nm_20261111/nm/ckpt/mme_vla_suite/nm/11999` |

The final checkpoint is step `11999`; the last scalar printed by the training
logger is step `11950` because logging occurs every 50 steps.

Training logs:

- `runs/harvestmem_v3_S2_20261111/sg/logs/sg_12000.log`
- `runs/harvestmem_v3_S2_20261111/k0/logs/k0_12000.log`
- `runs/harvestmem_v3_S2_nm_20261111/nm/logs/nm_12000.log`

## Deviations and risks

1. **SG coordinate augmentation was not implemented.** The SG training path
   appends `grounded_subgoal` through `install_grounded_prompt`, but it does
   not apply the intended `+/-4 px` coordinate noise augmentation. No SG
   retraining is done in this stage. The planned A arm uses oracle coordinates
   generated from the same projection source as the training labels, so this
   deviation is not expected to affect that oracle comparison. It must be
   addressed before a perception-source C arm.
2. **NM was omitted from the original formal launch.** The policy required all
   three arms; NM's three-step smoke passed, but the first formal command
   started only SG and K0. NM was subsequently trained in a separate formal
   run and completed successfully.
3. **Checkpoint location differs from the intended ECM run directory.** The
   SG/K0 checkpoints were written under `acm_vla/vendor/.../runs/` because the
   training subprocess resolves its relative `--run-dir` from the policy
   repository working directory. The files were not moved. NM used an
   absolute run directory and therefore resides under `ecm_vla/runs/`.

## Gate

| Check | Result |
|---|---|
| SG completed 12,000 steps | Pass |
| K0 completed 12,000 steps | Pass |
| NM completed 12,000 steps | Pass |
| Final checkpoints saved | Pass; all three `11999` checkpoints finalized |
| Formal driver errors | None |
| SG `+/-4 px` augmentation | Not implemented |
| NM training | Pass; separate absolute-path formal run |

The formal training report was written against commit
`414ef44b9afb7a84e6cec9573382700cba4b3151`; the worktree contains uncommitted
changes.

## Status

All three formal trainings are complete. SG/K0 remain in their original
`acm_vla` location; NM is in the absolute `ecm_vla/runs` location documented
above.
