# HarvestMem v3 E4b Design

E4b is design-only; no implementation or run was performed.

## Planner and environment

Use the same `FruitPickTaskV2-v0`, revision 9, `rm75b_eg2`, `no_lamp_wait=True`,
and `guard_v2` path as E4a. Construct the expert planner with:

```python
FruitPickSolver(env, episode.step, ik_mode="canonical_v31", ik_seed=0,
                ik_candidates=64)
```

The planner must use the same obstacle refresh and canonical-v31 settings as
`scripts/harvestmem_v3/run_episode.py`. Its `step_fn` is the episode wrapper,
so oracle snapshots, trace, videos, and action provenance continue through the
same recorder.

## Handoff

When the oracle selects a target, map its slot to the current actor and use the
same `horizontal_grasp_pose`/`PRE_GRASP=0.12` reach construction used by
`solver.pick`. The planner executes only the reach-to-pregrasp segment. At the
pregrasp boundary, return control to VLA for approach, grasp, pull, carry, and
release. If the oracle changes target after wind/drop recovery, hand control
back to the planner for the new target. E4a forced stop remains authoritative
after N is confirmed.

## Memory frames

Planner frames remain in the history buffer. They are observations from the
same episode and preserve the temporal state that VLA sees at handoff. Action
records mark planner and policy sources separately so the distribution can be
audited.

## Planning failure

On planning failure, record `planner_failure`, target slot, exception, and the
current oracle state. Then either return to VLA at the current observation or
abort the episode as an engineering failure; the choice must be fixed before
registration. Report planner failure rate, recovery rate, and downstream
success separately.

## Distribution-shift check

Use expert dev replay reach-end boundaries. Feed the exact history and prompt
through `open_loop.py`, compute the VLA/expert MAE ratio at those boundaries,
and compare it with E2's all-block ratio. A large increase indicates the
handoff state is outside the training distribution and should be reported
before closed-loop evaluation.

## Proposed evaluation

First evaluate K0+planner+forced-stop and NM+planner+forced-stop on Count and
Occluded dev 200-239. Keep SG as a secondary arm only if the handoff audit
shows its grounded prompt is being used. A 2-GPU server/client arrangement is
adequate; 80 episodes per arm should take roughly 40-60 minutes depending on
stalling. Any later val result must be separately pre-registered.
