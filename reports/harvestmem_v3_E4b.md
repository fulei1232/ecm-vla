# HarvestMem v3 E4b Report

Date: 2026-11-13

## E4a context

E4a was independently reviewed and passed. Forced stopping raised DR from
0/60 to 16/60 and had zero overpick or invalid stop triggers. The comparison
was nearly paired: paths before the stop intervention were effectively the
same within episode. The remaining DR bottleneck was start failure, while ER's
main bottleneck was fruit drops. SG was not carried into E4b because AR and DR
both finished at 16/60.

## E4b setup

E4b used the true-state tracker and true-state 3D fruit geometry. This is an
upper-bound simulation result, not a perception result. The planner was the
canonical-v31 `FruitPickSolver` reach path using the same `PRE_GRASP=0.12`
construction and `canonical_solver.py` call chain as demonstrations. Planner
failures were fixed to return control to VLA and remain counted in the run.

Formal runs: 480 episodes, all final outputs present:

- Count: DP1R, DPaR, EP1R, EPaR, 60 each; forced stop enabled.
- Occluded: D, DPa, E, EPa, 60 each; D/E are fresh off-controls, DPa/EPa use
  planner reach-all.
- SG was not evaluated.

## Results

Wilson intervals are 95% binomial intervals.

| Arm | Task | Success | Rate | Wilson 95% | Planner calls | Planner failures |
|---|---|---:|---:|---:|---:|---:|
| DP1R | Count, K0 first + stop | 14/60 | 23.3% | [14.4%, 35.4%] | 60 | 0 |
| DPaR | Count, K0 all + stop | 12/60 | 20.0% | [11.8%, 31.8%] | 195 | 2 |
| EP1R | Count, NM first + stop | 15/60 | 25.0% | [15.8%, 37.2%] | 60 | 0 |
| EPaR | Count, NM all + stop | 11/60 | 18.3% | [10.6%, 30.0%] | 189 | 1 |
| D | Occluded, K0 off | 1/60 | 1.7% | [0.3%, 8.9%] | 0 | 0 |
| DPa | Occluded, K0 all | 0/60 | 0.0% | [0.0%, 6.0%] | 91 | 0 |
| E | Occluded, NM off | 2/60 | 3.3% | [0.9%, 11.4%] | 0 | 0 |
| EPa | Occluded, NM all | 1/60 | 1.7% | [0.3%, 8.9%] | 92 | 0 |

Planner failure rates were 2/195 (1.0%) for DPaR and 1/189 (0.5%) for EPaR,
below the 5% engineering threshold.

All Count forced-stop triggers passed the E4a gate: trigger basket count was
exactly N, no overpick occurred, and every triggered episode ended with
`finish_signal` within 300 steps.

### Count by N

| Arm | N=2-4 | N=5-7 | N=8-10 |
|---|---:|---:|---:|
| DP1R | 3/16 | 8/24 | 3/20 |
| DPaR | 2/16 | 7/24 | 3/20 |
| EP1R | 2/16 | 10/24 | 3/20 |
| EPaR | 2/16 | 7/24 | 2/20 |

### Occluded fruit correctness

The fraction of robot-detached fruits that were initially ripe was:

| Arm | Ripe detachments / all detachments | Rate |
|---|---:|---:|
| D | 22/76 | 28.9% |
| DPa | 28/83 | 33.7% |
| E | 29/84 | 34.5% |
| EPa | 33/86 | 38.4% |

The 80% preregistered prediction did not hold. Planner reach-all improved the
NM correctness rate modestly but not to the target; K0 all was higher than K0
off but still below 80%.

## Paired comparisons

Pairs are by episode number against E4a DR/ER Count and fresh D/E Occluded.

| Pair | Both success | Planner arm only | Control only | Both fail |
|---|---:|---:|---:|---:|
| DP1R vs DR | 8 | 6 | 8 | 38 |
| DPaR vs DR | 7 | 5 | 9 | 39 |
| EP1R vs ER | 9 | 6 | 14 | 31 |
| EPaR vs ER | 8 | 3 | 15 | 34 |

## Planner/handoff statistics

| Arm | Planner calls | Replans | Planner steps | Recorded would-clip steps | Handoff target agreement |
|---|---:|---:|---:|---:|---:|
| DP1R | 60 | 0 | 2,694 | 2,694 | 23/60 = 38.3% |
| DPaR | 195 | 3 | 7,898 | 7,898 | 108/193 = 56.0% |
| EP1R | 60 | 0 | 2,694 | 2,694 | 25/60 = 41.7% |
| EPaR | 189 | 3 | 7,879 | 7,879 | 122/188 = 64.9% |
| DPa | 91 | 0 | 3,661 | 3,661 | 16/91 = 17.6% |
| EPa | 92 | 0 | 3,630 | 3,630 | 19/92 = 20.7% |

The `would_clip` field records the guard-v2 return object conservatively; all
planner steps were marked, so it is diagnostic rather than a success gate.

## Smoke and videos

Planner-first Count ep202 completed one planner call, 41 planner steps, and
handed control to VLA; ep207 likewise completed one planner call and 42 steps.
Planner-all Occluded ep200 and ep201 each completed the first handoff; later
VLA execution detached a wrong fruit before another target selection.

Representative formal videos:

- DP1R successes: `results/harvestmem_v3_E4b/DP1R/count/ep212/front_camera.mp4`,
  `results/harvestmem_v3_E4b/DP1R/count/ep220/front_camera.mp4`.
- DP1R failures: `results/harvestmem_v3_E4b/DP1R/count/ep200/front_camera.mp4`,
  `results/harvestmem_v3_E4b/DP1R/count/ep201/front_camera.mp4`.
- DPaR successes: `results/harvestmem_v3_E4b/DPaR/count/ep212/front_camera.mp4`,
  `results/harvestmem_v3_E4b/DPaR/count/ep220/front_camera.mp4`.
- DPaR failures: `results/harvestmem_v3_E4b/DPaR/count/ep200/front_camera.mp4`,
  `results/harvestmem_v3_E4b/DPaR/count/ep201/front_camera.mp4`.
- EP1R successes: `results/harvestmem_v3_E4b/EP1R/count/ep204/front_camera.mp4`,
  `results/harvestmem_v3_E4b/EP1R/count/ep211/front_camera.mp4`.
- EP1R failures: `results/harvestmem_v3_E4b/EP1R/count/ep200/front_camera.mp4`,
  `results/harvestmem_v3_E4b/EP1R/count/ep201/front_camera.mp4`.
- EPaR success: `results/harvestmem_v3_E4b/EPaR/count/ep204/front_camera.mp4`.
- EPaR failures: `results/harvestmem_v3_E4b/EPaR/count/ep200/front_camera.mp4`,
  `results/harvestmem_v3_E4b/EPaR/count/ep201/front_camera.mp4`.

## Decision

Prediction 1 did not hold: DP1R Count was 23.3%, below 40%. Prediction 3 also
did not hold: planner-all ripe-detachment rates were 33.7% for DPa and 38.4%
for EPa, below 80%. The structural planner reach did not solve the remaining
execution/selection problem in this implementation. No perception tracker or
training was started.

All current tracker and planner results use simulator truth, including 3D
fruit positions. A real system would need perception to supply those values.
