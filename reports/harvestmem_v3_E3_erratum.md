# HarvestMem v3 E3 Erratum

This is a correction to the interpretation and decision-layer accounting of
E3. The original E3 report and results are unchanged.

## Corrected decision metrics

Recomputed from `results/harvestmem_v3_E3/*/count|occluded_ripe/ep*/` using
`scripts/harvestmem_v3/decision_metrics.py`.

| Arm | Count start failures | Count reached N | Count decision failures | Count execution failures | Detach-to-basket | Target agreement |
|---|---:|---:|---:|---:|---:|---:|
| A / SG | 8/40 | 11/40 | 19 | 21 | 83% | 92% |
| D / K0 | 18/40 | 13/40 | 31 | 9 | 89% | 98% |
| E / NM | 5/40 | 15/40 | 20 | 20 | 82% | 96% |

The small D discrepancy from the task-sheet reference (89% vs 88%) is one
episode and within the permitted review tolerance. Occluded target agreement
and green-fruit statistics are retained in
`results/harvestmem_v3_E3_decision_metrics.json`.

## Corrected failure interpretation

The original conclusion under the pre-registered rule remains unchanged:
A was below 40% on both tasks, so E3 stopped at the low-level gate.

The attribution was incomplete. Count failures include a substantial decision
component: starting in place and triggering `finish_signal` with zero fruit,
and continuing after reaching N until overpick or budget exhaustion. These are
not ordinary manipulation failures. The remaining failures are execution
failures such as drops, TCP stalls, and never pulling a fruit outside the
start-failure subset.

The shared start failures were Count episodes 207, 218, 222, and 237 across
all three arms. They ended while the robot remained at rest, with zero fruit
in the basket. This confirms that start and stop are the same sparse rest-state
decision in the current VLA setup.

No original E3 result or report was modified.
