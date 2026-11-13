# Overnight R12 diagnosis (2026-11-09) — exploratory, not preregistered

Goal: see the trend before committing to a method. Four arms, the same data and the same recipe as D5
(79999 init, 12000 steps, 8-GPU FSDP, batch 64, LR 2.5e-5→2.5e-6, EMA .999, seed 42), on
`data/finetune_kf/fruit_task_v2_r12_20261109`. Temporal index `order` for every memory arm (closes the
absolute-step clock leak).

| arm | memory | prompt | question |
|---|---|---|---|
| K0 | uniform 16 frames | task instruction | MME-VLA baseline |
| KA | ≤12 release keyframes + uniform rest (16) | task instruction | our event memory |
| OR | uniform 16 frames | instruction + "Fruits already in the basket: k." | information ceiling (diagnostic only) |
| NM | empty memory (all slots masked) | task instruction | no-memory floor |

Queue order: K0 → KA → OR → NM. Each closed-loop dev evaluation runs as soon as its checkpoint exists, on the GPUs
not used by training. The order is chosen so that K0 vs KA, the core comparison, is visible first.

Evaluation: dev episodes 200–239 (R11 dev seed base 4200000+16·e; never used for training), Count and Occluded,
one model seed. No val/test. Count results are stratified by N.

Everything here is a dev diagnostic. It does not change any frozen protocol and is not written into the paper.
