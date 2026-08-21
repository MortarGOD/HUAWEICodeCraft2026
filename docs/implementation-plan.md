# Implementation Plan Draft

## Objective

Develop the next version from v16 while keeping online runtime below 300
seconds and improving output accuracy through calibration-time `A @ W` fitting
of `Q(A)` and/or `Q(W)`. The long-term online target is 22000; each version
must earn its place through an isolated, reversible improvement.

## Candidate directions

1. Modify/reject current form: low-regularization Q(W) fitting and repeated
   residual correction overfit calibration. v18 scored only 2887 online even
   though all local batches improved. Retain the A @ W objective, restore v16's
   trust region, and select parameters with held-out calibration outputs.
2. Next: activation-scale selection using calibration-derived blockwise output
   residual statistics in addition to the rank-2 `W^T W` approximation. Risk:
   state size and dynamic scoring cost may grow. Prototype only if direction 1
   cannot clear the 0.5% gain threshold.
3. Modify: Attention output-aware calibration must model the coupled softmax
   objective directly. Diagonal Q/K sensitivity and random sign selection were
   rejected; any successor must first beat v13 behavior on all 50 local groups.

## Executable steps

1. Use v16's recorded Linear groups 0-3 and Attention groups 0-4 result as the
   primary benchmark (`1904.398739`, `4.311844`, online `15539 / 156s`).
2. Completed: v19 branches from v16 and splits bounded rows within every
   calibration shape into fit/validation partitions. It compares lambda 10
   against trusted lambda 20, chooses once per layer on validation `A @ W`,
   and applies a correction-norm trust region.
3. Submit v19 for online feedback. Locally all tested layers select lambda 20,
   exactly reproducing v16 accuracy at an estimated 210 seconds. Record how
   many hidden layers select lambda 10 if future diagnostics expose the count.
4. Only after held-out selection tracks online direction should one residual
   correction be reintroduced as a separate ablation.
5. Run `py_compile`, `self_check.py`, `git diff --check`, and verify the v13
   Attention section hash if Attention was intentionally unchanged.
6. Update the candidate DAG with the evidence and mark the direction continue,
   modify, or abandon before committing and pushing.

## Acceptance criteria

- All 22 interface/output-format checks pass.
- Primary Linear MSE improves by at least 0.5%, or runtime improves by at least
  5% with no more than 0.2% Linear MSE regression.
- The disjoint subset agrees with the direction of the primary result.
- Estimated online runtime remains below 270 seconds, leaving a 10% margin to
  the 300-second limit.
- No Attention regression unless an intentional Attention experiment has
  evidence on both subsets.
- For adaptive controllers, selecting the trusted branch must reproduce v16
  output exactly and calibrated runtime must remain below 270 seconds.
