# Implementation Plan Draft

## Objective

Develop the next version from v20 while keeping online runtime below 300
seconds. Linear may use calibration-time `A @ W` only to fit fixed `Q(W)`;
`Q(A)` remains activation-only under the 0819-V2 rule. The long-term online
target is 22000; each version must earn its place through an isolated,
reversible improvement.

## 0819-V2 compliance boundary

- `A @ W` may be used to fit or validate fixed `Q(W)` during calibration.
- Never use `A @ W` or output error to infer/select `Q(A)`. Activation
  parameter profiles must be selected from activation-only reconstruction and
  distribution features; W-only sensitivity statistics remain separate.
- Submission code performs no file I/O. Any synthetic-layout profile produced
  by development tools must be embedded as small constants.
- The second phase may increase N and finals add performance scoring, so search
  complexity must be bounded independently of dataset count.

## Synthetic layout profile plan

1. Generate NVFP4-like proxy tensors for the four observed Linear aspect
   ratios and varied Gaussian, Laplace, heavy-tail, sparse-outlier,
   channel-outlier, block-outlier, correlated, and heteroscedastic layouts.
2. Extract scale-free layout features: aspect ratio, activation/weight peak to
   RMS, channel imbalance, block imbalance, sparsity, and correlation proxy.
3. For Q(A), rank legal scale-search profiles using activation reconstruction
   only. For Q(W), evaluate conservative ridge/trust profiles with held-out
   `A @ W`, never feeding output targets into Q(A) selection.
4. Summarize robust winners by layout bucket and emit an auditable profile
   report. Embed only profiles that win across multiple seeds and retain v16's
   lambda-20 path as mandatory fallback.
5. Online calibration evaluates at most two embedded Q(A) profiles on bounded
   A-only samples. Unknown or weakly separated layouts fall back to compact.

### Synthetic-profile acceptance criteria

- Every layout family has at least three deterministic seeds and a held-out
  seed; no profile is accepted from its training seed alone.
- Q(A) profiles are never scored with `A @ W` or Linear output error.
- A Q(W) profile must beat the trusted candidate on held-out synthetic layouts
  and real calibration holdout without exceeding a 1.5x deviation trust bound.
- Online selection uses two profiles and at most 64 rows per activation shape,
  requires no file I/O, and keeps total calibrated runtime below 270 seconds.

## Candidate directions

1. Modify/reject current form: low-regularization Q(W) fitting and repeated
   residual correction overfit calibration. v17 scored `7062 / 172s` with
   lambda 0.01 and no refinement; v18 fell further to `2887 / 206s` after two
   corrections, even though all local batches improved. Retain the A @ W
   objective, restore v16's trust region, and select parameters with held-out
   calibration outputs.
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
3. Completed: v19 returned `15543 / 198s`, only +4 score for +42 seconds over
   v16. Mark the held-out controller safe but inefficient; retain its validation
   principle while rejecting the full two-lambda Q(W) search as a production
   path unless shape-isolated feedback identifies a materially useful family.
4. Only after held-out selection tracks online direction should one residual
   correction be reintroduced as a separate ablation.
5. Completed in parallel as v20: build a synthetic layout/profile bank under
   0819-V2 compliance, reject unstable synthetic Q(W) lambda mappings, and
   retain only A-reconstruction-selected compact/wide Q(A) scale profiles.
6. Run `py_compile`, `self_check.py`, `git diff --check`, and verify the v13
   Attention section hash if Attention was intentionally unchanged.
7. Update the candidate DAG with the evidence and mark the direction continue,
   modify, or abandon before committing and pushing.
8. For v21, keep v20 Linear fixed and evaluate deterministic shared Q/K
   Hadamard block profiles (4, 8, and trusted 16) against the true calibration
   Attention output. Select once per Attention group with equal weight per
   calibration shape, capped at 64 deterministic rows; do not fit tensor values
   or use test data.

### v21 acceptance criteria

- Calibration-selected Attention profiles must improve both groups 0-4 and a
  disjoint group subset relative to fixed block 16.
- Selection/test winner agreement and aggregate regret are reported across all
  50 local groups. Block 16 remains the fallback for weak or invalid evidence.
- The v20-anchored calibrated estimate must exceed 17000 and remain below 270
  seconds before v21 is committed and pushed.
- Linear output must remain byte-identical to v20; all 22 interface checks and
  `git diff --check` must pass.

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

## Independent holdout rebuild after v21

The confirmed v21 result is `15678 / 163s`, exactly the same score as v20
despite the locally predicted Attention gain. A full audit found that every
paired calibration/test tensor in `datasets_0818` is an exact copy: 1500/1500
Attention carrier/scale tensors and 500/500 Linear tensors compare equal.
Previous calibration-to-test agreement is therefore invalid evidence.

1. Preserve `datasets_0818` as immutable source material; do not create another
   multi-gigabyte copy.
2. Build a deterministic donor manifest within identical structural signatures.
   Pair each group with the nearest different group by calibration-only NVFP4
   distribution features, producing independent test values without observing
   candidate errors.
3. Make `local_eval.py` apply this manifest by default and validate shape/head
   compatibility before evaluation.
4. Calibrate Linear score deltas from the controlled v16 (`15539`) to v20/v21
   (`15678`) online change. Set the Attention extrapolation coefficient to zero
   until an online controlled Attention-only version demonstrates nonzero gain.

### Holdout acceptance criteria

- No group may donate test data to itself, and all donor tensors must have the
  exact shapes required by the receiving group.
- The manifest must be deterministic and contain all 50 Linear and 50 Attention
  groups.
- v21's adaptive Attention advantage must be remeasured on the independent
  donor holdout; leaked-data results are retained only as rejected evidence.
- Evaluation tooling and documentation changes pass `py_compile`, manifest
  validation, and `git diff --check` before push.

### Holdout validation outcome

- Complete non-self mappings: passed for 50/50 Linear and 50/50 Attention.
- Exact calibration/test equality after remapping: 0/500 Linear and 0/1500
  Attention, passed.
- Historical direction: v17 and v18 now regress 40.00% and 47.56% from v16 on
  primary Linear, matching their severe online score collapse; passed.
- Attention score calibration: failed to transfer even after value leakage was
  removed. Keep its online coefficient at zero and require future online
  ablations before using it for score prediction.
