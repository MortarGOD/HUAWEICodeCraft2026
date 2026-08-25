# Implementation Plan Draft

## Objective

Develop the next version from v21 while keeping online runtime below 300
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

## v22 Linear-only optimization plan

Objective: exceed `17000` from the v21 anchor `15678 / 163s`, using only
leakage-free Linear evidence. Attention is frozen and contributes zero to the
prediction. The primary holdout is Linear groups 0-3; groups 4-7 and the other
three structural batches are disjoint gates.

Candidate DAG:

```text
v21 (15678 / 163s; runtime anchor)
  └─ v22 Linear parameter screen
      ├─ Q(A) scale offsets 0..4 / 0..5
      ├─ W-only output metric rank 4
      ├─ balance exponent 0.30 / 0.45 / 0.50
      └─ Q(W) ridge lambda 10 / 30 / 40
```

Execution and risk controls:

1. Screen each parameter in isolation against v21 on the independent donor
   holdout. No candidate may observe donor tests during calibration.
2. Reject low-lambda Q(W) fits unless every structural batch improves; v17/v18
   prove that calibration fit alone is unsafe.
3. Combine only independently positive parameters, then rerun groups 0-7 and
   all 50 Linear groups. Revert parameters whose gains disappear off-primary.
4. Use the v16-to-v21 online coefficient for score estimation. A push candidate
   must estimate above 17000, stay below 270 seconds, pass 22/22 format checks,
   and keep Attention byte-identical to v21.

Acceptance caveat: if no leakage-free candidate estimates above 17000, do not
manufacture that prediction. Push only a materially improving, cross-batch
candidate and report the evidence-based estimate and remaining gap.

### v22 selection outcome

- Retain: balance exponent `0.44`, Q(A) offsets `0..4`, fixed Q(W) offsets
  `0..3`, rank-4 W-only sensitivity for Q(A), and rank-4 calibration-A
  sensitivity for fitting fixed Q(W).
- Retain λ20 ridge: removing it slightly regressed the disjoint subset even
  though it was faster. Additional residual refinements selected no useful
  donor-generalizing rows and were rejected.
- Reject: α0.30/0.42/0.46/0.48/0.50, offset5, lower λ, block-wise α, quant-aware
  sign search, global Q(W) profiles, coordinate block fitting, channel sorting,
  rank16/32, and Hadamard32. Each was weaker, inconsistent, or reversed on the
  independent donor holdout.
- Full validation: all 50/50 Linear groups improve. Total MSE is
  `23785.392948` versus v21 `24713.431183` (-3.755%); every consecutive
  five-group batch improves by 2.45%-4.30%.
- Primary calibrated estimate is `16138 / 206s`; the all-group local runtime
  ratio is 1.295, projecting about `211s`. The evidence does not support a
  >17000 center estimate, but v22 is accepted as the first leakage-free,
  all-group-positive step toward that target.

## v23 learn-without-copying plan

New online anchors are v22 `16359 / 197s` and external `solution_cxw_v1.py`
`17562 / 249s`. The objective is to identify causal mechanisms in the latter,
then implement independent variants that exceed 17562 without reproducing its
code structure.

Mechanism hypotheses:

1. Linear sequential block error propagation is a candidate, not an assumed
   gain: after quantizing one 64-channel weight block, it moves residual into
   future blocks, but must beat v22 on the independent donor holdout.
2. Correct standard Attention is computed over sequence positions. The old
   local evaluator and v21 calibration accidentally computed head-to-head
   attention per token, invalidating all previous Attention proxy conclusions.
3. Reciprocal per-head Q/K balancing and a shared orthogonal transform may be
   the dominant Attention gain. Adaptive rotation and V profiles must justify
   their complexity independently.

Execution:

1. Correct only the evaluator's Attention axes and measure v22 versus cxw on
   independent donors. Separately report Linear and Attention.
2. Reject external Linear propagation for v23: on donor groups 0-3 its total
   MSE is `2153.199535`, versus v22's `1889.581486` (+13.95%). Keep v22 Linear
   exactly unchanged rather than trading verified generalization for a guess.
3. Independently ablate Attention balance, shared rotation, scale offsets, and
   energy preservation. Retain the smallest fixed mechanism that is stable
   across disjoint batches; do not reproduce the external controller.
4. The retained v23 uses reciprocal RMS balance, shared Hadamard-64, Q/K
   offsets `0..4`, V offsets `-1..3`, and energy penalty `0.125`. Full 50-group
   standard-Attention MSE is `75.937482`, down 14.52% from v22's `88.832125`;
   external is `74.921704`. Runtime is `45.16s`, versus v22 `78.35s` and
   external `81.30s` in the same Attention-only run.
5. Accept only after all 1100 interface/format checks, byte-identical v22
   Linear output, and projected runtime below 270 seconds. Because no clean
   Attention-only online anchor exists, report both a conservative floor and a
   mixed-anchor estimate instead of presenting a false-precision score.

Humanize is not installed in this workspace. The executable steps and
acceptance criteria above were therefore refined manually before code changes.

## v44 Rotated Runtime Smooth adaptation plan

Anchor v43 has no true runtime channel smoother. It applies a calibration-fixed
reciprocal multiplier and a paired Hadamard-64 transform, then chooses dynamic
HiF4 scales mainly from each row/block and a fixed output sketch. RRS adds the
missing current-input statistic, but its fused-GEMM formulation cannot be copied:
the paper multiplies each partial block output by its runtime scale, while this
contest exposes only one fixed `Q(W)` and ordinary dequantized matrix multiply.

Humanize is still unavailable, so the plan and acceptance criteria below were
refined manually.

Candidate DAG:

```text
v43 (fixed SmoothQuant-like balance + Hadamard-64)
  └─ v44 RRS adaptation
      ├─ exact paper primitive: runtime absmax per rotated channel
      ├─ reject: uncompensated A/s (changes A@W)
      ├─ retain: victim-aware candidate cost derived from runtime profile
      ├─ calibrate: compare v43 and RRS strengths by exact A@W per layout
      └─ online: route known row layouts; exact v43 fallback otherwise
```

Execution steps:

1. Compute `s_j=max_i |(A R)_ij|` on each current activation after the existing
   paired transform. Aggregate it inside each physical HiF4-64 group and derive
   bounded inverse-square victim weights. Rotation makes spike-driven scales
   more uniform, matching RRS's required order of operations.
2. Add those weights only to legal E6M2/E1_8/E1_16 candidate scoring. Returned
   dequantized activation stays in the original transformed domain; never return
   `A/s` without an inverse weight transform.
3. During calibration, evaluate v43 and RRS candidates on sampled rows from
   every calibration layout using exact `A@W` against the already fixed Q(W).
   Select a strength independently for each observed row-count layout; aggregate
   duplicate layouts before selection and fall back exactly on unseen layouts.
4. Preserve Attention byte-for-byte and retain every v43 Linear path as fallback.

Acceptance criteria:

- RRS statistics must be computed from the current online activation after
  rotation, not copied from calibration maxima.
- The FP32 paired transform remains output-equivalent, and the implementation
  never depends on unavailable partial-GEMM output scaling.
- A row-layout profile is stored only with at least 0.3% aggregate exact-output
  gain; duplicated row counts must agree in aggregate and unseen layouts fall
  back exactly to v43.
- On local raw and donor probes, catastrophic cases may not become worse than
  v43; the intended RRS-enabled groups must show positive exact-output evidence.
- State and all returned fields pass `self_check.py`; projected online runtime
  remains below 250 seconds and under the 300-second hard limit.

Selection outcome:

1. A global per-shape gate selected no RRS profile and reproduced v43 exactly,
   while adding about eight seconds of calibration search. Reject that empty
   implementation. Per-row-count routing exposes the paper's genuinely dynamic
   statistic while keeping exact fallback for unknown layouts.
2. Direct `A/s -> HiF4 -> *s -> HiF4` was tested separately on seven varied
   shapes. Its output MSE ratio versus v43 ranged from `1.296x` to `1.907x`;
   the second legal HiF4 projection destroys the theoretical benefit. Reject it.
3. On the first 9 valid Linear groups, v44 improves `-1661.416 -> -1653.848`
   (`+7.568`). Across all 43 non-degenerate random Linear groups it improves
   `-21040.128 -> -21034.544` (`+5.584`), with 5 wins, 23 ties, and 15 bounded
   losses. The largest gain is `+25.527`; the largest loss is `-7.763`, so no
   catastrophic tail is introduced.
4. Full-Linear paired timing is `200.16s` for v43 and `199.56s` for v44. The
   runtime absmax/profile cost is below run-to-run noise; use the v43 online
   anchor rather than claiming a speedup. Attention remains exactly unchanged.
5. The current random suite has mostly unique extreme signatures, so its stale
   old donor manifest cannot be rebuilt without self-pairs. Treat the independent
   test tensors and complete shape panel as the available transfer evidence.

## Post-v42 paper-list research plan

The 2026-08-24 audit of `docs/paper_list.md` is recorded in
`docs/quantization-strategy-library.md`. All 21 list entries have been resolved
to 20 unique primary sources and studied from full text. Humanize remains
unavailable, so the following steps and acceptance criteria were refined
manually.

Candidate order:

1. Implement an isolated MR-GPTQ-inspired static act-order for the existing
   residual-feedback Q(W). Compute the legal HiF4 grids in physical order,
   change only the calibration feedback order, and restore original order in
   the returned parameters. Compare group-, E1_8-, and leaf-unit granularity.
2. If static act-order is numerically stable, add an offline hierarchical beam
   over E6M2/E1_8/E1_16 using exact calibration `A@W`. Include an OAS-like
   controlled-saturation candidate and separate top-1 from body error.
3. Derive scale candidates from actual S1P2 hierarchy decision boundaries,
   union them with current code offsets, and remove duplicates before output
   scoring.
4. Separately prototype a Q(A) risk controller using row-RMS variation,
   top-1 energy, near-zero ratio, peak/RMS and shape. Calibration may fit the
   controller with `A@W`; test execution chooses at most two fixed profiles and
   always retains an exact v42 fallback.
5. Keep structured activation residual importance and deterministic EDEN
   shrinkage as diagnostic branches until one of the first four directions
   establishes a transferable signal.

Acceptance criteria:

- Each submitted version changes one major mechanism and keeps a byte-exact
  v42 fallback for signatures outside its gate.
- Q(W)-only branches add no online tensor operation and no measurable online
  runtime; Q(A) routing may add at most 5% to the v42 online anchor.
- Every candidate passes legal HiF4 state checks, FP32 equivalence for paired
  transforms, all 1100 interface checks, and catastrophic-tail tests.
- Local negative evidence can lower priority but does not veto a theoretically
  sound, bounded-risk online ablation, because local and online distributions
  are known to differ.
- Residual channels, extra GEMMs, LoRA, stochastic inference, new metadata bits,
  or any format outside the six-function interface are not submission paths.

## v43 controlled-saturation Q(W) plan

The confirmed v42 result is `17881 / 220s`, versus v41 `17930 / 217s`.
Therefore the gated Attention Q/K hierarchy is online-negative (`-49 / +3s`)
despite positive local donors. v43 restores v41 Attention exactly and changes
only Linear Q(W).

The existing feedback Q(W) searches E6M2 codes at offsets `0..3` from the
peak/7 no-overflow scale. Four Over Six and OAS show that sacrificing a small
peak error can improve the much larger body of a block. v43 adds offset `-1`
as one controlled-saturation candidate for Q(W) on ordinary wide shapes only
(`output < input <= 2*output`); the existing output-aware score chooses it
independently per HiF4 group. Q(A), global Q(A) search, feedback order, channel
layout, multiplier, and transform stay unchanged.

Acceptance criteria:

1. Attention output/state is byte-identical to v41, removing the measured v42
   regression.
2. Compare offset sets `0..3`, `-1..3`, and only if safe `-2..3` on nearest,
   farthest, independent, and random extreme Linear panels. Reject any set that
   creates a failed group or material catastrophic-tail regression.
3. Report how often negative offsets are selected and whether they affect all
   shape families, rather than inferring value from aggregate local score only.
4. Keep projected online runtime below 250 seconds from the v41/v42 anchors.
5. Pass `py_compile`, `git diff --check`, full HiF4 state checks, and all 1100
   interface cases before committing. Humanize is unavailable, so these steps
   and criteria were refined manually.

Selection outcome:

- Unconditional `-1` is nearly flat on ordinary groups (`+0.064` near and
  `+0.462` far) and on random groups 0-10 (`+0.018` aggregate), but a 128:1
  compression extreme regresses `-4.413`. Reject the unconditional form and
  do not test offset `-2`.
- The 2:1-wide gate changes only the online-relevant wide family. Across four
  repeated wide groups it gives `+0.081` near and `+0.609` far, with 2/4 and
  3/4 group wins respectively. The first near/far wide group improves on both
  panels; independent groups remain mixed, so forecast conservatively.
- The extra candidate costs about `+8s` on the four-wide near panel and
  `+21s` in a noisy far run. Since one of four online Linear families is wide,
  forecast total online time at `225-235s`, safely below 300 seconds.
- Non-wide Linear weight and activation parameters are exact v41 fallbacks;
  Attention code/constants are exact v41. Complete validation passes
  `1100/1100`, `py_compile`, and `git diff --check`.
- Forecast v43 at `17915-17955 / 225-235s`. Most of the expected recovery is
  the controlled v42-to-v41 Attention rollback; the new Q(W) candidate earns
  only a small positive allowance until online feedback proves transfer.

## v24 controlled Linear recovery plan

Online anchors are v22 `16359 / 197s`, v23 `16477 / 193s`, and the external
mixed implementation `17562 / 249s`. v24 keeps v23 Attention fixed and targets
the unexplained Linear/adaptation gap without copying the external code.

Candidate DAG:

```text
v23 (16477 / 193s)
  └─ within-group exact-twin-excluded evaluator
      ├─ trusted v22 fixed-weight ridge (fallback)
      ├─ bounded sequential output-residual compensation
      ├─ propagation window and trust shrinkage ablations
      └─ optional row-wise fallback selected on excluded calibration shape
```

Execution and acceptance criteria:

1. Add a leave-one-out mode that removes each test tensor's exact calibration
   twin before fitting state. It preserves the group's layout while preventing
   the target value from entering calibration. First require it to order the
   known v22/v23/external feedback more plausibly than raw or donor evaluation.
2. First correct the evaluation objective to the official per-case relative
   score. The MSE-sum ranking is rejected because it hides regressions on
   low-baseline, high-score-weight cases.
3. Screen one variable at a time on groups 0-3, gate on groups 4-7 and
   structural batches, and retain v23 when the validation margin is weak.
4. Keep Attention byte-identical to v23. Require all format checks, runtime
   below 270 seconds, and evidence in both twin-excluded and donor views.
5. Score estimates use only controlled same-branch online anchors. No mixed
   result is algebraically decomposed into component points.

### v24 selected branch

The official objective changes the diagnosis. Across all 50 raw Attention
groups, v23 has proxy `9097.752851`, while the external implementation has
`9329.868856`; their close MSE totals had hidden a meaningful per-case gap.
Linear remains weaker externally on both raw and donor primary groups, so v24
is kept as a controlled Attention-only change and Linear propagation is
deferred to a later isolated branch.

v24 changes the Q statistic from `sqrt(mean(head energy))` to
`mean(sqrt(head energy))` within each replicated KV group. This avoids letting
one high-energy Q replica dominate all peers. A calibration-only density gate
uses full-head rotation for one dense layout and Hadamard-64 otherwise. Q/K
offsets `0..4` and energy penalty `0.125` remain; copying the external offset
set and penalty regressed.

Acceptance evidence:

- Official raw Attention proxy: v23 `9097.752851`, external `9329.868856`, v24
  `9607.884916`. v24 wins 9/10 consecutive five-group batches versus v23.
- Twin-excluded groups 0-9: v23 `-781.733716`, external `-671.010899`, v24
  `-603.164937`. Absolute transfer remains poor, but v24 improves both relevant
  comparisons without seeing the held-out twin.
- Attention local runtime is `54.99s`, between v23 `45.48s` and external
  `67.90s`; Linear is unchanged, so the online projection remains below 210s.

## v25 bounded Linear residual propagation

The v24 online result `16500 / 198s` supplies another controlled Attention-only
anchor: its large local proxy gain produced only +23 points. v25 therefore
freezes Attention and isolates the remaining major external mechanism family,
fixed-weight sequential error compensation.

Candidate DAG:

```text
v24 (16500 / 198s)
  └─ Linear residual propagation
      ├─ diagonal future-channel projection: reject
      └─ 32-row sample-space ridge, natural order
          ├─ trust 0.75: modify (higher donor loss)
          └─ trust 0.50: retain as v25
```

Acceptance criteria and outcome:

1. Q(A) must be completely determined before any residual propagation; only
   the returned fixed Q(W) may change. Passed.
2. Primary and disjoint raw official proxies must improve, and donor regression
   must be materially smaller than the external implementation. Passed on
   groups 0-3 and 4-7.
3. All ten full-dataset batches must improve. Passed: total proxy gain is
   `+192.167940`, and aggregate MSE falls 1.68%.
4. Total projected runtime must remain below 270 seconds. Passed: Linear local
   time falls 31.9%, projecting total online runtime around `145-170s`.
5. Attention must remain byte-identical to v24 and all format checks must pass
   before push.

## v26 feedback capacity plan

Anchor v25 is `17049 / 176s`. Its all-50 raw Linear proxy gain of `192.167940`
maps to +549 online points for this exact branch, providing a local slope of
about `2.8568 points/proxy-unit`.

Candidate DAG:

```text
v25: rows32 / window4 / trust0.50
  ├─ trust0.75
  ├─ rows64 / window4
  ├─ rows128 / window4
  ├─ rows32 / window8
  └─ rows64 / window8
```

Acceptance criteria:

1. Change one capacity dimension at a time on groups 0-3, then confirm the
   winner on groups 4-7 without retuning.
2. Require positive official-proxy gain on both raw batches. Donor may regress
   because it reversed the verified v24-to-v25 online direction, but the loss
   must remain smaller than the external implementation's donor loss.
3. Run all 50 groups and require at least 100 additional proxy units, which
   projects v26 above 17300. Runtime must remain below 240 seconds online.
4. Preserve v25 Attention byte-for-byte and pass all 1100 format checks.

Humanize remains unavailable; the steps and acceptance criteria are refined
manually before implementation.

### v26 selection outcome

- Reject rows 64/128: both reduce the primary official proxy, confirming that
  32-row subsampling acts as regularization.
- Continue window 8 and trust 0.75: both are independently positive and their
  combination improves primary and disjoint raw batches while adding only a
  small donor penalty.
- Full 50-group proxy is `16027.331849`, +179.696813 from v25; every consecutive
  five-group batch improves. The controlled center estimate is `17562`.
- Local Linear runtime is `365.30s`, no regression from v25. Preserve v25
  Attention exactly and complete all format checks before push.

## v27 activation-energy ordering

Online anchor v26 is `17232 / 183s`. Its +179.696813 local proxy gain maps to
only +183 online points, so capacity expansion has reached diminishing returns.
v27 changes propagation topology while keeping every v26 capacity parameter.

Candidate DAG:

```text
v26: natural order / rows32 / window8 / trust0.75
  ├─ descending activation-block energy
  └─ ascending activation-block energy (diagnostic only)
```

Acceptance criteria:

1. Returned HiF4 blocks must be restored to original channel order; Q(A) must
   remain finalized before ordering or output-residual propagation.
2. Descending order must improve official proxy on groups 0-3 and 4-7 without
   donor loss approaching the external implementation.
3. Require all-50 gain on at least 8/10 batches and projected runtime below
   230 seconds. Otherwise retain v26.
4. Preserve v26 Attention byte-for-byte and pass all 1100 format checks.

Humanize remains unavailable, so acceptance criteria are refined manually.

### v27 selection outcome

- Descending activation energy beats natural order on both raw and donor
  primary/disjoint gates; ascending energy regresses and is rejected.
- Under energy ordering, window 16 and 32 remain positive on all four gates.
  Select window 32, then stop expansion because marginal gain is diminishing.
- Reject trust 1.0 and rows 64: both reverse the primary improvement.
- All-50 proxy improves by `107.548101`, with 10/10 batches positive. Local
  Linear time grows 12.0%, projecting total online runtime around 202 seconds.

## v28 standard-baseline replay and sparse Attention guard

Online anchor v27 is `17283 / 192s`. The new `rand_sample` and organizer
`hif4_quant.py` replace the submitted-solution proxy with the published
per-case score. The ten-version replay rejects the equal-weight random total
as an online estimator (Pearson -0.432 on the common valid intersection), but
retains it as a robustness suite and shape-family panel.

Candidate DAG:

```text
v27 (17283 / 192s)
  ├─ Linear feedback-drift fallback
  │   └─ reject: improves extremes but remains negative and mixes causality
  └─ Attention calibration-density guard
      ├─ dense: retain full-head product-preserving transform
      └─ sparse: identity transform + reciprocal Q/K balance (v28)
```

Acceptance criteria and outcome:

1. Score with the supplied standard quantizer and independent tests. Passed.
2. Eliminate `head_dim=32` zero-block failures. Passed: 47/47 valid Attention
   groups complete, versus three v27 failures.
3. Require a positive common-case score and bounded runtime. Passed: common
   Attention score `-20155.855 -> 5182.602`; local quantization time
   `32.80s -> 34.67s`.
4. Keep Linear byte-identical to v27 and pass every output/state check. Passed:
   `1100/1100` on `rand_sample`.

Humanize remains unavailable; the plan and acceptance criteria were refined
manually before selection.

## v29 complete cxw_v1 Linear transfer

Online anchor v28 is `17283 / 184s`. At the user's explicit direction, v29
transfers the complete `solution_cxw_v1.py` Linear path rather than selecting
only individual mechanisms, while retaining v28 Attention behavior.

Candidate DAG:

```text
v28: bounded low-row energy-ordered Linear + sparse-safe Attention
  └─ v29: cxw_v1 Linear constants/helpers/calibration/dynamic path
          + v28 Attention calibration/dynamic path
```

Transferred Linear mechanisms include exponent-0.39 balance, bounded sign
search, Hadamard-64, all-calibration activation quantization, contiguous
block-GPTQ, ridge 0.1, and shape/output-row adaptive propagation width and
weight offset space. Q(A) remains the source implementation's fixed dynamic
quantizer; calibration output fitting only produces fixed Q(W).

Acceptance criteria and outcome:

1. Linear source must match the reference implementation except version and
   Attention-only constants. Passed.
2. Attention outputs must match v28 on a direct donor comparison. Passed on
   all 25 checked cases.
3. Full official random evaluation must complete below 300 seconds with no
   failures. Passed: score `8849.404`, quantization `217.70s`.
4. Submission and version file must match and all format checks must pass.
   Passed: `1100/1100`.

Humanize is unavailable; acceptance criteria were refined manually.

## v30 calibration-output-gated global Q(A)

v29 is rejected as a submission base after the online result `17578 / 272s`:
relative to v28 it costs 88 seconds for 295 points. v30 therefore starts from
v28 and preserves its weight and Attention paths.

Candidate DAG:

```text
v28
  ├─ reject: fixed Q(A) offsets/gains (no held-out benefit)
  ├─ reject: output-selected signs (score loss, doubled local time)
  ├─ reject: unconditional rank-32 Q(A) (ordinary donor regression)
  └─ v30: rank-32 global W row-space Q(A)
          ├─ force only when in_features >= 32 * out_features
          └─ otherwise require exact calibration A@W MSE gain
```

The online path uses only A and a rank-32 W row-space sketch learned during
calibration. It never stores W or computes test A@W. The gate compares the
joint output MSE of v28 Q(A) and the global policy against calibration truth
after Q(W) has been fixed.

Acceptance evidence:

1. Six online-shaped donor groups are exactly equal to v28 after the gate.
2. Six ordinary random groups improve `421.077 -> 432.973` Linear points.
3. Two strong-compression tails improve `-25870.818 -> -4741.572` while the
   ten-group quantization time remains essentially flat.
4. Deterministic full random score is `-31395.201` (`Linear -36807.922`,
   Attention `5412.720`) in about `200.63s`, a large tail improvement over the v27/v28 family
   but not a directly transportable online-score estimate.

## v31 joint Q(W)/Q(A) selection

Online anchor v30 is `17521 / 174s`. v31 tests the complementary-error
hypothesis without paying v29's unconditional Q(W) cost.

```text
v30 baseline Q(W), baseline Q(A)
  ├─ baseline Q(W), global Q(A)
  ├─ aggressive Q(W), baseline Q(A)   (compression >= 32:1 only)
  └─ aggressive Q(W), global Q(A)     (compression >= 32:1 only)
        ↓
aggregate A@W gate + five-layout robust gate
```

The robust gate requires at least three improving layouts, no layout worse
than 5%, and at least 0.3% robust geometric gain. Ordinary shapes must also
pass v30's aggregate gate. This preserves v30's generalization behavior while
allowing Q(W)/Q(A) error cancellation only where it is consistent.

Evidence:

1. Online-shaped donor groups 0-5 remain exactly equal to v30/v28.
2. Random groups 2/3/5 change `236.701 -> 237.395` with essentially equal
   runtime; group 2 and 5 preserve v30 while group 3 rejects its regression.
3. Compression groups 8/9 improve `-4741.572 -> -4071.926`.
4. Full random score improves `-31395.201 -> -30967.769`; quantization time is
   `162.70s` in the final run and all scoring groups complete.

## v32 row-layout Q(A) policy

The confirmed v31 result is `17493 / 202s`, versus v30 `17521 / 174s`.
Therefore the aggressive strong-compression Q(W) candidate and the repeated
five-layout joint scorer are rejected: the local tail gain did not transfer,
while their runtime cost did.

```text
v30: one fast Q(W) + one aggregate baseline/global Q(A) gate
  └─ v32: preserve the same Q(W)
      ├─ aggregate gate remains mandatory
      ├─ separately reject global Q(A) for identifiable row layouts
      ├─ use a 2% margin below 16 rows and 0.5% otherwise
      └─ duplicate/unknown row layouts fall back to the aggregate decision
```

Acceptance criteria and outcome:

1. Remove v31's second Q(W) and per-layout joint candidate loop. Passed.
2. Preserve v30 exactly when the aggregate gate rejects, when layouts are not
   identifiable by row count, and on strong-compression fallback shapes.
3. Independent donor groups 0-7 are case-for-case identical to v30; their
   Linear total is `2496.869`. The local runtime delta is about 2%.
4. Full random score is `-31392.796` (`Linear -36805.516`, Attention
   `5412.720`) in `156.34s`, a small `+2.405` over v30 rather than a claimed
   online-score predictor. All random groups complete without failure.
5. Submission state uses string keys only and must pass all 1100 format checks
   before push.

Humanize remains unavailable; the executable steps and acceptance criteria
were refined manually.

## v33 joint W/Q(W) output-space optimization

The confirmed v32 result is `17507 / 188s`, so its layout-conditioned gate is
not a useful continuation. v33 returns to v30's Q(W) and changes the Q(A)
objective itself.

```text
v30: fixed Q(W) + rank-32 ||(A-Q(A))W.T||
  └─ v33: fixed Q(W)
      ├─ project W and Q(W) into the same rank-2048 output subspace
      ├─ minimize ||A W.T - Q(A) Q(W).T|| by block coordinates
      ├─ exact calibration A@W selects baseline/rank-32/joint policy
      ├─ 3 sweeps for rows <= 512; 1 sweep for longer activations
      └─ batched candidates + QR-only tall-matrix subspace acceleration
```

Acceptance criteria and evidence:

1. Preserve v30's Q(W) and v28 Attention, so the new evidence isolates Q(A).
   Passed.
2. Require positive transfer on two disjoint donor batches. Passed:
   `+81.253` on groups 0-7 and `+95.021` on groups 8-15, with no selected
   group regression.
3. Exceed a calibrated 19000 online estimate. Passed: the 16-group mean delta,
   expanded to 50 Linear groups and mapped with the historical coefficient,
   predicts `19095` from the v30 online anchor.
4. Avoid an unconditional multi-sweep cost on the largest row layout. Passed:
   1024-row cases use one sweep; shorter equal-weight cases use three.
5. Validate all output/state constraints before push. Passed `1100/1100` on
   the complete random suite.

Candidate DAG decision: continue v30 -> v33. Reject v32's row-layout gate,
rank-4096 PCA (lower donor score and higher cost), adaptive Attention energy
profiles (donor regression), ascending coordinate order, and unconditional
second Q(W). Retain the strong-compression alternating Q(W) branch only behind
its existing exact robust calibration gate.

Humanize remains unavailable; acceptance criteria were refined manually.

## v34 distribution-shift and efficiency plan

v33's confirmed result is `17682 / 249s`. Its joint output objective adds real
score, but only 10.2% of the predicted increment transfers online. v34 will
therefore preserve the mechanism while reducing capacity and selecting it
only with evidence that survives distribution shift.

Acceptance criteria:

1. Add a far-neighbor holdout within each exact structural signature and use
   it together with the existing nearest-neighbor donor panel.
2. Measure score and quantization time by all four repeated Linear signatures;
   do not count repeated group IDs as independent shape evidence.
3. Prefer a candidate that retains at least 70% of v33's far-holdout gain while
   recovering at least half of its runtime increase over v30.
4. Keep v28 Attention unchanged and validate output/state constraints
   `1100/1100` before push.
5. Forecast with the v30-to-v33 mechanism-specific coefficient (`0.292`) and
   report an interval; do not claim 19000 without a new controlled online
   anchor.

Candidate DAG:

```text
v30 (fast output-aware Q(A))
  └─ v33 (rank-2048 joint W/Q(W), positive but expensive)
      ├─ v34-a: lower rank / one-sweep joint search
      ├─ v34-b: shape-gated rank-2048 search
      └─ v34-c: short-row-only multi-sweep refinement
```

Humanize remains unavailable; these executable acceptance criteria are the
manual refinement required by the project edict.

Outcome:

1. Far-neighbor holdout added and deterministic. Passed.
2. All four repeated signatures measured independently. Passed; square
   matrices provide the strongest gain and ordinary wide matrices the weakest.
3. Final v34 retains 88.2%/87.4% of v33's nearest/farthest gain. It removes the
   wide-matrix rank-2048 path and two tall-matrix sweeps; algorithmic work is
   strictly lower, although absolute local timings show thermal-order noise.
4. Strong-compression behavior is byte-for-byte score-equivalent to v33 on
   random groups 8-9. Attention remains unchanged.
5. Mechanism-specific forecast: `17620-17710 / 210-235s`, centered near
   `17663 / 222s`. Full random-suite validation passes `1100/1100`.

Candidate DAG decision: select v34-b+c (shape gate plus short-row square
refinement). Reject the global five-offset cut because it loses 7.5% gain for
only a small measured time reduction. Hold rank reduction for a later online
ablation because it falls below the 70% far-gain acceptance boundary.

## v35 controlled tall-sweep restoration

The confirmed v34 result is `17456 / 183s`. Its 66-second runtime recovery is
useful, but the 226-point loss proves that local one-sweep retention is not an
online score proxy.

```text
v33: joint search on all ordinary shapes, 3 short-row sweeps
  └─ v34: ordinary wide -> baseline; ordinary tall -> 1 sweep
      └─ v35: ordinary wide -> baseline; ordinary tall -> 3 sweeps
```

Acceptance criteria:

1. v35 must be functionally identical to v33 for tall, square, and forced
   strong-compression shapes.
2. v35 must remain functionally identical to v30 for ordinary wide shapes.
3. Attention must remain unchanged.
4. Full output/state validation must pass `1100/1100` before push.
5. Forecast conservatively between v34 and v33: `17580-17680 / 205-235s`.

Humanize remains unavailable; the controlled one-variable restoration and
acceptance criteria are refined manually.

Outcome:

1. Far-holdout groups 0-2 are case-for-case identical to v33 (`999.378937`
   Linear points for both), covering the retained tall/square branches.
2. Far-holdout ordinary-wide group 3 is case-for-case identical to v30
   (`274.211533`), so the v34 efficiency gate remains intact.
3. Random strong-compression group 8 is case-for-case identical to v33
   (`-1323.261058`), confirming that forced global fitting bypasses the gate.
4. Attention group 0 is identical in every comparison. Full validation passes
   `1100/1100`; `py_compile` also passes.
5. Candidate decision: submit v35 as the controlled tall-sweep restoration.
   Keep the forecast at `17580-17680 / 205-235s`; this version tests whether
   v34's lost online score came from the tall sweep cut rather than the wide
   gate.

## v36 low-cost wide-joint restoration

The confirmed v35 result is `17501 / 212s`, only `+45 / +29s` over v34. Since
v35 and v34 differ only in ordinary-tall sweep count, the extra two tall
sweeps are positive but too expensive. Conversely, v33 and v35 differ only in
the ordinary-wide joint branch, whose observed upper-bound contribution is
`+181 / +37s`. A one-sweep probe selected the baseline on both local wide
groups and recovered none of v33's gain, so v36 tests two sweeps instead of
discarding the branch.

```text
v33: tall=3, square=3, ordinary-wide=3
  ├─ v34: tall=1, square=3, ordinary-wide=off
  │   └─ v35: tall=3, square=3, ordinary-wide=off
  └─ v36: tall=1, square=3, ordinary-wide=2
```

Risks and acceptance criteria:

1. Reduced-sweep wide fitting can select a different Q(A) policy and may retain
   less hidden score than its local calibration error suggests. Require
   non-regression versus v30 on both nearest and far donor wide groups.
2. v36 must be identical to v34 on ordinary-tall, square, and forced
   strong-compression groups; only ordinary-wide behavior may change.
3. The wide branch must recover a material fraction of v33's local wide gain
   at lower measured quantization time than v33.
4. Attention remains unchanged and full validation must pass `1100/1100`.
5. Online forecast will use only the controlled v33/v35 wide delta and the
   measured local one/full-sweep retention, not the obsolete cross-version
   extrapolation coefficient.

The one-sweep probe matched v30 exactly on near wide groups 3/7 while costing
more, so that leaf is rejected before full validation. Humanize is unavailable,
so the candidate isolation, risks, and executable acceptance criteria are
refined manually as required by the project edict.

Outcome:

1. One wide sweep is rejected: it matches v30 exactly on near groups 3/7 and
   pays projection overhead without score gain.
2. Two wide sweeps retain `80.6%` of v33's near wide gain (`+2.476/+3.070`)
   and `88.4%` of its far wide gain (`+2.971/+3.362`), with no group-level
   regression versus v30.
3. Far groups 0-2 are case-for-case identical to v34 (`995.400361`), and
   forced-compression random group 8 is also identical (`-1323.261058`).
   Attention remains unchanged.
4. Local timing is order-sensitive: the far panel is about 2 seconds faster
   than v33, while the near panel is noisy and slower. Algorithmically one
   coordinate sweep is removed, but online runtime is forecast from the
   v33/v34/v35 controlled observations rather than these absolute timings.
5. Full validation passes `1100/1100`. Select the two-wide-sweep v36 leaf and
   forecast `17570-17650 / 205-230s`, centered near `17610 / 215s`.

## v37 QR range-basis projection

The confirmed v36 result is `17638 / 236s`: the score forecast is accurate,
while runtime is 6 seconds above the interval. Relative to v35 it gains 137
points for 24 seconds. Two wide sweeps therefore recover most of the hidden
wide benefit; adding a third sweep or restoring tall sweeps would mostly
reconstruct v33 (`17682 / 249s`) rather than open a new score direction.

The wide global sketch currently runs `pca_lowrank` even when its requested
rank equals the number of output rows. In that full-row-rank case, an identity
output basis preserves every squared output norm and cross product exactly:

```text
||AW.T - Q(A)Q(W).T||²
  = ||(AW.T - Q(A)Q(W).T) U||²,  U U.T = I
```

Candidate DAG:

```text
v36 (wide=2, tall=1, square/forced=3)
  ├─ v37-a: full-row-rank identity projection, same objective
  └─ v37-b: independent Q(W) feedback parameter probe
```

Risks and acceptance criteria:

1. Floating-point evaluation in a different orthogonal basis may change ties.
   Require policy and case-score equivalence on near/far wide panels and the
   forced-compression random panel; any material regression rejects v37-a.
2. Measure calibration time in interleaved/repeated order because earlier
   absolute local timing was thermally noisy. Require a clear median speedup.
3. Probe one bounded Q(W) parameter family only. It must improve both near and
   far non-wide panels without violating the v36 branch equivalence; otherwise
   reject it and submit the exact acceleration alone.
4. Do not increase sweep count, rank, dynamic state size, or online work.
5. Attention remains unchanged and full validation must pass `1100/1100`.

Humanize remains unavailable. The algebraic invariant, controlled probe, and
acceptance criteria are refined manually under the project edict.

Outcome:

1. The full-row-rank identity branch preserves scores exactly on random wide
   groups 2/4/5 and reduces their panel time from `6.77s` to `5.41s` before
   Attention. The rank-truncated QR branch is not numerically equivalent to
   `pca_lowrank`; its sequential block decisions are materially better, so
   v37-a is modified from pure acceleration into a new precision direction.
2. On near wide groups 3/7, QR improves v36 by `+53.299` points; on the far
   donors it improves by `+58.380`. A second independent pair, groups 11/15,
   improves by `+49.525` near and `+51.976` far. Every case family is positive.
3. Forced-compression random groups 8/9 improve by `+59.621`. Non-wide far
   groups 0-2 and Attention remain case-for-case identical to v36.
4. Row chunk 64 is output-identical to chunk 32 and reduces the near wide
   panel by about `1.19s`; select it. End-to-end QR timing is mixed across
   thermal order, so forecast online runtime conservatively at `225-260s`.
5. The Q(W) feedback trust/window/ridge family is not reopened: v25-v28 already
   tested it across four raw/donor gates and fixed the current parameters.
   Combining it now would confound the new subspace result.
6. Full validation passes `1100/1100`. A direct per-case transport gives a
   lower scenario near `17950-18200`; the v36 wide online sensitivity permits
   a much higher but unreliable scenario. Use `17950-19500` as the official
   v37 score interval and require online feedback before extrapolating further.

## v38 robust fixed-Q(W) fallback selection

The confirmed v37 result is exactly `17638`, equal to v36, while runtime falls
from `236s` to `224s`. The QR/identity implementation is retained as a useful
12-second acceleration, but its `+50`-scale local wide gains have zero hidden
score value. This is direct evidence that further Q(A) subspace tuning on the
current donors is not a valid next direction.

v31 already tested a full second Q(W) feedback solve and scored `-28 / +28s`
versus v30. v38 must not repeat that rejected branch. Instead it cheaply forms
a direct quantization candidate alongside the existing sequential-feedback
Q(W), then uses stratified calibration A@W evidence to replace an output row
only when the direct candidate wins consistently across layouts.

```text
v37 (fast QR, fixed v28 feedback Q(W))
  ├─ rejected: ordinary-shape second feedback Q(W) solve (v31 evidence)
  └─ v38: existing feedback Q(W) vs direct Q(W)
      ├─ stratified rows from all five calibration layouts
      ├─ per-output-row exact A@W comparison
      └─ conservative cross-layout win/margin gate
```

Risks and acceptance criteria:

1. Per-row calibration selection can overfit. Require non-regression on both
   nearest and far donor panels, not merely raw calibration output.
2. The direct candidate is only a fallback; never replace a feedback row on a
   single aggregate win. Require a majority of layouts and a bounded worst
   ratio with a positive margin.
3. Do not add another residual solve, global sweep, or dynamic test-time work.
   The 224-second anchor leaves 76 seconds, but target `<=260s` forecast.
4. If the selector chooses no rows or donor gain is negligible, reject it and
   search a different fixed-Q(W) mechanism rather than submit a no-op version.
5. Attention remains fixed and full validation must pass `1100/1100`.

Humanize remains unavailable; the historical rejection, conservative gate,
and executable criteria are refined manually under the project edict.

Outcome:

1. The direct/feedback Q(W) fallback is rejected. Near groups 0-3 change by
   only `-0.006` points while quantization time rises by about 11 seconds; a
   50-group projection would exceed the online budget for no accuracy value.
2. A cheaper one-step, two-row-per-layout mantissa residual refinement is also
   rejected. The same groups regress by `0.089` and cost about 2.5 seconds.
   Current feedback Q(W) is locally stable; neither candidate is submitted.
3. Final v38 uses the evidence-backed fallback leaf: retain v37 QR/chunk64,
   restore three sweeps for ordinary tall shapes, keep ordinary wide at two,
   and leave square/forced shapes at three. This changes one online-measured
   variable without reopening v31's rejected second-Q(W) path.
4. On far tall groups 0/1, v38 improves v37 from `634.546104` to `638.524680`
   with no case-family regression. Wide, square, forced, and Attention paths
   remain v37 behavior.
5. Full validation passes `1100/1100`. The controlled v34-v35 anchor predicts
   about `+45 / +29s` over v37, so forecast v38 at
   `17665-17720 / 240-260s`, centered near `17683 / 253s`.

## v39 tall-1024 multi-sweep extension

The confirmed v38 result is `17682 / 228s`: score exactly matches v33 while
runtime improves by 21 seconds. Relative to v37, restoring tall short-row
sweeps adds exactly 44 points for only 4 seconds online. This independently
confirms that the tall coordinate-search direction transfers, while v37's
wide subspace score changes did not.

The current quantizer still forces every 1024-row activation to one sweep,
even on ordinary-tall shapes. v39 tests three sweeps for that single tall
layout while preserving every other v38 branch.

```text
v38: tall rows<=512 -> 3; tall rows=1024 -> 1
  ├─ v39-a: tall rows<=1024 -> 3
  └─ held: tall rows<=512 -> 5 (do not mix variables yet)
```

Risks and acceptance criteria:

1. The 1024-row case has equal score weight but higher compute. Require a
   measurable tall donor gain and forecast below `285s`, leaving 15 seconds
   of online safety margin.
2. Store the row threshold in calibration state and pass it explicitly to the
   dynamic quantizer. Ordinary wide, square, forced-compression, and Attention
   must remain case-for-case identical to v38.
3. Require the changed 1024-row cases to be non-regressing on both nearest and
   far tall donor panels; aggregate-only improvement is insufficient.
4. Do not increase rank, candidate offsets, or sweep count in the same version.
5. Full state/output validation must pass `1100/1100` before push.

Humanize remains unavailable; the single-variable extension and executable
acceptance criteria are refined manually under the project edict.

Outcome:

1. Near tall groups 0/1 improve by `+0.977005`; far donors improve by
   `+0.980894`. Both structural signatures are positive and only test case 3,
   the 1024-row layout, changes; the other eight tall cases are exact matches.
2. The two-group quantization cost rises by `2.55s` near and `2.94s` far.
   Calibrating this cost with v37-v38's measured local/online tall ratio gives
   an online center near 238 seconds, well below the 285-second gate.
3. Far square group 2, ordinary-wide group 3, forced-compression random group
   8, and Attention are case-for-case identical to v38.
4. Full state/output validation passes `1100/1100`; `py_compile` and
   `git diff --check` pass.
5. The v37-v38 mechanism-specific score ratio maps the new tall gain to about
   11 online points. Select v39-a and forecast
   `17688-17708 / 235-250s`, centered near `17693 / 238s`.

## v40 literature-driven layout optimization

The confirmed v39 result is `17697 / 234s`, only `+15 / +6s` over v38. This
matches the conservative forecast but confirms that additional coordinate
sweeps are near their cost-benefit limit. v40 therefore starts a new mechanism
rather than tuning another sweep count or lambda.

A full-text review of sixteen primary low-precision papers is recorded in
`docs/quantization-paper-synthesis-v40.md`. SmoothQuant/AWQ-style reciprocal
scaling, GPTQ/AdaRound-style output-aware rounding, and QuaRot-style local
Hadamard transforms already have close counterparts in v39. RPTQ's channel
reordering across quantization-group boundaries is the strongest missing,
interface-compatible mechanism.

```text
v39 (original contiguous channel groups)
  └─ v40: paired A/W permutation before balance + signed Hadamard
       ├─ spread sensitivity quantiles across each 64-channel group
       ├─ preserve A@W exactly with the same permutation
       ├─ store one permutation for dynamic A
       └─ retain identity beyond the online 2:1 compression boundary
```

Risks and acceptance criteria:

1. A layout that helps activation quantization can hurt weight quantization.
   Rank candidate statistics using both A and W, then require nearest/far
   donor agreement; do not choose parameters from donor tests.
2. Validate `(A[:, p] @ W[:, p].T) == A @ W.T` to FP32 tolerance before
   measuring quantized output.
3. Keep the online delta to one `index_select`; no dense affine matrix, online
   search, extra Q(W) solve, or additional coordinate sweep is allowed in the
   same version.
4. Attention stays identical to v39. Full state/output validation must pass
   `1100/1100`, and projected online time must stay below 270 seconds.
5. If the new layout is not positive on both donor panels, reject it and use
   the SpinQuant-inspired learned block-orthogonal basis as the next branch;
   do not submit a paper-inspired change without local evidence.

Humanize remains unavailable; the review-derived steps and executable
acceptance criteria were refined manually under the project edict.

Outcome:

1. Direct RPTQ clustering is rejected: grouping similar-range channels raises
   the simple calibration-output proxy by 30%-57% on all four online shapes.
   This is a real interaction with Hadamard-64, which needs large and small
   channels mixed rather than concentrated.
2. The retained method sorts channels by joint A/W RMS sensitivity, then uses
   a snake round-robin layout so every block receives the full sensitivity
   spectrum. Paired A/W permutation has FP32 relative error `1.05e-6`, caused
   only by changed accumulation order.
3. Nearest donor groups 0-3 improve `1707.044071 -> 1705.119469`; far donors
   improve `1707.377129 -> 1703.783798`. Independent groups 4-7 improve
   `1780.493763 -> 1762.529987` nearest and
   `1766.604370 -> 1751.180107` farthest.
4. Random 4:1 compression regresses before gating, so v40 applies the layout
   only when `input <= 2 * output`; stronger compression becomes exactly v39.
   Across the complete standard random suite, Linear score improves
   `-29716.980 -> -22204.281` (`+7512.698`), dominated by recovery of severe
   tail layouts. Attention remains exactly `5412.720`.
5. The complete random run is `246.51s` versus v39 `190.61s`, but the delta is
   concentrated in extreme distributions that select different joint
   policies. Online-shaped donor panels are flat to slightly faster. Forecast
   v40 conservatively at `17720-18150 / 240-265s`; do not map the random tail
   gain linearly to the hidden score.
6. Full validation passes `1100/1100`; `py_compile` and `git diff --check`
   pass. Select the shape-gated balanced-permutation branch as v40.

## v41 discrete learned orthogonal basis

The confirmed v40 result is `17808 / 243s`, versus v39 `17697 / 234s`:
`+111 / +9s`. This validates cross-group layout as an online-positive new
mechanism, while the complete-random `+7512.698` tail gain clearly cannot be
transported linearly to hidden score. Future forecasts use the ordinary
near/far donor panels and the controlled v39-v40 online delta; random extremes
are robustness gates only.

SpinQuant shows that quantized accuracy varies substantially across equivalent
rotations. A dense learned rotation is too expensive with only 57 seconds of
online margin. v41 first tests its discrete signed-Hadamard subset: expand the
fixed calibration candidate bank while leaving dynamic execution and state
size unchanged.

```text
v40 (balanced channel layout + 8 signed-Hadamard bases)
  ├─ v41-a: 16 deterministic signed-Hadamard bases
  ├─ v41-b: 32 bases only if 16 has strong gain/cost evidence
  ├─ held: sparse two-stage butterfly/Givens learning
  └─ reject: dense 64x64 online rotation
```

Risks and acceptance criteria:

1. More calibration candidates can overfit peak statistics. Require aggregate
   improvement on both nearest and far donors and check the second group batch
   without selecting candidate count from its test errors.
2. Candidate generation remains deterministic and calibration-only. It may
   use A/W statistics, but dynamic activation never observes W or test labels.
3. Dynamic transform count, activation-state tensor count, and coordinate
   sweep schedule must remain identical to v40. Only calibration candidate
   scoring may grow.
4. Require measured ordinary-panel calibration cost below 1.12x v40 and
   forecast online below 275 seconds. Reject 32 bases unless 16 bases earns a
   material cross-panel gain.
5. Attention remains byte-identical. Any accepted version must pass the full
   random robustness suite and `1100/1100` before push.

Humanize remains unavailable; the one-variable candidate tree and executable
acceptance criteria were refined manually under the project edict.

Outcome:

1. Unconditional 16-base selection improves the nearest panel but regresses
   far donors `1703.783798 -> 1709.728313`; reject it as calibration overfit.
   A 1% challenger trust gate returns exactly to v40 scores while retaining
   its candidate-scoring cost, so that leaf is also rejected.
2. Retain eight signed-Hadamard bases and apply a six-bit reversal to the 64
   sensitivity lanes inside each balanced group. This is fused into v40's
   stored channel permutation and changes neither dynamic operator count nor
   state size. It aligns Walsh basis order with HiF4's nested 4/8/16 scales.
3. Groups 0-3 improve `1705.119469 -> 1615.713978` nearest and
   `1703.783798 -> 1618.191845` farthest. Independent groups 4-7 improve
   `1762.529987 -> 1662.879684` and `1751.180107 -> 1663.641296`.
   The four gains are `89.405/85.592/99.650/87.539`, unusually consistent.
4. Complete standard random Linear improves `-22204.281 -> -21039.758`
   (`+1164.523`), with wins/ties/losses `21/14/8`. The isolated linear run
   falls from v40's `204.13s` to `169.66s`, although cross-run timing includes
   policy and thermal noise. Strong-compression fallbacks remain exact.
5. v40 converted `38.906` summed four-panel MSE improvement into 111 online
   points. v41's corresponding `362.186` is 9.3x larger, but historical
   transport is nonlinear. Forecast conservatively at
   `18200-18900 / 240-255s`, rather than extrapolating a 1000-point center as
   certainty.
6. Attention remains byte-identical to v40. Select the bit-reversed hierarchy
   layout for v41 subject to `1100/1100` and final repository checks.

## Permanent online-exploration rule after v41

The confirmed v41 result is `17930 / 217s`, versus v40 `17808 / 243s`:
`+122 / -26s`. The user explicitly clarified that local and online datasets
are different, so a locally neutral or negative mechanism can still be useful
online. From v42 onward:

1. Local evidence ranks candidates and detects catastrophic risk; it is not a
   universal veto.
2. A mathematically sound, interface-legal mechanism with bounded runtime may
   be submitted as a controlled online ablation even when local gains are weak
   or mixed.
3. Such exploration must change one attributable mechanism, preserve a known
   fallback, and state the online hypothesis before feedback.
4. Runtime safety, invalid output, numerical failure, or a broad catastrophic
   local regression remain rejection conditions.
5. Online feedback is the final generalization evidence and must update the
   candidate DAG rather than being retrofitted to the local narrative.

## v42 Q/K hierarchical orthogonal layout

v41 shows that the order of an equivalent Hadamard basis relative to HiF4's
nested 4/8/64 scale hierarchy matters. Attention Q and K currently use the
same reciprocal balance and head-wise Hadamard but retain natural hierarchy
order. v42 applies the same 64-lane bit reversal independently inside every
Q/K head block while leaving V unchanged:

```text
Q' = ((Q * C) P) H
K' = ((K / C) P) H
Q' K'^T = (Q * C) P H H^T P^T (K / C)^T = Q K^T
```

Candidate DAG and acceptance criteria:

```text
v41 (Linear hierarchy aligned; Attention natural order)
  └─ v42: shared Q/K bit-reversal inside each 64-lane block
       ├─ preserve QK scores exactly before quantization
       ├─ V and Linear remain v41
       └─ one fixed index operation in Q and K dynamic paths
```

1. Verify full-precision QK equivalence numerically for every observed
   head-dimension family.
2. Measure local standard Attention on independent donor and random layouts.
   Mixed or neutral local results do not automatically reject this controlled
   online hypothesis; a broad catastrophic regression does.
3. Require projected online runtime below 250 seconds from the 217-second v41
   anchor, leaving at least 50 seconds to the contest limit.
4. Do not change Attention balance, scale offsets, energy penalty, V, or any
   Linear behavior in the same version.
5. Pass complete random robustness and `1100/1100` before submission.

Humanize remains unavailable; the online-exploration boundary and executable
criteria were refined manually under the project edict.

Outcome:

1. QK full-precision equivalence passes on the observed `(16, 2, 256)` head
   signature: relative L2 error `4.69e-8`, maximum absolute error `6.10e-5`.
2. Unconditional Q/K hierarchy layout improves four donor panels: groups 0-4
   nearest/farthest improve `0.588/0.308` MSE and groups 5-9 improve
   `0.333/0.111`. Timings are flat in paired runs.
3. The unconditional complete-random Attention score improves
   `5412.720 -> 5547.791`, but one out-of-domain group regresses by 776 points.
   Its calibration near-zero ratio is about 0.41, while online-like donors are
   tightly concentrated at 0.26-0.29.
4. Add a calibration-distribution trust gate using the existing near-zero
   statistic: apply the fixed bit reversal at ratios `<=0.35`, otherwise store
   identity and exactly reproduce v41. No test tensor or test error is used by
   this decision.
5. The gated complete-random score is `5457.780` (`+45.060` over v41), with
   the catastrophic group restored exactly. Linear weight/activation and
   Attention V outputs are parameter-for-parameter identical to v41.
6. This is an Attention-only controlled online ablation. Forecast
   `17940-18020 / 218-228s`; retain or widen the gate only from online
   feedback, not by forcing the local aggregate higher.
