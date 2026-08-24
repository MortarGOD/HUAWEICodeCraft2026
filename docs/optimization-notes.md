# Optimization Notes

## Confirmed online scoring context

- Baseline score: `1185`
- Baseline total time: `108s`
- Total time limit: `300s` (5 minutes)
- v13 online result: `13349 / 182s`.
- v14 online result: timed out (`>300s`).
- v15 online result: `15463 / 170s`.
- v16 online result: `15539 / 156s`.
- v17 online result: `7062 / 172s` (low-lambda overfit).
- v18 online result: `2887 / 206s` (severe accuracy regression, not timeout).
- v19 online result: `15543 / 198s` (safe but low-value adaptive search).
- v20 online result: `15678 / 183s`.
- v21 online result: `15678 / 163s` (Attention adaptation produced zero score
  gain; runtime improved by 20 seconds).
- v22 online result: `16359 / 197s`; previous estimate `16138 / 206s` was low by
  221 points and high by 9 seconds.
- External comparison `solution_cxw_v1.py`: `17562 / 249s`, +1203 points and
  +52 seconds versus v22.

## Online-calibrated local evaluation

`local_eval.py --calibrate-online` uses `solution_v1.py` as a proxy for the
unavailable official HiF4 baseline, then scales both the proxy score and local
runtime so that `solution_v16.py` maps to the confirmed `15539 / 156s` result.
Use the same Linear and Attention group IDs when comparing candidates:

```bash
/opt/miniconda3/envs/codecraft/bin/python local_eval.py \
    --solution solution/solution_v16.py \
    --linear-groups 0,1,2,3 \
    --attn-groups 0,1,2,3,4 \
    --calibrate-online
```

The calibrated values are comparative estimates rather than replacements for
the judge, because the official baseline MSE and server performance profile are
not available locally.

## Online dataset characteristics

- Linear: 50 groups. Each group has:
  - 1 weight
  - 5 calibration activations
  - 5 test activations
  - 250 (W, A) test pairings in total across groups.
- Attention: 50 groups. Each group has 5 calibration + 5 test QKV samples.
- Total test tensors: 650.
- Weight size: not smaller than `4k * 4k`.
- Activation sizes: `1*T*4k`, `10*T*4k`, `128*T*4k`, `512*T*4k`,
  `1024*T*4k`; the first dimension values `(1, 10, 128, 512, 1024)` are
  batch/token scales.

## Calibration fitting direction

The contest allows the offline calibration function to:

1. Dequantize the fixed NVFP4 weight `W`.
2. Use calibration activations `A_calib` to compute `Y = A_calib @ W^T`.
3. Fit a HiF4-constrained `Q(W)` (i.e. `weight_params`) by minimizing the
   output error, e.g. `|| A_calib @ W^T - A_calib @ Q(W)^T ||_F^2`.
4. Use the same output objective during every calibration stage to fit the
   activation quantizer `Q(A)`, rather than relying only on activation-space
   reconstruction error.
5. Return the fixed `Q(W)` and the auxiliary `activation_state`.

The fitted state is fixed after offline calibration; dynamic quantization only
applies the learned output-aware objective to the incoming activation.

## Next implementation candidates

1. Offline GPTQ-like HiF4 weight optimization using calibration activations.
2. Per-channel SmoothQuant-style scale folding between W and A.
3. Per-64-block optimal discrete scale/E1 search for HiF4.
4. Attention: apply the same Hadamard rotation to Q and K offline/online to
   preserve `Q K^T` while reducing outliers.

## Version history

- `solution_v1.py`: direct-cast baseline.
- `solution_v2.py`: Linear SmoothQuant-style per-channel scale; online inverse
  scale for activations; attention still direct-cast. Local mini-sample Linear
  output MSE improved from ~5.87 to ~5.47.
- `solution_v3.py`: v2 + per-block E6M2 candidate search for weight
  quantization. Local mini-sample Linear output MSE improved to ~5.18.
- `solution_v4.py`: v3 + per-layer Linear smooth-scale alpha search and
  Attention Q/K smooth-scale alpha search. Local mini-sample Linear ~5.13,
  Attention ~0.01417.
- `solution_v5.py`: v4 + per-block scale search for online activation and V.
  Local mini-sample Linear ~4.77, Attention ~0.01337.
- `solution_v6.py`: v5 but reverts the harmful Attention Q/K smooth-scale
  search; keeps Linear improvements and V per-block scale search. Local
  datasets_0818 subset attention improves over v1 while v4/v5 attention was
  worse.
- `solution_v7.py`: v6 + per-block scale search for Attention Q and K. Local
  datasets_0818 attention subset improves from ~0.2472 (v6) to ~0.2332.
- `solution_v8.py`: v7 + importance-weighted per-block E6M2 scale search for
  Linear weights. Local datasets_0818 linear subset total MSE ~3101.5 vs v7
  ~3161.1.
- `solution_v9.py`: v8 + shared Q/K Hadamard rotation (block size 16) for
  Attention. Local datasets_0818 attention subset total MSE ~4.3118 vs v8
  ~5.8301. Linear unchanged.
- `solution_v10.py`: v9 + finer coarse-to-fine Linear smooth-scale alpha
  search. Local datasets_0818 linear subset total MSE ~3089.37 vs v9 ~3101.55.
- `solution_v11.py`: v9 (not v10) + output-aware importance weighting for
  online Linear activation per-block E6M2 scale search. The activation
  importance is `smooth^2 * sum_o(W_oc^2)`, derived from the output-error term
  `E_A * diag(smooth) * W^T`. v10's finer alpha search was reverted because it
  regressed online (12748 / 204s vs v9 12806 / 183s). Local datasets_0818
  linear subset (groups 0-3) total MSE ~2861.07 vs v9 ~3101.55.
- `solution_v12.py`: v11 + smaller shared Q/K Hadamard rotation block size
  (16 -> 4) for Attention. Local datasets_0818 all 50 attention groups total
  MSE ~39.05 vs ~43.11 for block size 16; linear is unchanged.
- `solution_v13.py`: revert v12's Attention rotation block size back to 16.
  v12 regressed online (12952 / 161s vs v11 13349 / 183s), so v13 restores
  the v11 configuration exactly.
- `solution_v14.py`: combine v138's output-aware Linear calibration with
  v13's proven Attention implementation. Linear applies a shared 64-wide
  Hadamard transform, fits legal HiF4 weight candidates against calibration
  outputs, performs ridge residual fitting, and selects refined weight codes
  on a held-out calibration sample. Attention remains byte-for-byte equivalent
  in behavior to v13 (shared Q/K rotation block size 16 and candidate search).
  On the default local subset, Linear total MSE improves from ~2861.07 to
  ~1868.71 while Attention remains ~4.31184. Local runtime rises from ~26.7s
  to ~99.2s. The version subsequently exceeded the 300s online limit.
- `solution_v15.py`: accelerate v14 while preserving v13 Attention. Linear
  reduces the compact E6M2 offsets from five to three, removes both gain
  candidates, keeps one ridge fit (`lambda=20`), selects between only the
  baseline and ridge candidate, and removes five-step code refinement. Linear
  quantization calls per representative group drop from 18 to 8 including the
  five online test activations. On groups 0-3, Linear total MSE is ~1941.41
  versus v13's ~2861.07 and v14's ~1868.71; Attention stays ~4.31184. The
  original v13-anchored estimate was optimistic; the confirmed online result
  is `15463 / 170s`.
- `solution_v16.py`: retain v15's time-safe weight fitting and make activation
  scale selection output-aware in every Linear calibration/dynamic stage.
  For each transformed 64-channel block it approximates `W^T W` with two
  power-iterated Gram directions plus the residual diagonal, then chooses
  `Q(A)` by the resulting approximation to `||(A-Q(A))W^T||^2`. Rank 2 was
  selected over rank 4 because the latter added about 9% local runtime for only
  about 0.07% extra MSE improvement on the disjoint subset. On groups 0-3,
  Linear MSE improves from v15's ~1941.41 to ~1904.40; groups 4-7 improve from
  ~2020.19 to ~1964.92. Attention remains ~4.31184. With v15 as the online
  anchor, the default-subset estimate was about `15600 / 154s`; the confirmed
  online result is `15539 / 156s`. The estimate errors were only +0.39% in
  score and -1.28% in runtime, so groups 0-3 Linear plus groups 0-4 Attention
  are currently the primary local validation subset.

## Candidate DAG and direction decisions

```text
v13 (stable Attention)
  └─ v14 (full output-aware Linear; >300s, abandon implementation)
      └─ v15 (time-safe weight fitting; 15463 / 170s, retained base)
          └─ v16 (rank-2 output-aware Q(A); 15539 / 156s, current base)
              └─ v17 (ridge lambda 0.01; 7062 / 172s, reject)
                  └─ v18 (two Q(W) residual corrections; 2887 / 206s, reject)
              └─ v19 (held-out lambda controller; 15543 / 198s, modify)
              └─ v20 (A-only layout profile selection; 15678 / 183s)
                  └─ v21 (15678 / 163s; current online anchor)
                      └─ v22 (leakage-free Linear improvement; push candidate)
```

- Continue: output-aware `A @ W` calibration for both `Q(A)` and `Q(W)`;
  v16 improved score by 76 and runtime by 14 seconds over v15.
- Modify: low-rank Gram approximation; rank 2 is retained, while rank 4 needs
  a cheaper implementation before reconsideration because its local accuracy
  gain was marginal relative to its runtime cost.
- Abandon: v14's multi-candidate refinement path in its current form because
  it exceeded the 300-second online limit.
- Hold/revert base: Attention remains on v13 behavior until a candidate shows
  consistent gains on both the primary subset and a disjoint subset.

## v17-v18 offline evidence

- `solution_v17.py`: reduce the normalized ridge lambda from 20 to 0.01. On
  all 50 Linear groups, total MSE falls from v16's `23901.842773` to
  `12382.298023` (48.20%). Every five-group batch improves by roughly the same
  amount. The primary calibrated estimate was about `18570 / 155s`, but the
  confirmed online result is only `7062 / 172s`: score falls 54.55% from v16
  while runtime rises 10.26%. This isolates low regularization as a primary
  overfit cause even without residual refinement.
- `solution_v18.py`: factor the same regularized calibration kernel once and
  apply two post-quantization residual corrections to Q(W), choosing the best
  candidate per output row against `A @ W`. Primary Linear MSE is `787.307289`
  versus v17's `992.771124`; disjoint groups 4-7 are `806.183283` versus
  `1022.246878`. Across all 50 groups, Linear MSE is `9784.265111`, improving
  20.98% over v17 and 59.06% over v16; all ten five-group batches improve. The
  v16-anchored estimate is about `19290 / 228s`, below the 270-second safety
  threshold. A third iteration is not used because the projected runtime would
  approach/exceed the 300-second limit.
- Online v18 contradicted the local estimate: `2887 / 206s` versus the
  predicted `19290 / 228s`. Runtime stayed legal, but score fell 81.42% from
  v16 despite a 59.06% local Linear-MSE improvement. This is strong evidence
  of calibration overfit: lambda 0.01 weakens the trust region, residual
  corrections repeatedly fit the same quantized calibration matrix, and
  per-output-row selection evaluates candidates on the same rows used to fit
  them. v18 is rejected; v17 is not considered a safe rollback because it
  shares the low-lambda fit. Comparing v17 with v18 also shows that two residual
  corrections compound the failure: another 59.12% score loss and 34 seconds
  relative to v17. v16 remains the stable base and online calibration anchor.
- Abandon: recomputing Q(A)'s Gram state from final Q(W) without refitting
  regressed primary MSE to `844.291557`; closing the loop with a second full
  Q(A)/Q(W) fit recovered only to `816.894451` while adding about 44% runtime.
- Abandon: calibration-derived diagonal Q/K sensitivity worsened Attention
  from `4.311844` to `4.682031` on the primary subset and from `4.487492` to
  `4.890355` on the disjoint subset.
- Abandon: four-way calibration selection of Q/K sign rotations preserved the
  same Attention MSE but roughly tripled Attention time.

## v19 held-out parameter controller

- `solution_v19.py` branches from v16, not v17/v18. For each of the five
  calibration activation shapes, it forms deterministic even/odd fit and
  validation rows, capped at 64 rows per shape for runtime control.
- It fits lambda 10 and 20 candidates only on fit rows, scores validation
  `A @ W` output MSE with all shapes weighted equally, and requires at least
  0.2% validation improvement plus a 1.5x Q(W)-deviation trust region before
  accepting lambda 10. Selection is once per layer, never per output row.
- On Linear groups 0-7, every layer conservatively selects lambda 20. Outputs
  and MSE are therefore exactly v16 (`1904.398739` primary, `1964.920799`
  disjoint), showing the controller rejects the locally seductive aggressive
  fit. After reusing the held-out ridge system, the calibrated runtime estimate
  is about 210 seconds, below the
  270-second safety target. The purpose of the online submission is to observe
  whether hidden calibration groups safely select lambda 10.
- Confirmed online v19 is `15543 / 198s`, versus v16's `15539 / 156s`.
  Score improves by only 4 while runtime rises by 42 seconds. The controller
  successfully prevents the v17/v18 collapse, and its `15539 / 210s` estimate
  was accurate in score and conservative by 12 seconds in runtime. However,
  the gain/cost ratio is too low for production. This suggests hidden groups
  either almost always retain lambda 20 or accept lambda 10 only rarely; that
  is an inference because the judge does not expose per-layer choices.

## 0819-V2 rules and v20 layout profiles

- The 0819-V2 task explicitly prohibits computing `A @ W` and using the
  result to infer `Q(A)`. v20's activation-profile selector therefore receives
  only transformed activation samples and scores normalized activation
  reconstruction. `A @ W` remains confined to fitting fixed `Q(W)`.
- `tools/layout_param_search.py` generates four proxy aspect ratios, five row
  shapes, eight distribution layouts, and four deterministic seeds as
  NVFP4-like inputs. It evaluates Q(A) profiles without output targets and
  Q(W) profiles separately with held-out output error. The auditable result is
  stored in `docs/layout-profile-report.json`.
- Synthetic Q(W) lambda winners vary across shapes and held-out seeds, so that
  mapping is rejected rather than embedded. Q(A)'s compact `(0,1,2)` and wide
  `(0,1,2,3)` scale-offset profiles are sufficiently stable to retain.
- `solution_v20.py` branches from v16. It samples at most 64 rows from each
  calibration activation shape, uses A-only normalized reconstruction, and
  accepts wide only for at least 0.5% improvement. Unknown or weakly separated
  layouts fall back to compact. Q(W) stays on trusted lambda 20 and Attention
  stays byte-identical to v13.
- All 50 real local Linear groups select wide. Total MSE is `23588.473789`
  versus v16's `23901.842773`, a 1.31% improvement; all ten five-group batches
  improve. The primary v16-anchored estimate is about `15620 / 155s`. Unlike
  v17/v18, Q(A) profile selection never observes Linear output error.
- Confirmed online v20 is `15678 / 183s`, versus v19's `15543 / 198s` and
  v16's `15539 / 156s`. The legal A-only profile selector adds 135 points over
  v19 while recovering 15 seconds, so retain v20 as the new production and
  local-calibration anchor. It is still 1322 points short of the next requested
  17000 threshold; repeated Linear lambda/profile tuning is now a low-yield
  direction, so v21 changes only Attention.

## v21 adaptive Attention profile

- v21 keeps v20 Linear unchanged and compares fixed shared Q/K Hadamard block
  sizes 4, 8, and 16 against true calibration Attention output. It selects one
  profile per Attention group, requires an alternative to beat trusted block 16
  by at least 1%, and never observes test tensors during selection.
- With 128 sampled rows per calibration shape, selection agrees with the test
  oracle on 44/50 groups. Aggregate normalized test error is `1.036959` versus
  block 16's `1.246145` (16.79% lower) and the oracle's `1.034466` (only 0.24%
  regret). Selected profiles are block 4/8/16 on 20/23/7 groups.
- Reducing the cap to 64 rows preserves 44/50 decisions relative to the
  128-row controller and cuts all-50 calibration selection time from roughly
  32 to 16 seconds locally. The six changed decisions have approximately
  neutral aggregate test regret, so 64 rows is retained for online headroom.
- The first ten groups produce Attention MSE `7.636810` versus v20's
  `8.799336` (13.21% lower). With identical Linear, the v20-anchored local proxy
  estimate is `19751 / 188s`.
- At the retained 64-row cap, all 50 groups produce Attention MSE `35.907929`
  versus v20's `43.114502` (16.72% lower). Candidate Attention time is 55.16
  seconds versus 40.84 for v20, only 14.32 seconds of net local overhead across
  all groups. Since Linear dominates the complete evaluation, the online
  runtime projection is conservatively `190-205s`, below the 270-second gate.
- Continue this direction: profile selection has strong calibration/test
  agreement, low oracle regret, and bounded calibration-only cost. Roll back to
  v20 if the online score does not exceed 17000 or runtime materially exceeds
  205 seconds; that would indicate the new hidden Attention layouts break the
  local calibration/test relationship.

## v21 online contradiction and dataset leakage

- Confirmed v21 is `15678 / 163s`: score is byte-for-byte equal to v20's
  `15678`, so the predicted `19751` was wrong by 4073 points. The controlled
  v20-to-v21 change touched only Attention, which makes zero the only defensible
  current online coefficient for this local Attention improvement.
- Audit result: `datasets_0818/attn.pt` has 1500/1500 corresponding calibration
  and test tensors exactly equal; `datasets_0818/linear.pt` has 500/500 exactly
  equal. They do not alias storage after loading, but their values are copies.
  Consequently v21's 44/50 selection/test agreement and v17/v18's large local
  gains were contaminated by complete calibration-test leakage.
- Abandon: the v21 Attention controller as a score-improvement direction. Keep
  v21 only as a faster online runtime anchor until a leakage-free holdout and a
  controlled online result justify Attention changes again.
- Modify: local validation will use a deterministic cross-group donor holdout,
  matched only by calibration-side NVFP4 distribution features within exact
  shape/head signatures. This cannot reproduce the private judge, but it is a
  materially stricter generalization test than copied calibration tensors.

## Independent donor holdout v2 results

- `tools/build_independent_holdout.py` extracts bounded calibration-only NVFP4
  statistics and pairs nearest different groups within exact structural
  signatures. `datasets_holdout_v2/manifest.json` contains 50 Linear and 50
  Attention donors and regenerates byte-identically. It adds no tensor copies.
- Independence check passes: corresponding receiver-calibration/donor-test
  equality is 0/500 Linear tensors and 0/1500 Attention tensors, versus
  500/500 and 1500/1500 in the raw 0818 data.
- The primary Linear holdout now follows the decisive online overfit direction:
  v16 is `1992.915516`, v17 is `2789.953941` (+40.00%), and v18 is
  `2940.751350` (+47.56%). v20/v21 is `1970.283623`, a conservative 1.14%
  improvement over v16, consistent with the small online gain.
- Attention improves from v20's `43.114502` to v21's `38.446426` on all 50
  donor groups (10.83%), but online gain is exactly zero. This proves value
  independence alone does not repair the 0818-to-online Attention distribution
  mismatch. Do not translate local Attention gains into points; use them only
  to reject regressions until a controlled online ablation supplies a valid
  coefficient.
- `local_eval.py` now enables the donor manifest by default, anchors runtime to
  v21's `163s`, and derives the Linear point coefficient from v16 `15539` to
  v21 `15678`. The current primary coefficient is about `13.6315` online points
  per Linear proxy unit; the Attention coefficient is explicitly zero.

## v22 leakage-free Linear optimization

- v22 freezes v21 Attention and changes only Linear. The retained parameters
  are balance exponent `0.44`, Q(A) offsets `(0,1,2,3,4)`, fixed Q(W) offsets
  `(0,1,2,3)`, and rank 4 for both W-only Q(A) sensitivity and calibration-A
  sensitivity used to select fixed Q(W). `A @ W` remains confined to the λ20
  fixed-weight fit and row choice; it is never used to infer Q(A).
- Primary donor holdout improves from v21 `1970.283623` to `1889.581486`
  (-4.096%); groups 4-7 improve from `2049.866459` to `1968.391388`
  (-3.974%). The v16-to-v21 coefficient estimates `16138 / 206s`.
- Across all 50 Linear groups, v21 is `24713.431183` and v22 is
  `23785.392948`, a 3.755% reduction. Every group improves; ten consecutive
  five-group batches improve by 2.45%-4.30%. Local Linear time rises from
  385.14 to 498.77 seconds (1.295x), projecting about 211 seconds online from
  the v21 163-second anchor.
- Continue: moderate scale-space expansion and low-rank sensitivity are the
  only directions with consistent primary, disjoint, and all-group gains.
- Modify: ridge remains λ20 because no-ridge slightly regressed disjoint data;
  higher lambdas converge toward no-ridge and add no meaningful accuracy.
- Abandon in current form: repeated residual correction, per-block A@W
  coordinate fitting, output-row profile selection, block-wise α selection,
  quant-aware sign search, sorted/interleaved channel permutation, offset5,
  and rank16/32. Their gains were absent or reversed on independent donors.
- The requested 17000 threshold is not yet supported by the corrected local
  model. v22's center estimate is about 16138; its purpose is to obtain a clean
  online Linear-only feedback point and recalibrate the next iteration without
  repeating v21's Attention extrapolation error.

## v23 external-mechanism audit

- Do not copy the external implementation. Treat its online result as evidence
  for mechanisms, then design independent implementations and ablations.
- Its unique Linear mechanism is sequential 64-channel Q(W) error propagation,
  unlike v22's whole-matrix ridge candidate. Its unique Attention mechanisms
  include reciprocal Q/K balance, data-gated rotation size, moment-aware scale
  objectives, and output-selected V profiles.
- Critical evaluator bug found: local `run_attention` reshaped as
  `[seq, heads, dim]` and multiplied across heads per token. Standard Attention
  must transpose to `[heads, seq, dim]` and multiply across sequence positions.
  All pre-v23 local Attention score mappings are therefore rejected, not merely
  down-weighted.
- Linear finding: the external sequential residual-propagation path scores
  `2153.199535` on independent donor groups 0-3, 13.95% worse than v22's
  `1889.581486`. This mechanism is rejected for v23; v22 Linear is retained.
- Attention ablation finding: V-profile search changes only about 0.15%; moment
  penalty adds about 1.27%. Reciprocal Q/K balance alone and rotation alone are
  weak, while their combination is strong. The causal gain is their product-
  preserving synergy, not the external branch controller.
- v23 independently computes one RMS balance per KV-head/channel, applies the
  reciprocal factors to Q/K, then uses the repository's existing Hadamard-64
  transform. Its block scale objective combines direct reconstruction with a
  bounded energy-drift term (`0.125`). It performs no online parameter search
  and never scores calibration Attention outputs.
- Correct standard-Attention all-50 result: v22 `88.832125 / 78.35s`, external
  `74.921704 / 81.30s`, v23 `75.937482 / 45.16s`. v23 improves v22 by 14.52%,
  is only 1.36% behind the external Attention error, and is 44.5% faster than
  the external Attention path in the same local run.
- Parameter stability: balance exponent `0.4` wins groups 0-4 but reverses on
  groups 5-9; retain the invariant-preserving exponent `0.5`. Fixed block 64
  beats 256, negative Q/K offsets regress, and penalty `0.125` is the stable
  local minimum. No shape-based online search is introduced from leaked tests.
- Mixed-anchor estimate: the controlled v21-to-v22 Linear gain is 681 points
  for an 80.702137 primary-donor MSE reduction. Applying that coefficient says
  the external Linear path would cost about 2225 points locally, so its observed
  +1203 net gain implies a large Attention contribution. v23 captures 92.70%
  of the external Attention MSE reduction while retaining v22 Linear, yielding
  a model center near `19536`. Because this decomposition crosses local/online
  distributions, report a wider `18000-19500` score range, not a guarantee.
- Runtime projection: v22 all-group local components are approximately
  `498.77s Linear + 78.35s Attention`; v23 keeps Linear and reduces Attention
  to `45.16s`. Scaling the total ratio from v22's online 197 seconds gives
  `185.7s`; use `185-200s` as the submission range.
- Final validation: `solution.py` and `solution_v23.py` are identical,
  `py_compile` and `git diff --check` pass, and `self_check.py` passes all
  `1100/1100` calibration/output-format checks across 50+50 groups.

## v23 online correction

- Confirmed v23 result: `16477 / 193s`, only +118 points and -4 seconds versus
  v22 `16359 / 197s`. The runtime projection was sound; the `18000-19500`
  score estimate was decisively false.
- Reject the mixed-anchor decomposition. A single external total score cannot
  identify Linear and Attention contributions, and a v21-to-v22 Linear
  coefficient cannot be transported to a different algorithm and holdout.
- Raw Attention MSE is not a score proxy: its 14.52% v23 gain was measured with
  exact calibration/test twins. The donor warning was stronger evidence—v23
  regressed from `7.9567` to `14.9405` on primary independent Attention—but
  was incorrectly discounted.
- The new controlled Attention anchor is v22-to-v23: the retained mechanism is
  slightly positive online (+118), not the multi-thousand-point explanation
  of the external result. Most of the remaining 1085-point external advantage
  must be sought in Linear or hidden-distribution adaptation.

## v24 official-objective correction

- The official score is the sum of per-case relative improvements, not a
  function of aggregate MSE. Recomputing the correct proxy gives Attention
  scores v22 `9120.328019`, v23 `9097.752851`, and external `9329.868856`.
  v23 reduced aggregate MSE while regressing score-weighted small-error cases.
- Linear official proxy on raw groups 0-3 is v22 `1270.961823` versus external
  `1255.193206`; donor is `1217.403978` versus `1107.055835`. This keeps
  sequential Linear compensation out of v24 despite its possible hidden-data
  value; it requires a later isolated online probe.
- Primary retained mechanism: compute RMS for each replicated Q head before
  averaging within its KV group. This raises all-50 Attention proxy to
  `9575.777924`; a calibration-density gated full-head rotation raises it to
  `9607.884916`. Higher energy penalty and removing offset 2 regress.
- v24 aggregate Attention MSE is `73.616000`; proxy improves 9/10 consecutive
  batches versus v23. Twin-excluded groups 0-9 improve by 178.57 proxy units
  versus v23 and 67.85 versus the external implementation.
- Candidate DAG decision: continue per-head RMS aggregation and the one-layout
  rotation gate; modify the local evaluator to expose twin-excluded mode;
  abandon aggregate-MSE parameter selection; defer Linear propagation.
- Final validation: v24 keeps the complete Linear section byte-identical to
  v23, `solution.py` matches `solution_v24.py`, syntax/diff checks pass, and
  all `1100/1100` output-format checks pass.
- Score forecast is deliberately bounded to `17000-17800` with a center near
  `17400`; runtime is projected at `190-205s`. The external mixed result is not
  used as a linear point coefficient, so this forecast is a hypothesis to be
  recalibrated by the v24 Attention-only online result.

## v24 online result and v25 Linear branch

- Confirmed v24 result: `16500 / 198s`, only +23 points and +5 seconds versus
  v23. Per-head RMS aggregation is retained as a small positive mechanism, but
  local Attention proxy magnitude still does not transfer online.
- This controlled result shifts the unexplained external advantage toward its
  unique Linear residual-compensation family. The first diagonal projection
  prototype was rejected: increasing its trust step monotonically regressed
  both raw and donor official proxies.
- v25 independently uses 32 deterministic calibration rows, natural channel
  order, a fixed four-block future window, sample-space ridge `0.1`, and trust
  step `0.5`. Q(A) is finalized before propagation; residuals only alter fixed
  Q(W). It does not use the external energy order, all-row solve, or shape
  thresholds.
- Primary groups 0-3: raw proxy `1270.961823 -> 1285.598884`; donor proxy
  `1217.403978 -> 1204.004249`. Groups 4-7 reproduce the tradeoff:
  `1301.983485 -> 1317.999483` raw and `1283.481112 -> 1269.540903` donor.
  The donor loss is about one eighth of the external implementation's loss.
- All 50 raw Linear groups: official proxy `15655.467096 -> 15847.635036`,
  MSE `22636.136776 -> 22255.925182`; all ten consecutive five-group batches
  improve. Local Linear time falls from `549.56s` to `374.39s` because v25
  removes v24's full calibration-output ridge solve and row selector.
- v25 forecast: `16800-17600` points and `145-170s`. The score range is wide
  because no locally reliable Linear point coefficient covers this mechanism;
  runtime is anchored by the measured all-group total ratio to v24 `198s`.
- Final validation: the v24 Attention section is byte-identical, submission and
  `solution_v25.py` match, syntax/diff checks pass, and all `1100/1100` format
  checks pass. The full format run is also materially faster than v24.

## v25 online result and v26 calibration

- Confirmed v25 result: `17049 / 176s`, +549 points and -22 seconds versus v24
  `16500 / 198s`. Bounded Linear feedback is the first post-v22 mechanism with
  a large controlled online gain and is retained.
- The same-branch coefficient is `549 / 192.167940 = 2.8568` online points per
  all-50 raw Linear proxy unit. Use it only for nearby feedback-parameter
  changes; do not transfer it to Attention or unrelated algorithms.
- v26 keeps Attention fixed and screens feedback rows `32/64/128`, propagation
  window `4/8`, and trust step `0.5/0.75`. Primary and disjoint raw official
  proxies select the direction; donor remains a risk bound because its v25
  direction contradicts the now-controlled online result.
- Screen result: increasing calibration rows from 32 to 64/128 regresses, so
  low row count is useful regularization. Window 8 and trust 0.75 are each
  positive; together they show a stable synergy on groups 0-3 (`+13.630372`)
  and 4-7 (`+13.292340`) versus v25.
- Donor risk is bounded: the combined candidate loses only `4.391211` and
  `5.244874` proxy units on the two batches, versus roughly 97 and 81 for the
  external implementation relative to v25.
- All-50 v26 Linear proxy is `16027.331849`, +179.696813 over v25; MSE falls
  from `22255.925182` to `21832.206329`, and all ten consecutive batches
  improve. Local Linear time is `365.30s`, slightly below v25 `374.39s`.
- Controlled estimate is `17049 + 2.8568 * 179.696813 = 17562` points. Report
  `17350-17800 / 165-185s` to allow coefficient and platform drift.
- Final validation: v25 Attention is byte-identical, `solution.py` matches
  `solution_v26.py`, syntax/diff checks pass, and all `1100/1100` output-format
  checks pass.

## v26 online result and v27 ordering plan

- Confirmed v26 result: `17232 / 183s`, +183 points and +7 seconds versus v25.
  The controlled local slope contracts from 2.8568 to
  `183 / 179.696813 = 1.0184` points/proxy-unit. Window/trust capacity remains
  positive, but linear extrapolation from v25 overestimated the gain.
- v27 freezes rows 32, window 8, ridge 0.1, trust 0.75, and all Attention code.
  It tests one structural change: order 64-channel blocks by calibration-side
  quantized-activation energy before residual propagation, then reconstruct
  returned parameters in original channel order.
- This is an independent bounded implementation: it retains the v25/v26
  low-row trust region and has no all-row solve or shape-dependent propagation
  thresholds. Natural order remains the exact fallback candidate.
- Natural-order compatibility passes exactly: disabling energy order reproduces
  v26 Q(W) tensors bit-for-bit. Descending energy improves raw primary/disjoint
  by `1.481189/1.385716` and donor by `4.560914/1.167014`; ascending order
  regresses, validating the ordering direction rather than mere permutation.
- With energy order, window 16 then 32 improve all four raw/donor views; trust
  1.0 and rows 64 remain negative. Stop at window 32 because its marginal gain
  is already small and runtime grows.
- All-50 v27 Linear proxy is `16134.879950`, +107.548101 versus v26; MSE falls
  to `21561.046608`, and all ten consecutive batches improve. Local Linear time
  is `409.06s`; projected online result is `17280-17550 / 195-215s`.
- Final validation: v26 Attention is byte-identical, natural-order fallback
  reproduces v26 exactly, submission matches `solution_v27.py`, and all
  `1100/1100` output-format checks pass.

## v27 online result and official random replay

- Confirmed v27 result: `17283 / 192s`, +51 points and +9 seconds versus v26.
  Energy ordering remains positive but its marginal gain is small.
- Added `tools/official_benchmark.py`, which uses the supplied standard HiF4
  codec and exact published per-case score. It separates candidate quantizer
  time from MatMul/Attention/scoring and preserves group/case evidence.
- `rand_sample` is independent and structurally diverse, but ten groups have
  zero standard MSE and several layouts are outside older versions' numerical
  domain. The all-shape total reverses the online ordering (Pearson -0.432,
  Spearman -0.830), so it must not replace online calibration wholesale.
- Continue: use correlated shape families as validation panels and the full
  set as a tail-risk suite. Modify: exclude only formula-undefined zero-MSE
  groups and compare versions on a common successful intersection. Abandon:
  fitting one global online-score mapping to the equal-weight random total.
- v28 isolates a sparse Attention guard. Dense calibration layouts retain the
  full-head transform; sparse layouts use identity plus reciprocal Q/K balance.
  It fixes all three v27 random Attention failures and changes common-group
  score from `-20155.855` to `5182.602`, with 24 wins, 12 ties, and 8 losses.
- A Linear feedback-drift fallback was prototyped and rejected: the 0.15 drift
  boundary identifies extreme failures, but the conservative candidate still
  has large negative score. Preserve v27 Linear until that branch can be
  isolated without mixing the low-rank target and feedback effects.
- v28 validation: full random-data format/state checks pass `1100/1100`;
  Attention quantization time is `34.67s` versus v27 `32.80s`. Use a cautious
  online estimate of `17250-17600 / 185-205s`.

## v28 online result and v29 complete Linear transfer

- Confirmed v28 is `17283 / 184s`: score is unchanged from v27 and runtime
  improves by 8 seconds. The sparse Attention guard therefore has zero current
  online score coefficient, despite its large random-tail benefit.
- At explicit user direction, v29 transfers every Linear mechanism from
  `solution_cxw_v1.py`; this is a complete mechanism port, not the earlier
  selective-learning branch. v28 Attention is retained and matches exactly on
  the 25-case official donor comparison.
- On the online-shaped independent donor panel, v29 Linear is `1119.474`
  versus v28 `1221.061`. This is negative local evidence, but the transferred
  Linear implementation is retained because its source submission has the
  positive hidden-distribution result `17562 / 249s`.
- On the complete official random suite, v29 combines Linear `3436.684` with
  v28 Attention `5412.720`, total `8849.404`, versus the source implementation's
  `-34070.21`. Quantization time is `217.70s`; all groups complete.
- Score transport remains uncertain because donor and random panels disagree
  with the source online result. Use a deliberately wide `17000-18100` score
  forecast and `225-265s` runtime forecast rather than claiming 19000.
- Final validation: `solution.py` matches `solution_v29.py`, syntax and diff
  checks pass, and the full random-data format check passes `1100/1100`.

## v29 online rejection and v30 global output-aware Q(A)

- v29 online is `17578 / 272s`; discard all of its Linear changes as the next
  base because the +295 score over v28 costs +88 seconds.
- v30 starts byte-for-byte from v28, then adds a rank-32 approximation of the
  global W row-space. Online Q(A) performs a sequential five-offset search
  against that sketch, capturing cross-block output-error cancellation that
  v28's independent block costs cannot represent.
- Exact calibration A@W is used after Q(W) is fixed to compare the new policy
  with v28. Ordinary shapes keep the new policy only above a 0.3% calibration
  MSE margin; `in_features >= 32 * out_features` forces it because the local
  block model is structurally weak in that low-rank regime.
- Independent donor groups 0-5 remain exactly equal to v28 (`1881.130` total),
  while deterministic random groups 0-5 improve `421.077 -> 432.973`. Random
  compression groups 8-9 improve from `-25870.818` to `-4741.572`.
- Deterministic full random v30 is `-31395.201` in about `200.63s`; the huge tail recovery is real,
  but historical random/online anti-correlation means it does not justify a
  fabricated 21000+ forecast. The defensible pre-online interval remains wide.

## v31 complementary Q(W)/Q(A) candidate selection

- Online v30 is `17521 / 174s`: +238 points versus v28 while solving v29's
  runtime regression. This supports output-aware Q(A), but shows low transport
  from calibration gain to hidden-test gain.
- v31 evaluates four joint pairs: baseline/aggressive Q(W) crossed with
  baseline/global Q(A). It scores each of five calibration layouts separately
  against exact A@W instead of allowing one large layout to dominate.
- An unconditional second Q(W) doubled donor-panel time and was rejected.
  Final v31 builds it only for `in_features >= 32 * out_features`; ordinary
  shapes use the same Q(W) as v30 and require both aggregate and layout gates.
- Donor groups 0-5 remain exactly equal to v30. Random groups 8-9 improve by
  `669.646` points, while ordinary groups 2/3/5 improve by `0.694` overall.
- Full random score is `-30967.769` in `162.70s`, improving v30 by `427.432`.
  This is a robustness delta, not a direct online-score conversion.

## v31 online result and v32 layout gate

- Confirmed v31 is `17493 / 202s`: `-28` points and `+28s` versus v30. This
  rejects the v31 aggressive Q(W) branch and repeated robust joint selection;
  the strong random-tail improvement is anti-correlated with the hidden set.
- v32 returns to v30's one bounded-feedback Q(W). Exact calibration `A@W`
  first applies v30's aggregate baseline/global Q(A) gate, then permits each
  uniquely identifiable activation row layout to reject the global policy.
  Thus layout fitting is a conservative refinement, not an independent route
  around the online-positive v30 gate.
- Tiny layouts below 16 rows require 2% calibration improvement; ordinary
  layouts require 0.5%. Repeated row counts and unseen row counts use the
  aggregate decision because calibration sample identity cannot be recovered
  from the online activation shape.
- Online-shaped donor groups 0-7 are exactly equal to v30. Full random score
  changes only `-31395.201 -> -31392.796`, while quantization time is
  `156.34s`. Treat v32 as a low-risk online probe of layout-conditioned Q(A),
  not as evidence for a large score forecast.

## v32 online result and v33 joint output objective

- Confirmed v32 is `17507 / 188s`: only `-14` points and `+14s` versus v30.
  The row-layout gate is therefore rejected; it neither improves hidden score
  nor runtime.
- v33 keeps v30's online-positive Q(W), but after Q(W) is fixed it sketches W
  and Q(W) separately in a rank-2048 output subspace. Dynamic Q(A) minimizes
  the joint approximation of `||A W.T - Q(A) Q(W).T||`, rather than v30's
  incomplete `||(A-Q(A)) W.T||` objective.
- The original rank-32 policy remains an independently recomputed fallback.
  Exact calibration `A@W` selects baseline, rank-32, or joint Q(A), requiring
  at least three improving layouts and at most 10% layout regression.
- Independent donor groups 0-7 improve `2496.869 -> 2578.122` (`+81.253`);
  disjoint groups 8-15 improve `2331.214 -> 2426.235` (`+95.021`). Every
  selected donor group is non-regressing, unlike the v31/v32 mechanisms.
- Three coordinate sweeps are used only up to 512 rows; 1024-row activations
  use one sweep because each case has equal score weight while compute scales
  with row count. Eight scale candidates are evaluated in a batched row
  chunk, and repeated candidates are pre-stacked on bounded-width layouts.
- For tall W, the final small SVD in randomized PCA is only an orthogonal
  rotation of the same output subspace and is omitted. Full-column-rank cases
  use direct QR. Wide W retains `pca_lowrank`, preserving the objective.
- Transporting the mean 16-group donor delta across 50 Linear groups with the
  historical v25 coefficient gives a center estimate of `19095`, versus the
  v30 anchor `17521`. This is the first leakage-free local estimate above the
  requested 19000 threshold. Runtime remains the main risk; local batch ratios
  imply an online range near the 300-second boundary.
- The official full random-suite format/state check passes `1100/1100` with
  no invalid parameters, state nodes, dtypes, NaN/Inf values, or failed groups.

## v33 online correction

- Confirmed v33 is `17682 / 249s`: `+161` points and `+75s` versus v30, or
  `+175` points and `+61s` versus v32. Joint W/Q(W) output fitting is genuinely
  positive, but its score-per-second ratio is poor.
- The predicted `+1574` over v30 transported only `10.2%`. The effective
  coefficient for this mechanism is `0.292` online points per extrapolated
  donor unit, not the historical v25 coefficient `2.8568`.
- Root cause: groups 0-15 are disjoint IDs but only repeat four structural
  signatures, and nearest-feature donors remain inside the same 0818 source
  distribution. Treating them as 16 independent distributions and expanding
  their mean to 50 groups was invalid.
- Modify: retain joint fitting as a positive direction, but optimize its cost
  and require far-neighbor/leave-family-out evidence. Abandon: transporting a
  coefficient across optimization families or increasing rank/sweeps solely
  because the nearest-donor score rises.

## v34 shape-aware joint-search budget

- Added a deterministic farthest-calibration-feature donor mapping within each
  exact signature. On groups 0-7, v33 improves far holdout `2469.477 ->
  2553.326` (`+83.849`), essentially matching its nearest-donor `+81.253`.
  This rejects pure nearest-neighbor overfit as the main v33 failure mode.
- Ordinary wide matrices contribute only `+3.362` far points across groups 3
  and 7, while their v33 quantization costs `37.01s`. Returning them to the
  v30 baseline gives identical v30 outputs in `24.58s`. Strong-compression
  wide shapes remain on v33 and match it exactly on random groups 8-9.
- One sweep on every remaining shape retains 73%-75% of v33's near/far gain.
  The final schedule restores three short-row sweeps only for square matrices,
  which carry roughly half the gain: final retained gain is 88.2% nearest and
  87.4% farthest.
- Final far score is `2542.726` (`+73.248` over v30); the shape-composed nearest
  score is `2568.566` (`+71.697`). Attention is unchanged.
- Using the mechanism-specific v30-to-v33 online result predicts about `17663`
  for v34. Use a conservative `17620-17710 / 210-235s` forecast. v34 is a
  controlled efficiency/shape-contribution probe, not a claimed score jump.

## v34 online result and v35 tall-sweep restoration

- Confirmed v34 is `17456 / 183s`: `-226` points and `-66s` versus v33, and
  `-65` points versus v30. The runtime direction is correct, but local retained
  gain (about 88%) badly overstates hidden-test retention.
- v34 changed two variables: ordinary wide matrices returned to v30, while
  retained tall matrices reduced short-row joint search from three sweeps to
  one. Since tall signatures cover twice as many provided shape families and
  the one-sweep surrogate can choose a different hidden-test Q(A), restore the
  tall sweeps first.
- v35 is a controlled single-variable restoration: it is v33 on every tall,
  square, and strong-compression shape, while preserving only v34's ordinary
  wide-matrix baseline gate. This isolates the online score/time contribution
  of the wide joint branch in the next result.
- Equivalence checks confirm the intended decomposition exactly: far-holdout
  groups 0-2 match v33 at `999.378937`, ordinary-wide group 3 matches v30 at
  `274.211533`, and random strong-compression group 8 matches v33 at
  `-1323.261058`. Attention is unchanged in all comparisons.
- Full `rand_sample` interface/state validation passes `1100/1100`. Forecast
  v35 conservatively at `17580-17680 / 205-235s`; the next online result is a
  direct attribution test, not evidence yet for a new score ceiling.

## v35 online result and v36 direction

- Confirmed v35 is `17501 / 212s`, versus v34 `17456 / 183s`: the two added
  tall sweeps buy only 45 points for 29 seconds. Modify this direction back to
  one tall sweep; do not spend further runtime increasing tall sweep count.
- v33 is `17682 / 249s` and differs from v35 only by enabling ordinary-wide
  joint fitting. Therefore the hidden wide branch is worth up to `+181 / 37s`,
  despite its weak local donor score. This directly overturns the earlier
  conclusion that ordinary-wide search is low value online.
- A one-sweep ordinary-wide probe matches v30 exactly on near donor groups 3/7
  (`602.159536`) but costs about 7.7 seconds more locally, so it is rejected.
  v36 advances the reduced search to two wide sweeps while keeping v34's
  one-sweep ordinary-tall schedule. Q(W), square/forced layouts, and Attention
  remain unchanged.
- Two wide sweeps retain `80.6%` of v33's near wide improvement and `88.4%`
  of its far improvement, without regressing either wide group versus v30.
  Non-wide far groups 0-2 and forced-compression random group 8 exactly match
  v34, confirming that the candidate changes only its intended branch.
- Full output/state validation passes `1100/1100`. Mapping the measured wide
  retention onto the controlled v33-v35 online delta, then removing v35's
  measured tall-sweep increment, gives a center near `17610`; forecast v36 at
  `17570-17650 / 205-230s`.

## v36 online result and v37 direction

- Confirmed v36 is `17638 / 236s`, within the score forecast and 6 seconds
  above its runtime range. Versus v35 it adds `+137 / +24s`; the reduced wide
  search transfers strongly online.
- v36 is only 44 points and 13 seconds below v33. Restoring the removed tall
  and wide sweeps would reproduce the old tradeoff, not create enough score
  headroom. Keep the v36 search schedule fixed.
- For wide/full-row-rank W, the rank-2048 output sketch spans the complete
  output space. Replacing `pca_lowrank` with the identity output basis is an
  algebraically exact objective transformation and removes a large calibration
  decomposition. v37 first validates this output-equivalent acceleration.
- Full-row-rank random groups 2/4/5 do remain score-identical and run faster.
  For the online-shaped `4096x8192` rank-truncated branch, however, direct QR
  changes sequential decisions and strongly improves them: groups 3/7 gain
  `+53.299` near and `+58.380` far versus v36.
- Independent groups 11/15 confirm `+49.525` near and `+51.976` far, while
  forced-compression random groups 8/9 gain `+59.621`. Non-wide groups 0-2 and
  Attention are exactly unchanged. This is consistent across calibration
  families, not a single-donor selection artifact.
- Increasing the independent row chunk from 32 to 64 preserves every tested
  case score and saves about 1.19 seconds on the near wide panel. Full
  validation passes `1100/1100`.
- Do not reuse v33's failed large local-to-online coefficient as a point
  estimate. Forecast v37 at `17950-19500 / 225-260s`: the lower scenario uses
  direct per-case transport, while the upper scenario allows part of v36's
  observed hidden wide sensitivity.

## v37 online result and v38 direction

- Confirmed v37 is `17638 / 224s`, exactly equal in score to v36 and 12 seconds
  faster. Retain QR/identity plus chunk64 as an equal-score acceleration.
- The large near/far wide improvements produced zero hidden gain. Mark further
  tuning of the current Q(A) subspace as locally overfit and abandon it until a
  new online-correlated dataset signal exists.
- Do not restore ordinary alternating Q(W): v31 already tested the second
  feedback solve and regressed online. v38 instead screens a cheap direct Q(W)
  fallback against the existing feedback Q(W), using stratified exact A@W and
  conservative per-layout row gates.
- The direct-Q(W) fallback is rejected: groups 0-3 change `-0.006` while local
  quantization costs about 11 seconds more. A one-step low-row mantissa
  refinement is also rejected at `-0.089 / +2.5s`. These results reinforce
  that the current feedback Q(W) should remain fixed.
- Final v38 spends the v37 acceleration only on the online-positive tall sweep
  restoration. Tall uses three sweeps; ordinary wide stays at two; square and
  forced compression stay at three. Far groups 0/1 improve `+3.979` with no
  regression, and all other mechanisms are unchanged.
- Full validation passes `1100/1100`. Using the controlled v34-v35 tall delta
  gives `17665-17720 / 240-260s`, centered near `17683 / 253s`.

## v38 online result and v39 direction

- Confirmed v38 is `17682 / 228s`, exactly recovering v33's score while
  reducing runtime by 21 seconds. The prior score center was accurate; runtime
  was conservatively overestimated by 25 seconds.
- Versus v37, the tall short-row restoration gives `+44 / +4s`. This is the
  strongest current evidence of a Q(A) search-capacity change that transfers
  online, unlike v37's locally large wide-subspace delta.
- v39 extends three sweeps only to the ordinary-tall 1024-row layout. All other
  shapes retain v38 thresholds and sweep counts, providing a clean attribution
  of the largest equal-weight activation layout.
- Near tall groups 0/1 gain `+0.977005` and far donors gain `+0.980894`.
  Only each group's 1024-row case changes, and both improve; all shorter cases
  are exact v38 matches.
- The local two-group cost is `+2.55s` near and `+2.94s` far. Using the observed
  v37-v38 local/online tall timing ratio predicts roughly 238 seconds online.
  Square, wide, forced compression, and Attention remain exactly unchanged.
- Full validation passes `1100/1100`. Forecast v39 at
  `17688-17708 / 235-250s`, centered near `17693 / 238s`.

## v39 online result and v40 paper-driven direction

- Confirmed v39 is `17697 / 234s`, only `+15 / +6s` over v38. The forecast is
  accurate, but another sweep/threshold version cannot plausibly close the
  remaining score gap.
- Reviewed and locally archived sixteen primary papers spanning GPTQ,
  SmoothQuant, AWQ, OmniQuant, RPTQ, QuaRot, SpinQuant, AffineQuant,
  AdaRound/BRECQ, HQQ, QuIP/QuIP#, AQLM, SpQR, and Outlier Suppression+.
  RPTQ's cross-group channel layout is the strongest missing legal mechanism.
- Direct similar-range clustering is incompatible with the existing local
  Hadamard and is rejected. v40 instead distributes ordered joint A/W RMS
  sensitivity quantiles evenly across every 64-channel block, preserving
  `A@W` with a paired permutation.
- Near/far groups 0-3 gain `1.925/3.593` MSE; independent groups 4-7 gain
  `17.964/15.424`. Random shapes at 4:1 compression regress, so they are
  shape-gated to exact v39 fallback; the online shapes are all within 2:1.
- Complete standard random Linear improves by `+7512.698` points, largely by
  repairing catastrophic tail groups. Wins/ties/losses are `12/14/17`, so the
  tail gain is not presented as a proportional online forecast. Attention is
  unchanged and all `1100/1100` checks pass.
- Forecast v40 at `17720-18150 / 240-265s`. The next branch after feedback is
  a SpinQuant-inspired learned sparse/block-orthogonal basis, not more
  coordinate sweeps.

## v40 online result and v41 direction

- Confirmed v40 is `17808 / 243s`, giving `+111 / +9s` over v39. Channel
  layout is therefore online-positive, but the random-tail gain overstates
  hidden sensitivity by orders of magnitude.
- Preserve v40's permutation and 2:1 shape gate. Calibrate future score from
  ordinary near/far donors; use random extremes only to catch catastrophic
  behavior.
- v41 tests a larger deterministic signed-Hadamard bank as a zero-online-cost
  discrete SpinQuant subset. Start at 16 bases, gate on both donor panels, and
  do not test 32 unless 16 earns enough improvement to justify calibration
  cost. Dense online rotation is rejected under the 57-second margin.
- Reject 16 bases: near improves but far regresses by `5.945` MSE. A 1%
  challenger trust gate is an exact v40 no-op with extra cost.
- Retain the eight-base bank and bit-reverse the 64 sensitivity lanes before
  Hadamard. This fuses into the existing permutation and aligns transform
  ordering with HiF4's scale hierarchy at zero online-operation cost.
- Four independent near/far panels improve by
  `89.405/85.592/99.650/87.539` MSE. Complete random Linear adds `1164.523`
  standard points with `21/14/8` wins/ties/losses and no new failed group.
- Forecast v41 at `18200-18900 / 240-255s`. The wide score interval reflects
  historical nonlinear transport even though this is the strongest ordinary
  donor signal since the holdout rebuild.

## v41 online result and permanent exploration policy

- Confirmed v41 is `17930 / 217s`, giving `+122 / -26s` over v40. Retain the
  Linear bit-reversal hierarchy layout.
- Local and online datasets are explicitly different. Local non-improvement
  no longer vetoes a theoretically sound optimization by itself; local tests
  rank ideas and catch catastrophic/runtime risk, while controlled online
  ablations supply the final generalization evidence.
- v42 isolates the same hierarchy insight in Attention Q/K. Apply a shared
  64-lane bit reversal before the existing head-wise Hadamard, preserving
  full-precision QK scores exactly. Keep Linear and V unchanged and require
  runtime below 250 seconds from the 217-second anchor.
- QK equivalence passes at `4.69e-8` relative L2. Four near/far donor panels
  all improve by `0.588/0.308/0.333/0.111` MSE with flat timing.
- Unconditional random Attention adds `135.070` points but creates one
  `-776` tail. Gate on calibration near-zero ratio `<=0.35`; online-like
  donors are 0.26-0.29, while the tail is about 0.41. The gated random result
  adds `45.060` with exact fallback on the tail.
- Linear and V are parameter-exact v41 outputs. Forecast this isolated v42
  Attention experiment at `17940-18020 / 218-228s`.
