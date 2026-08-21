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
