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
              └─ v19 (held-out lambda controller; current online probe)
              └─ v20 (A-only layout profile selection; current candidate)
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
