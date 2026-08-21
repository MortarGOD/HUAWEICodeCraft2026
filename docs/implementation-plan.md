# Implementation Plan Draft

## Objective

Develop the next version from v16 while keeping online runtime below 300
seconds and improving output accuracy through calibration-time `A @ W` fitting
of `Q(A)` and/or `Q(W)`.

## Candidate directions

1. Reuse the rank-2 Gram state across weight and activation candidate scoring.
   Risk: extra tensor materialization can erase the runtime gain. Validate on
   the primary and disjoint subsets with peak memory and phase timing recorded.
2. Replace the fixed ridge fit with a small data-driven lambda choice using
   already-computed calibration outputs. Risk: another candidate may increase
   calibration time. Reject unless it improves Linear MSE by at least 0.5%
   without adding more than 5% local runtime.
3. Compress the output-aware activation objective while retaining both Gram
   directions. Risk: approximation error can reverse the v16 online gain.
   Compare exact v16 codes, output MSE, and runtime per group.

## Executable steps

1. Run v16 on Linear groups 0-3 and Attention groups 0-4 as the primary
   benchmark; record raw MSE and phase time.
2. Implement one isolated candidate from the directions above and archive it
   under a new version number.
3. Run the same primary benchmark, then Linear groups 4-7 and Attention groups
   5-9 to reject subset overfitting.
4. Run `py_compile`, `self_check.py`, `git diff --check`, and verify the v13
   Attention section hash if Attention was intentionally unchanged.
5. Update the candidate DAG with the evidence and mark the direction continue,
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
