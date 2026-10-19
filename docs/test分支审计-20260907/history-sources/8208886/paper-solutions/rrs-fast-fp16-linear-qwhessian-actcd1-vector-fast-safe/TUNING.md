# Q(W)-Hessian activation refinement

## Candidate

- Submission file: `solution.py`
- Submission archive: `submission.zip`
- Parent baseline: `rrs-fast-fp16-online-refresh-safe`
- Target: improve Linear accuracy without changing Attention or adding calibration replay cost.

## Mathematical changes

1. Linear weighted scale search uses radius 4; Attention remains at radius 3.
2. The dynamic activation quadratic is built from the emitted quantized weight:
   `Hq = Q(W)^T Q(W) / d_out`.
3. After the normal hierarchical scale search, all eight `lv2/lv3` group
   configurations are evaluated in one vectorized pass under `Hq`.
4. Each 64-channel block applies only its best negative-objective group update.
   This is a one-step Jacobi refinement, so the update is deterministic and bounded.

The final candidate does not retain the output-error STE route. That route added
only 0.0823 local Linear points but raised strict worker time from 56.08s to
76.83s.

## Constraint audit

- Calibration may derive `Hq` from `Q(W)`.
- Dynamic activation quantization uses only the current activation and frozen
  state (`Hq`, importance, transform, and balance).
- Dynamic code does not calculate `A @ W`, `A @ Q(W)`, or reconstruct a target
  activation from a reference output.
- No module-global cross-call cache is used.

## Strict local result

Dataset: `qwen2_5_14b_online_profiled_v2`

- Status: `valid`
- Linear: `176.7426686843695`
- Attention: `138.8995530261668`
- Total: `315.6422217105363`
- Worker time: `56.08052330091596s`
- Fresh-interface, call-order, and cross-sample checks: passed

The parent baseline measured Linear `171.1168961977201`, Attention
`138.8995530261668`, and worker time about `56.19s` on the same local suite.
The Linear gain is `+5.6257724866494` with no material local runtime increase.

## Rejected variants

- Static calibration cross term `2 e b`: a fitted mean `b` did not transfer and
  also worsened calibration MSE because inner hierarchical choices were not
  jointly optimized. It is not used here.
- Full eight-group coordinate descent: Linear `177.0068979431538`, but strict
  worker time was `119.98s`.
- One-group refinement plus STE: Linear `176.8250159048524`, but strict worker
  time was `76.83s`.
