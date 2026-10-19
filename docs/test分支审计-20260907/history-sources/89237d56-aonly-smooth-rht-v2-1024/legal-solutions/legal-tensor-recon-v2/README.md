# A-Only Smooth-RHT HiF4 V2

This candidate enforces the strict rule that no information derived from `W`
may reach dynamic activation quantization. Calibration may change later
hyperparameters, but the Linear state is derived from calibration A only.

## Algorithm

Linear uses an exact equivalent transform. Calibration computes per-channel RMS
from A, centers log-RMS in 1024-channel blocks, and forms
`balance = 2 ** (-alpha * centered_log_rms)` with `alpha=0.388671875` and a
`+/-3` log2 clamp. A is multiplied by `balance`; W is divided by the same
balance. Both are then transformed by the same fixed signed 64-point RHT.

Attention computes a reciprocal Q/K balance from calibration Q/K RMS only.
Q is multiplied and K is divided by that balance with exponent `0.359375`,
per-64 log centering, and a `+/-2` log2 clamp. Q/K then use the same fixed
signed 64-point RHT. V stays in its original coordinates.

All streams search seven E6M2 top-scale ranks `(-3, ..., 3)`. HiF4 top scale,
lv2, lv3, sign, and mantissa are selected using only the current tensor's FP32
reconstruction error.

## No-W information flow

- `_linear_activation_balance` accepts only `calib_activation_list`.
- `activation_state` contains exactly `{"balance": A_only_balance}`.
- The W tensor is read only after the A-only balance has been computed.
- No W-derived tensor, scalar, boolean, route, index, shape encoding, error,
  Hessian, projection, or correction is returned in state.
- Negating W and multiplying every W source scale by `1.125` left the returned
  state bit-identical for all 82 Linear groups across both datasets.
- Attention calibration receives no W. Its states contain only Q/K-derived
  reciprocal balance parameters.

## Formal local results

| Dataset | Linear | Attention | Total | Cases | Worker |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 profiled-v2 | 159.370758 | 149.674183 | 309.044942 | 500/500 | 30.79s |
| Qwen3.5-35B-A3B | 106.805717 weighted | 47.513465 weighted | 154.319181 | 210/210 | 20.61s |
| Combined | 266.176475 | 197.187648 | **463.364123** | 710/710 | 51.40s |

Both formal ZIP reports are `valid`. Qwen3.5 maps its 160 Linear and 50
Attention cases to 250 reference cases per branch. These are local surrogate
scores, not online scores.

The prior adaptive-RHT baseline has normalized combined score `233.690978`;
this candidate improves it by `229.673145` locally.

## Rejected searches

- W-dependent route selection was removed as forbidden information flow.
- A-only identity/RHT routing reduced Qwen2 score and was rejected.
- The paper's `q=0.994` robust quantile was weaker than RMS in this HiF4 setup.
- Radius 4 was identical to radius 3; asymmetric five-rank windows regressed.
- Alternative fixed RHT sign seeds did not improve the two-dataset objective.

Submit-ready ZIP: `legal-solutions/zips/legal-aonly-smooth-rht-v2-1024.zip`

SHA-256: `046DB04D817AD0EFC0511656E57A2C344CC0AADAF790D63A54532190AA603DB9`
