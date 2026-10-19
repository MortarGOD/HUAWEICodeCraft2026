# SpinQuant-LWC candidate

This candidate extends the RRS solution with a learned Linear transform:

- SmoothQuant balancing is applied first.
- A shared block-64 orthogonal rotation starts from the normalized Hadamard
  matrix.
- A six-step straight-through calibration objective jointly models weight and
  activation HiF4 error.
- The rotation uses a Cayley parameterization, so the full-precision Linear
  output remains equivalent before clipping and quantization.
- Learnable weight/block and activation/block clipping ratios are optimized
  with the rotation.
- The best calibration iterate competes with every existing RRS family under
  the normal stability gate; numerical failures fall back to RRS.

## Local results

| Dataset | Status | Linear | Attention | Total | Runtime |
|---|---|---:|---:|---:|---:|
| compact v1 (20 cases) | valid | 1.112738 | 1.185177 | 2.297915 | about 4 s |
| Qwen3.5 trace (210 cases) | valid | 75.542190 | 6.876861 | 82.419051 | about 77 s |

The prior RRS candidate scores `82.355253` on the same Qwen trace. Local
scores are diagnostic; online evaluation remains authoritative.

## Rejected variants

- One independent rotation per block overfit the three calibration samples
  and scored `81.609628`.
- Four interleaved rotation experts scored `82.108806`.
- Activation-only SpinQuant training scored `82.010427`; this task quantizes
  both operands and cannot apply the paper's later GPTQ weight correction.
- Leave-one-out rotation training scored below the all-sample calibration.
- A stricter second-stage learned-family gate removed small but useful gains.

