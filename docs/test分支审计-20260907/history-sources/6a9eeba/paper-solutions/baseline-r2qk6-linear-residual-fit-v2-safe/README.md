# Corrected r2qk6 with robust post-quant compensation selection

This candidate preserves the corrected 23328-point baseline Attention source
byte-for-byte. It changes only the calibration-time selection of the existing
Linear output-error compensation strength.

## Algorithm

The corrected baseline first fits a continuous activation correction and then
quantizes the corrected activation. A continuous optimum can cross HiF4
rounding boundaries after quantization, so this candidate compares two legal
strengths using the actual quantized calibration output:

```text
baseline strength = fitted continuous strength
candidate strength = min(1.5, baseline strength + 0.25)
```

The stronger route is enabled only when its mean relative calibration gain is
at least 8% and every calibration sample is non-regressing. The threshold is a
distribution-independent margin gate: weak Qwen3.5 selections at 3.28% and
2.39% were rejected, while retained Qwen2.5 selections start at 8.12%.

Dynamic Linear execution is unchanged except for the selected scalar already
present in baseline state. There are no case IDs, layer tables, tensor
fingerprints, runtime Q/K/V caches, or exception-driven routes.

## Validation

| Dataset | Linear | Attention | Total | Passed | Worker time |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 online structure | 248.0326 | 137.3025 | 385.3351 | 500/500 | 94.54s paired |
| Qwen3.5 model holdout | 230.6642 | 44.3483 | 275.0126 | 210/210 | 31.18s |

Against corrected r2qk6, Qwen2.5 Linear improves by `+0.1563`; Qwen3.5 and
both Attention branches are score-identical. The paired Qwen2.5 baseline took
`91.57s`, giving a candidate/baseline ratio of `1.0325x`. Scaling that ratio by
the baseline's 244-second online run predicts about `251.9s`, inside the
engineering target. Online timing remains authoritative.

Run `verify_composition.py` to check syntax, forbidden routing tokens, and the
byte-identical Attention suffix.

Formal archive: `../zips/baseline-r2qk6-linear-residual-fit-v2-safe.zip`

SHA256: `8CB404FA95C2D43D9A706AA89D896DCE8C5B2EF7CB178E8AC9E2E8F12FD26D9D`
