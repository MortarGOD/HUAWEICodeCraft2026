# RRS runtime variants

The original RRS calibration evaluates five Linear transform families and four
weight/activation routes per family. On the Qwen trace, all 32 Linear groups
select the same result: SmoothQuant followed by Hadamard, with weighted
radius-3 block-Hessian quantization for both operands.

The fast path applies that algorithm directly whenever at least two
calibration samples are available. It avoids redundant reference matmuls,
permutation construction, candidate quantization, and candidate output
evaluation. A one-sample input retains the original search as a fallback.

## Results

| Candidate | Qwen total | Runtime | Relative speed | Purpose |
|---|---:|---:|---:|---|
| original RRS | 82.355253 | 49.95 s | 1.00x | Reference |
| `rrs-fast` | 82.355199 | 20.94-23.41 s | 2.1-2.4x | Near-exact fast path |
| `rrs-fast-fp16` | 82.348330 | 19.83-20.67 s | 2.4-2.5x | Recommended for slower judges |
| `rrs-lite` | 73.650137 | 16.52 s | 3.0x | Emergency timeout fallback |

All three archives pass the compact judge with 20/20 cases and score
`2.297915`. The Qwen runs pass all 210 cases.

`rrs-fast-fp16` stores the online block-Hessian state as FP16 and restores it
to FP32 inside the quantizer. `rrs-lite` removes the Hessian and weighted
radius-3 search entirely, which explains its larger quality loss.

