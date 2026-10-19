# RRS-fast FP16 with SOAR-style HiF4 scale decoupling

This paper baseline adapts SOAR's Decoupled Scale Search to the fixed HiF4
format. Weight calibration searches separate high-precision quantization
scales and legal E6M2 dequantization scales. The candidate set includes the
original coupled scale, and a layer switches only when its mean calibration
gain is at least 0.1 percent with no sample regressing by more than 1 percent.

Dynamic Activation and all Attention paths are byte-equivalent to
`rrs-fast-fp16`; the new search runs only during multi-sample Weight
calibration. No case IDs, layer tables, dynamic-call caches, or dataset
fingerprints are used.

## Validation

| Dataset | Baseline | SOAR | Delta | Passed | Worker time |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 online structure | 209.3497 | 209.3837 | +0.0341 | 500/500 | 52.69s |
| Qwen3.5, 1:1 weighted | about 152.3085 | 152.3145 | about +0.0060 | 210/210 | 35.23s |

The implementation is valid and improves both traces, but the gain is too
small for online submission and the calibration search is materially slower.
It is retained as a format-correct SOAR baseline rather than a leaderboard
candidate.

Formal archive: `../zips/rrs-fast-fp16-soar-safe.zip`
