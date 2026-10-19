# Online-safe refine20 Linear + fixed-attn4 Attention

This is the recommended repair for the runtime160/170 Wrong Answer family.
It removes the failed Linear output-compensation route and avoids the
K-Hessian Attention route whose latest online feedback regressed score and
reduced the timeout margin.

- Linear is exactly `combo-projection-target-refine20`, which scored
  `16631.316` online in `139.552s` with all Linear cases passing.
- Attention is exactly the fixed-attn4 component from
  `rrs-fast-fp16-fixed-attn4-safe-anchor`, which passed all online Attention
  cases and completed the combined anchor in 144.21 seconds.
- No case IDs, layer IDs, failure-group tables, or dataset-specific routing are
  used.

## Validation

| Dataset | Linear | Attention | Total | Passed | Time |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 online structure | 228.4678 | 186.0392 | **414.5069** | 500/500 | 61.35s |
| Qwen3.5 | 132.6442 raw | 13.1555 raw | 273.0343 weighted | 210/210 | 40.60s |
| compact v1 | - | - | 5.4199 | 20/20 | 4.20s |

`verify_composition.py` confirms byte-exact source boundaries against the two
online-passing components. The shared shape robustness suite passed five
Linear layouts and five MHA/GQA layouts: 20 repeated Linear calls and 47
Attention calls, including missing and stale cache conditions.

The known-WA runtime170 candidate scores `456.7363` locally, so this repair
retains 90.8 percent of that new local score while replacing both risky
components with online-passing paths. Online evaluation remains required for
the combined runtime and score.

Formal archive SHA256:
`621DED02BB74738EB6995D233C2D0CE8AB5778666A784A84C485E6E2C1445D83`.

## 2026-08-25 refresh

The current online-structure datasets reproduce the archived scores:

| Dataset | Linear | Attention | Total | Worker time |
|---|---:|---:|---:|---:|
| Qwen2.5 online structure | 228.4678 | 186.0392 | 414.5069 | 56.48s |
| Qwen3.5, 1:1 weighted | 207.2566 | 65.7777 | 273.0343 | 35.39s |

All 710 judge cases passed. The shared robustness runner also passed 20
repeated Linear calls and 47 MHA/GQA Attention calls. Relative to the corrected
23328 baseline's Qwen2.5 score of 385.1788, direct proportional scaling gives
about 25104 online points. This is only a prioritization estimate: hidden
distribution transfer and the 300-second online runtime remain authoritative.

This candidate is preferred over the SOAR, pairwise MR-GPTQ, Hessian mantissa,
and Smooth-RHT experiments tested on the same date. Those candidates remained
valid but did not improve both local model traces.
