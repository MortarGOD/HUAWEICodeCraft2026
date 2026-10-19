# RRS-fast FP16 online refresh anchor

This candidate freezes the original `rrs-fast-fp16` implementation without
algorithm changes. It exists to remeasure the historically online-effective
RRS mechanism on the current judge after the PSOT exact-Gram checkpoint showed
only a `+26.97` online gain.

The Linear path uses calibration-derived activation energy, SmoothQuant-style
channel balancing, a 64-wide Hadamard transform, and block-covariance-aware
HiF4 search. The frozen block Hessian is stored as FP16 and restored to FP32
for dynamic activation quantization. Dynamic calls do not compute the current
`A@W` or consume reference outputs.

## Evidence

- Historical remote feedback for `rrs-fast-fp16`: `20000+` (runtime was not
  recorded in the archived report).
- Latest profiled-v2 strict check: 500/500 cases, Linear `171.116896`,
  Attention `138.899553`, total `310.016449`.
- Strict interface isolation passes fresh Q/K/V imports, all call-order
  permutations, and cross-sample checks.
- Local strict wall time: `58.43s`; this is a correctness and transfer anchor,
  not a new local-score claim.

Submit the root-only archive
`../zips/rrs-fast-fp16-online-refresh-safe.zip` unchanged. Record the current
online result in this directory before selecting any RRS-derived refinement.

Archive SHA256:
`6D285F6E56BAD230FBA20B15CF044F985A274E3B0D1DC33A1451D810518F5DEC`.
