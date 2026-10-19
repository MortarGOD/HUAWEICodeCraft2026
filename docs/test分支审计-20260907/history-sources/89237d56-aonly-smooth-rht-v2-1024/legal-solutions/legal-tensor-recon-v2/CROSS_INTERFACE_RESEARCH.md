# Cross-Interface Quantization Research

Rule used for this candidate: calibration may select later hyperparameters, but
no runtime state value may depend on W. Global constants can be tuned offline;
per-group W-dependent constants, tensors, booleans, indices, routes, and
control-flow encodings cannot cross into the dynamic activation interface.

## Paper-to-rule mapping

| Source idea | Decision | Reason |
|---|---|---|
| Fixed random Hadamard transform | Use | Orthogonal, input-independent, and needs no W-derived state. |
| Per-tensor HiF4 reconstruction search | Use | Each interface optimizes only the tensor it currently owns. |
| A-only RMS channel scale | Use | State is derived exclusively from calibration A; W only consumes the reciprocal scale. |
| Calibration Q/K reciprocal balance | Use | Attention calibration receives no W and Q/K dot products are preserved. |
| SmoothRot scale using both A and W statistics | Reject | The scale contains W information and would be exposed to dynamic A. |
| Learned SpinQuant/FlatQuant rotation for each layer | Reject | A learned layer matrix can encode W/output information even if later frozen. |
| Output-error, Hessian, projection, or residual fitting | Reject | These couple W or A@W into state or into dynamic activation decisions. |
| W-selected identity/RHT boolean | Reject | A one-bit route is still W information. |

## Relevant findings

- Czakó et al. define their robust SmoothRot scale with an activation quantile
  and a W-channel maximum. The quantile insight is separable, but their full
  formula and output-error STE objective are not legal under the no-W-state
  rule. Local HiF4 sweeps also found `q=0.994` weaker than A-only RMS.
- QuaRot establishes that fixed orthogonal rotations preserve the full-
  precision computation while reducing outliers. This is the cleanest legal
  transform family here.
- SpinQuant reports meaningful variance between random rotations, motivating a
  fixed-seed sweep. The final candidate keeps seed 0 because alternatives did
  not improve the two-dataset objective; it does not learn a layer rotation.
- NVIDIA's NVFP4 recipe enables RHT by default and computes hierarchical scales
  from the tensor being quantized. HiF4 uses 64-element groups and three-level
  scale metadata, motivating the fixed 64-point RHT and exact local scale-rank
  search.

## Primary sources

- Czakó et al., 2026: https://arxiv.org/abs/2606.09927
- QuaRot: https://arxiv.org/abs/2404.00456
- SpinQuant: https://arxiv.org/abs/2405.16406
- FlatQuant: https://arxiv.org/abs/2410.09426
- HiFloat4: https://arxiv.org/abs/2602.11287
- NVIDIA NVFP4 documentation:
  https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.14/user-guide/features/low_precision_training/nvfp4/nvfp4.html
