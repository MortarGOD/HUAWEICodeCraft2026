# Low-Precision Quantization Synthesis for v40

## Scope

This review maps sixteen primary papers to the fixed HiF4 interface rather
than treating language-model results as directly transferable. The contest
requires one fixed HiF4 `Q(W)`, dynamic HiF4 `Q(A)`, contiguous 64-value scale
groups, and an online runtime below 300 seconds. Calibration may optimize any
equivalent representation with `A @ W`; online code may use only the stored
state and the arriving activation.

The PDFs and extracted full text are stored in `docs/papers/quantization/`.
Humanize is unavailable in this workspace, so the candidate plan and
acceptance criteria were refined manually.

## Paper-to-contest mapping

| Paper | Mechanism | Existing coverage | Contest decision |
|---|---|---|---|
| GPTQ (2210.17323) | inverse-Hessian sequential error compensation | residual-feedback `Q(W)` and output-aware projections | retain; further low-lambda solves previously overfit |
| SmoothQuant (2211.10438) | reciprocal per-channel A/W scaling | exponent-0.44 multiplier | mature; do not reopen alone |
| AWQ (2306.00978) | activation-aware channel scaling | RMS balance and importance weighting | substantially covered |
| OmniQuant (2308.13137) | learned equivalent scales and clipping | scale-offset search and calibration output gates | partially covered; gradient search has runtime/overfit risk |
| RPTQ (2304.01089) | cluster and reorder channels with similar ranges | **missing across 64-channel boundaries** | highest-priority v40 direction |
| QuaRot (2404.00456) | equivalent Hadamard rotations | signed Hadamard-64 inside fixed groups | covered only within existing blocks |
| SpinQuant (2405.16406) | optimize orthogonal rotations for quantization loss | random sign choice only | promising successor; dense online rotation is expensive |
| AffineQuant (2403.12544) | learned invertible affine basis | diagonal scaling only | permutation/block-rotation subsets are feasible |
| AdaRound (2004.10568) | output/Hessian-aware rounding decisions | coordinate output search and feedback rounding | covered; high-capacity refinement already transferred weakly |
| BRECQ (2102.05426) | block reconstruction with coupled errors | calibration output objective | covered in spirit; block training risks calibration overfit |
| HQQ (2309.15531) | fast proximal weight quantization | no direct equivalent | fixed HiF4 codebook limits applicability |
| Outlier Suppression+ (2304.09145) | equivalent shift and scale | scale covered; shift needs bias compensation | reject shift under returned-state interface |
| QuIP (2307.13304) | incoherence processing and lattice quantization | Hadamard incoherence covered | lattice format incompatible; retain incoherence principle |
| QuIP# (2402.04396) | improved incoherence and lattice code | signed Hadamard partially covered | codebook incompatible |
| AQLM (2401.06118) | additive vector codebooks | absent | incompatible with fixed HiF4 representation |
| SpQR (2306.03078) | isolate sensitive weights/outliers | importance only; no side storage | mixed precision incompatible, sensitivity useful for grouping |

## Main deductions

### 1. Contiguous groups make channel layout an optimization variable

RPTQ identifies persistent per-channel range differences and groups channels
with similar calibration minima/maxima before sharing quantization parameters.
This matters unusually strongly here because HiF4 always shares scale metadata
over contiguous groups of 64. The current signed Hadamard operates separately
inside each original group and therefore cannot repair a poor assignment of
channels to groups.

For a permutation `P`, applying the same channel permutation to activation and
weight preserves the full-precision target exactly:

```text
(A P) (W P)^T = A P P^T W^T = A W^T.
```

It changes only which channels share each HiF4 group. Dynamic inference needs
one `index_select` before the existing multiplier/Hadamard path; no test
activation is compared with `W` and no search is performed online.

### 2. Random orthogonal transforms have meaningful variance

SpinQuant reports large variation across random rotations and learns an
orthogonal basis using the quantized loss. This explains why selecting only
eight sign patterns inside fixed blocks can saturate. A learned 64x64 block
rotation, or a sparse sequence of butterflies, is a plausible later branch.
It is not the first v40 implementation because storing and applying dense
cross-block matrices can consume the remaining runtime margin.

### 3. Calibration fit is not enough

AdaRound, GPTQ, BRECQ, OmniQuant, and AffineQuant all optimize reconstruction
or output loss, but the v17/v18/v29/v33-v39 evidence shows that additional
capacity can improve local output without proportional hidden transfer.
The new mechanism must therefore be deterministic from distribution features,
preserve the exact FP output, and pass both nearest- and far-donor panels. It
must not add another unconstrained output-error sweep merely because the
calibration objective permits it.

## Candidate DAG

```text
v39 (fixed original 64-channel layout)
  ├─ v40-a: RPTQ/QuaRot-inspired cross-block permutation
  │    ├─ reject direct similar-range clustering under Hadamard-64
  │    ├─ spread joint A/W sensitivity quantiles across blocks
  │    └─ identity fallback beyond 2:1 compression
  ├─ v40-b: learned 64x64 orthogonal basis (SpinQuant subset; held)
  ├─ v40-c: sparse butterfly/Givens affine basis (held)
  └─ rejected for now
       ├─ more Q(A) coordinate sweeps (v33-v39 saturation)
       ├─ second Q(W) solve (v31/v38 negative cost-benefit)
       ├─ mixed precision/outlier side table (invalid return format)
       └─ additive/lattice codebooks (invalid HiF4 format)
```

## v40-a acceptance criteria

1. Verify FP32 equivalence of the paired permutation before quantization.
2. Change only Linear; Attention must remain identical to v39.
3. Improve aggregate Linear score on both nearest and far donor panels, with
   no structural family showing a material regression.
4. Use a calibration-statistic-only permutation. Do not select it from donor
   test errors and do not compare a dynamic test activation with `W`.
5. Keep state growth to one input-channel index tensor and forecast total
   online runtime below 270 seconds.
6. If unconditional ordering fails, allow only an A/W-calibration output gate
   between identity and the fixed permutation; never perform online search.
7. Pass `py_compile`, `git diff --check`, and all 1100 interface/format checks
   before archiving and pushing v40.

## Primary sources

- GPTQ: https://arxiv.org/abs/2210.17323
- SmoothQuant: https://arxiv.org/abs/2211.10438
- AWQ: https://arxiv.org/abs/2306.00978
- OmniQuant: https://arxiv.org/abs/2308.13137
- RPTQ: https://arxiv.org/abs/2304.01089
- QuaRot: https://arxiv.org/abs/2404.00456
- SpinQuant: https://arxiv.org/abs/2405.16406
- AffineQuant: https://arxiv.org/abs/2403.12544
- AdaRound: https://arxiv.org/abs/2004.10568
- BRECQ: https://arxiv.org/abs/2102.05426
- HQQ: https://arxiv.org/abs/2309.15531
- Outlier Suppression+: https://arxiv.org/abs/2304.09145
- QuIP: https://arxiv.org/abs/2307.13304
- QuIP#: https://arxiv.org/abs/2402.04396
- AQLM: https://arxiv.org/abs/2401.06118
- SpQR: https://arxiv.org/abs/2306.03078
