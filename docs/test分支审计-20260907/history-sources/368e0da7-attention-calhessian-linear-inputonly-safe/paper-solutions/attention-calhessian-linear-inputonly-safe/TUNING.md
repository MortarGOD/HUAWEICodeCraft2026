# Input-isolated alpha-search V-Hessian checkpoint

## Algorithm

- Linear uses calibration-only SmoothQuant/FWHT/one reflector, exact HiF4
  hierarchical search, calibration-activation Hessian weight quantization, four
  mantissa coordinate rounds, two rounds of discrepancy pair escape, gated
  `lv3` subtree flips, and a short CD/pair recovery.
- Linear selects the smooth exponent per layer from `(0.16, 0.24)` for extreme
  calibration dispersion and `(0.35, 0.50)` otherwise. The lower value is the
  anchor and the second route must improve mean calibration output loss. Dynamic
  input scale search uses E6M2 rank radius 3; radius 6 was exactly score-neutral.
- V stores the calibration mean `E[P^T P]`, shrinks it as `0.7 H + 0.3 I`,
  and applies four token-coordinate rounds, one two-token pair escape, one
  `lv3` subtree pass, then a short CD/pair recovery.
- Q and K use the input-only hierarchical route. Their output-Hessian paths
  were score-neutral and were removed from calibration and State.

## Compliance boundary

- Calibration may use reference Linear/Attention outputs to fit frozen State.
- Linear dynamic quantization reads only the current activation and frozen
  invertible-transform State. The fitted diagonal balance is a transform
  parameter, not a weight tensor or weight reconstruction. Its main route is
  unweighted. No `W`, `Q(W)`,
  weight energy, weight error, correction matrix, or weight factor is stored in
  `activation_state`. It does not evaluate current `A @ W`, `A @ Q(W)`, or a
  weight-factor projection of current error.
- The fallback route may store scalar importance computed only from calibration
  activations. It never derives dynamic importance from `W` or `Q(W)`.
- Q, K, and V dynamic calls are isolated. V reads only current V and its frozen
  token Hessian; it never reads current Q, K, softmax, or attention output.
- The submission archive contains only `solution.py`.

## Current-dataset result

- Dataset: `qwen3_8b_profiled_v1`
- Dataset digest: `4d17a2de957808aece18e022f98937b694a4144fbefefcd3caab532ba2b8b78e`
- Config digest: `1a4b9d1e208e177dd386bdf3070cb944879d8dd7f883ab3a8464591e13e95328`
- Status: `valid`
- Total: `340.6911726240306`
- Linear: `199.2916680102964`
- Attention: `141.39950461373425`
- Worker: `43.295790798962116s` in the final artifact-matched run.
- Submission ZIP SHA256: `61a6393c548899fa7de11aee60c6abb2154474d7337d885ed7eb70863b52991e`.

The prior safe online-refresh local baseline was `277.3851542785867`, so this
checkpoint is `+22.82%` on the current local dataset. This is not an online
score and does not prove the 30000 target.

## Strict interface-isolation result

- Dataset: `qwen2_5_14b_online_profiled_v2`
- Status: `valid`
- Total: `359.38917065029443`
- Linear: `197.98415511320977`
- Attention: `161.40501553708467`
- Worker: `49.25814639776945s`.
- Fresh-interface, all call-order permutations, and cross-sample checks passed.

## Rejected ablations

- Per-layer V damping selection overfit calibration and reduced the current
  dataset total to `333.78795395793225`.
- Variance-estimated token-Hessian shrinkage reduced it to `331.07813870771724`.
- Independent per-block Walsh target search reduced it to `334.64325931953715`.
- V E6M2 scale reselection and wider pair searches added time without score.
- Three alpha candidates scored `340.4344870126703` at `58.44790060073137s`;
  the two-candidate low-anchor policy is both faster and better. Shifting the
  pair to `(0.30, 0.45)` or `(0.35, 0.45)` reduced score.
- Exact calibration Gram reduced total to `325.2297778129326`; covariance
  shrinkage is required for held-out stability.
- Direct `Q(W)`-energy dynamic importance scored only `0.17` higher locally but
  is excluded under the stricter cross-interface rule.
