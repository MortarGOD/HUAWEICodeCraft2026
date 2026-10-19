# SmoothRot Learned Scale Manual-STE Baseline

This candidate combines the strongest strict local HiF4 baseline with a
calibration-only learned SmoothRot scale update for Linear groups.

## Linear Method

1. Build the existing LET scale and block-64 Hadamard/reflector transform.
2. Fit one log-scale update from the first three calibration samples.
3. Share each learned residual across 16 source channels and remove its mean
   within each 64-channel HiF4 block.
4. Compute the one-step STE gradient explicitly with matrix chain rules. This
   is equivalent to the first zero-initialized Adam step and avoids autograd.
5. Clamp log2 scales to `[-3, 3]` and freeze the selected scale for dynamic
   activation quantization.
6. Keep the learned route only when exact HiF4 calibration probes show a
   moderate full-calibration gain and a robust gain on the final two held-out
   calibration samples.

The calibration function may use reference `A @ W.T` outputs. Dynamic calls
use only the current activation and frozen calibration state; they do not
access the weight, compute `A @ W`, or share mutable state across interfaces.

Attention remains the strict strong baseline route.

## Local Result

Dataset: `qwen2_5_14b_online_profiled_v2`

- Status: `valid`, 500/500 cases.
- Total: `354.6418925485317`.
- Linear: `193.86075122216846`.
- Attention: `160.78114132636324`.
- Worker: `33.77998059988022s`.
- Fresh-interface, call-order, and cross-sample isolation: passed.

The corresponding unlearned strong baseline scored `353.6763254878113`
with a `28.782959900796413s` worker run in the same environment.
