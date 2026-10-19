# SmoothRot Quantile N3 Pure Baseline

Paper: Patrik Czakó, Gábor Kertész, and Sándor Szénási,
"Trainable Smooth-Rotation Transforms with Learned Channel Scales for LLM
Quantization," 2026, arXiv:2606.09927.

Source: https://arxiv.org/abs/2606.09927

## Implemented Policy

This directory keeps the paper's no-training scale formula and tunes only its
two scalar hyperparameters for the local HiF4 dataset:

- locally tuned `alpha = 0.100`
- locally tuned activation quantile `q = 0.977`
- `s_j = Q_q(|X_j|)^alpha / max(|W_j|)^(1-alpha)`
- activation transform: `(X / s) H`
- weight transform in repository layout: `(W * s) H`
- 64-wide normalized Hadamard blocks
- basic HiF4 RTN for transformed weights and activations

The activation and weight transforms are inverse-equivalent before
quantization. Calibration computes and freezes `s`; the dynamic Linear call
uses only the current activation and the frozen reciprocal scale. It does not
access the weight, `A @ W`, output targets, or another dynamic interface.

Attention is outside the paper's down-projection scope and uses basic HiF4 RTN
without RRS, fitted routes, or output-aware logic.

## Deliberately Excluded

- PSOT, GPTQ/Hessian objectives, importance weighting, residual correction,
  route gating, RRS, and Attention fitting.
- The paper's T1/T2 STE training. That is kept separate because this directory
  is the clean no-training baseline and the paper's 800-step schedule does not
  fit the 30-second judging budget.

The original paper evaluates symmetric INT4 on LLaMA-3.2-1B. This challenge
uses HiF4 on Qwen traces, so paper-reported gains are not assumed to transfer.

## Local Result

Dataset: `qwen2_5_14b_online_profiled_v2`

- Status: `valid`, 500/500 cases completed.
- Total and Linear score: `73.18974712102808`.
- Attention score: `0.0` because the paper leaves Attention unchanged and this
  baseline uses the challenge's basic RTN path there.
- Fresh-interface, call-order, and cross-sample isolation: passed.

Worker time is `20.2143s`. This remains a paper-reproduction control, not a
recommended score submission. Parameter tuning removes the catastrophic
failure of the paper replay point `(0.475, 0.994)`, but the pure formula still
underperforms the format-adapted HiF4 baselines. See `TUNING.md` for the search
grid and selected point.
