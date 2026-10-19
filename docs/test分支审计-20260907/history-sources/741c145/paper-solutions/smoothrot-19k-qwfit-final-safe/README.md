# SmoothRot Q(W)-Fitted Q(A) Candidate

This candidate keeps all output-aware fitting in calibration and freezes a
self-contained policy for each dynamic interface.

## Linear

- Initialize learned channel scales from a half-strength LET balance.
- Apply two explicit manual-STE updates to zero-mean log2 scale residuals shared
  across 16 source channels.
- Use a block-64 Hadamard transform and a calibration-selected low-rank
  reflector. Weight and activation transforms are exact inverses before
  quantization.
- Quantize `W` with a radius-12 scale search. Derive frozen activation
  importance from `Q(W)` column energy.
- During calibration, evaluate three seven-offset Q(A) scale windows by the
  actual `Q(A) @ Q(W).T` output loss. Store only the selected offset tuple and
  importance vector in activation state.
- During dynamic calls, use the current A and frozen state only. The code does
  not retain W, calculate the current `A @ W`, or project current error through
  a weight-derived output basis.

## Attention

Q and K use reciprocal calibrated channel scales before the Hadamard transform,
so their unquantized logits are unchanged. Q/K/V states remain independent and
do not communicate through module globals or caches.

## Selected Parameters

- weight scale radius: `12`
- dynamic Linear offset-window width: `7`
- learned-scale warm-start blend: `0.5`
- manual-STE steps: `2`
- manual-STE learning rate: `0.075`
- channel sharing width: `16`
- Attention balance exponent: `0.25`

The final local report is `profiled-v2-final-strict.json`; the submission
artifact is `submission-qwfit-final.zip`.

## Strict Local Result

- status: `valid`
- Linear: `195.9834081832158` (`250/250`)
- Attention: `160.78114132636324` (`250/250`)
- total: `356.76454950957907`
- worker time: `47.18588409945369s`
- fresh-interface, call-order, and cross-sample isolation: passed
- artifact SHA-256: `e8ea09af5c67d9c33ca5082e841c3199360396dbebd082ab9a8825c7a6863514`

Multiplying the local aggregate by 50 gives `17838.2275` as a display-scale
diagnostic only. It is not an online result and does not meet the 19k target.
