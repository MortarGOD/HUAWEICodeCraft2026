# Tuning Record

## Selected Parameters

- scale sharing width: `16`
- one-step learning rate: `0.060`
- reliable source group: at least 8 of 16 channels observed twice
- log2 scale limit: `3.0`
- full-calibration gain gate: `[0.03, 0.25]`
- held-out mean gain gate: `>= 0.10`
- held-out worst gain gate: `>= -0.02`

## Learning Rate Sweep

Linear scores with width 16:

- `0.03`: `192.832338908753`
- `0.04`: `192.724233789201`
- `0.05`: `192.544077902423`
- `0.058`: `193.557605198576`
- `0.059`: `193.391250826324`
- `0.060`: `193.860751222168`
- `0.061`: `193.706892884249`
- `0.062`: `193.405812174364`
- `0.07`: `192.582761175042`
- `0.08`: `191.106995765356`

The sharp optimum is expected because the HiF4 hierarchy changes at discrete
scale boundaries.

## Sharing Width Sweep

At learning rate 0.060:

- width 8: `191.629038360614`
- width 16: `193.860751222168`
- width 32: `193.042099033501`

The local sweep utility is `sweep_learning_rate.py`; it is not included in the
submission ZIP.
