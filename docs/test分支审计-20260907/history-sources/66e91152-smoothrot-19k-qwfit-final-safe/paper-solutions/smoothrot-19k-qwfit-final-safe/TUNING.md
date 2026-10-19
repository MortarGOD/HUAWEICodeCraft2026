# Tuning Record

Dataset: `qwen2_5_14b_online_profiled_v2`.

## Retained

- weight radius `4`: Linear `194.127557218522` with residual blend `0.035`
- weight radius `8`: Linear `195.627853392863`
- weight radius `12`: Linear `195.983408183216`
- weight radius `16/24/32`: Linear `195.850640252715`, showing saturation
- radius-12 learning rate `0.075`: best of `0.06..0.09`

## Rejected

- global paper q/alpha transfer: large Linear regressions
- Q/K balance exponents other than `0.25`: Attention regressions
- wider Attention and Linear offset windows: regressions
- per-block fixed offsets fitted with Q(W): held-out calibration overfit
- input-statistic offset predictor: rejected by calibration gate; forced use
  regressed Linear
- deployed-radius STE proxy: regressed Linear
- deployed-chain learned-scale gate: accepted extra groups but regressed Linear
- Q(W) residual-importance blend at radius 12: no improvement

Local scores are development measurements and do not prove the online score.
