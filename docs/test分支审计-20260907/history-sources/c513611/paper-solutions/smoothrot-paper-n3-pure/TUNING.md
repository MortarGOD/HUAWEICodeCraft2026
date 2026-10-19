# Pure Paper Parameter Tuning

Dataset: `qwen2_5_14b_online_profiled_v2`

Only the two parameters from the paper scale formula were tuned. No guards,
fallback routes, importance weights, fitted corrections, or non-paper
transforms were added.

## Search

- Coarse alpha search at `q=0.994`: `0.0` through `0.475`.
- Fine alpha search: `0.08` through `0.14`.
- Quantile search: `0.95`, `0.97`, `0.975`, `0.98`, `0.985`, `0.99`,
  `0.994`, `0.997`, and `1.0`.
- Final local grid: alpha `0.096` through `0.104`, q `0.976` through
  `0.984`.

## Selected Point

- `alpha = 0.100`
- `q = 0.977`
- Linear sweep score: `73.189747121028`

The paper replay point `(0.475, 0.994)` scored approximately
`-1.222e10` on this dataset. The much smaller selected alpha is necessary
because each local calibration group contains only ten activation rows;
channels absent from calibration otherwise produce extreme reciprocal scales.

The sweep utility is `sweep_paper_params.py`. It is a local research tool and
is not included in `submission.zip`.
