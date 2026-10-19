# Rejected learned per-head Attention rotation

This experiment preserves corrected r2qk6 Linear and adds an interface-safe,
calibration-fitted Q/K rotation. It is archived as a negative result and must
not be submitted as a score candidate.

## Algorithm

For every KV head and 16-channel block, calibration combines the Q and K
covariances, computes their eigenvectors, and maps the eigenbasis through a
normalized Hadamard matrix. Q heads sharing one KV head receive the same
orthogonal matrix:

```text
Q' = Q R
K' = K R
Q' K'^T = Q K^T
```

The matrices are serialized independently in q_state and k_state. Dynamic Q,
K, and V do not share module globals or runtime caches. A route requires at
least 0.5% mean calibration gain and no sample worse than -5%.

## Result

Only one of ten Qwen3.5 Attention groups passed the calibration gate. Its mean
calibration gain was about `+0.99%`, with a worst sample at `-4.27%`. On the
independent test split, weighted Attention fell from corrected baseline
`44.3483` to `42.8226` (`-1.5257`); Linear remained byte-equivalent at
`230.6642`. All 210 cases were structurally valid.

The rank-1 Householder precursor was even weaker: all ten groups had negative
mean calibration gain. These results reject covariance-flattening rotations as
the next online direction. No submission ZIP is produced.
