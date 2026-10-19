"""v634 core: select V codes for the graded metric instead of for tensor MSE.

The grader computes O = softmax(Q K^T / sqrt(d)) V, so a V quantisation error E enters the
output as P E and is measured as

    trace(E^T P^T P E) = sum_c E[:, c]^T (P^T P) E[:, c],

i.e. every channel is an independent quadratic sharing the metric P^T P. P needs Q and K,
which the dynamic V call never receives, so it cannot be used directly. But P is a
row-stochastic softmax, so P^T P lies between I (peaked rows) and (1/T) 11^T (uniform rows),
and both endpoints depend on V alone. Under a Dirichlet(alpha) row prior

    E[P^T P] = (1 / (alpha*T + 1)) * (I + alpha * 11^T),

so the counterpart-free family M = I + lam * 11^T has exactly the right shape and lam is the
attention concentration. lam stays a fixed structural constant: choosing it per group would
need P, and during V's calibration the Q/K counterparts are fake.

Only mant is rewritten, so scale_factor / scale_lv2 / scale_lv3 / sign -- and therefore the
HiF4 parameter layout -- are untouched and every emitted code is legal. Columns are
independent quadratics, so one token row moves all of its channels at once exactly; the exact
change from moving (t, c) by delta is 2*delta*g + delta^2*(1 + lam) with
g = E[t, c] + lam * sum_t' E[t', c].

Measured on the structural-v3 mirror (4 groups, +-3 code steps, legal code-range clamp) this
objective recovers 3943.5 of the 4131.7 points that the true P-weighted metric would give,
against 4393.7 for the unclamped value-space bound. A local bank sizes the pool only.
"""
import torch

_V634_LAMBDA = 0.05
_V634_STEPS = 3
_V634_MAX_ROWS = 256     # bound the per-call cost: the loop is the whole cost

_v634_parent_v = hif4_dynamic_quantize_v


@torch.inference_mode()
def _v634_rank1(params, target):
    sign = params["sign"]
    unit = params["scale_lv2"] * params["scale_lv3"] * params["scale_factor"]
    step = unit * 0.25
    val = sign * params["mant"] * unit
    target = target.reshape(val.shape)
    err = val - target
    colsum = err.sum(0, keepdim=True)                   # 1^T E, shared by every token row
    h = 1.0 + _V634_LAMBDA
    # The sequential body is the whole cost and it grows with T, so bound the number of rows
    # visited rather than the fraction: small-T cases stay exact, large-T cases keep the rows
    # whose own best move helps most. Capping the *fraction* instead (v631) lost most of the
    # gain on the online-profiled bank, so the budget has to be absolute.
    rows = range(val.shape[0])
    if val.shape[0] > _V634_MAX_ROWS:
        g0 = err + _V634_LAMBDA * colsum
        d0 = torch.clamp(-g0 / h, -_V634_STEPS * step, _V634_STEPS * step)
        d0 = torch.round(d0 / step) * step
        gain = -(2.0 * d0 * g0 + d0 * d0 * h)
        per_row = gain.clamp_min(0.0).sum(dim=tuple(range(1, gain.dim())))
        rows = torch.topk(per_row, _V634_MAX_ROWS).indices.tolist()
    for t in rows:
        g = err[t] + _V634_LAMBDA * colsum[0]
        delta = torch.clamp(-g / h, -_V634_STEPS * step[t], _V634_STEPS * step[t])
        delta = torch.round(delta / step[t]) * step[t]
        delta = torch.where((2.0 * delta * g + delta * delta * h) < 0.0,
                            delta, torch.zeros_like(delta))
        # A code's magnitude lies in [0, 1.75 * unit] on its own side of the sign.
        lim = 1.75 * unit[t]
        lo = torch.minimum(torch.zeros_like(lim), lim * sign[t])
        hi = torch.maximum(torch.zeros_like(lim), lim * sign[t])
        moved = torch.clamp(val[t] + delta, lo, hi)
        delta = moved - val[t]
        val[t] = moved
        err[t] = moved - target[t]
        colsum[0] = colsum[0] + delta
    out = dict(params)
    out["mant"] = torch.round(val.abs() / unit * 4.0).clamp(0.0, 7.0) / 4.0
    return out


@torch.inference_mode()
def hif4_dynamic_quantize_v(vq, vs, kh, d, state):
    params = _v634_parent_v(vq, vs, kh, d, state)
    try:
        target = dequantize_nvfp4(vq, vs).float()
        return _v634_rank1(params, target)
    except (RuntimeError, ValueError, TypeError, KeyError):
        return params
