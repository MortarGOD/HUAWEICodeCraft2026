"""Constant Q correction from the final deployed output residual.

Calibration only: solve a damped Gauss-Newton problem in a restricted shared
shift subspace. This is not full per-token residual reconstruction.
"""
_rb_parent_calibration = hif4_calibration_attention
RB_DEBUG = {}


def _rb_apply_shift(x, state, nh, d):
    if isinstance(state, dict) and 'rb_shift' in state:
        shift = state['rb_shift'].float().repeat_interleave(nh // state['kvh'], 0)
        return (x.reshape(-1, nh, d) + shift[None]).reshape(x.shape).contiguous()
    return x


def _rb_moments(query, key, value, target, d):
    p = torch.softmax(query @ key.T / d**.5, -1)
    out = p @ value
    h = torch.zeros(d, d)
    g = torch.zeros(d)
    for i in range(query.shape[0]):
        jac = key.T @ (p[i, :, None] * (value - out[i])) / d**.5
        h += jac @ jac.T
        g += jac @ (out[i] - target[i])
    return h / query.shape[0], g / query.shape[0]


def _rb_solve(h, g, query_norm):
    if not bool(torch.isfinite(h).all() and torch.isfinite(g).all() and torch.isfinite(query_norm).all()):
        return None
    scale = h.diagonal(dim1=-2, dim2=-1).mean(-1)
    damped = h + (.1 * scale.clamp_min(1e-12))[:, None, None] * torch.eye(h.shape[-1])
    step = -torch.linalg.solve(damped, g.unsqueeze(-1)).squeeze(-1)
    radius = .1 * query_norm
    step *= (radius / step.norm(dim=-1).clamp_min(1e-20)).clamp_max(1)[:, None]
    step *= .5
    return step if bool(torch.isfinite(step).all()) else None


def _rb_observe(sample, states, qh, kh, d, count):
    n = sample['k'][0].shape[0]
    ix = torch.linspace(0, n - 1, min(n, count)).long()
    raw = [dequantize_nvfp4(sample['q'][0][ix], sample['q'][1][ix]).float()]
    raw += [dequantize_nvfp4(*sample[r]).float() for r in 'kv']
    target = _joint_output(*raw, qh, kh, d, ix, query_subset=True)
    query = _joint_quant_queries(sample, states['q_state'], qh, kh, d, ix).float()
    key = _deq_params(hif4_dynamic_quantize_k(*sample['k'], kh, d, states['k_state'])).float()
    value = _deq_params(hif4_dynamic_quantize_v(*sample['v'], kh, d, states['v_state'])).float()
    return ix, target, query, key, value


@torch.inference_mode()
def hif4_calibration_attention(samples, qh, kh, d):
    parent = _rb_parent_calibration(samples, qh, kh, d)
    RB_DEBUG.clear(); RB_DEBUG.update(accepted=0, error=False, folds=0)
    if len(samples) < 3 or d % 64 or not parent['q_state'].get('gq') or 'u' not in parent['q_state']:
        return parent
    selected = samples[:-1][-2:]
    if not all(4 <= s['k'][0].shape[0] <= 4096 for s in list(selected) + [samples[-1]]):
        return parent
    try:
        hs, gs, norms = [], [], []
        for sample in selected:
            ix, target, q, k, v = _rb_observe(sample, parent, qh, kh, d, 8)
            q = q.reshape(-1, kh, qh // kh, d)
            k = k.reshape(-1, kh, d); v = v.reshape(-1, kh, d)
            moments = [_rb_moments(q[:, j].reshape(-1, d), k[:, j], v[:, j],
                                  target[:, j].reshape(-1, d), d) for j in range(kh)]
            hs.append(torch.stack([m[0] for m in moments]))
            gs.append(torch.stack([m[1] for m in moments]))
            norms.append(q.square().sum(-1).mean((0, 2)))
            RB_DEBUG['folds'] += 1
        shift = _rb_solve(torch.stack(hs).mean(0), torch.stack(gs).mean(0),
                          torch.stack(norms).mean(0).sqrt())
        if shift is None:
            RB_DEBUG['nonfinite'] = True
            return parent
        hold = samples[-1]
        ix, target, q, k, v = _rb_observe(hold, parent, qh, kh, d, 16)
        trial = dict(parent['q_state'], rb_shift=shift.cpu().contiguous())
        newq = _joint_quant_queries(hold, trial, qh, kh, d, ix)
        def cost(query):
            err = (_joint_output(query, k, v, qh, kh, d, ix, query_subset=True) - target).square().mean((-1, -2))
            return torch.stack([err[::2].mean(0), err[1::2].mean(0)])
        before, after = cost(q), cost(newq)
        take = (after < before * .99).all(0) & torch.isfinite(after).all(0)
        RB_DEBUG.update(accepted=int(take.sum()), ratios=(after.mean(0) / before.mean(0).clamp_min(1e-30)).tolist(),
                        shift_norm=shift.norm(dim=-1).tolist())
        if not bool(take.any()):
            return parent
        trial['rb_shift'] = (shift * take[:, None]).cpu().contiguous()
        return dict(parent, q_state=trial)
    except (RuntimeError, ValueError) as exc:
        RB_DEBUG.update(error=True, error_type=type(exc).__name__)
        return parent
