"""v391: conditional token-metric integer descent from the full v387 V."""
_KI_ROUNDS = 4
KI_DEBUG = {}
_ki_parent_calibration = hif4_calibration_attention
_ki_parent_v = hif4_dynamic_quantize_v


def _ki_fit(samples, qh, kh, d):
    active = torch.ones(kh, dtype=torch.bool)
    for sample in samples:
        k = dequantize_nvfp4(*sample['k']).float().reshape(-1, kh, d)
        v = dequantize_nvfp4(*sample['v']).float().reshape(-1, kh, d)
        active &= ((k == v) & torch.isfinite(v)).all(0).all(-1)
        active &= v.square().sum((0, 2)) > 0
    if not bool(active.any()):
        return None
    probes = []
    for sample in samples[max(0, len(samples)-3):-1]:
        q = dequantize_nvfp4(*sample['q']).float()
        ix = torch.linspace(0, q.shape[0]-1, min(4, q.shape[0])).long()
        probes.append(q[ix].reshape(-1, kh, qh//kh, d).permute(1, 0, 2, 3).reshape(kh, -1, d))
    if not probes:
        return None
    return dict(ki_probes=torch.cat(probes, 1).contiguous(), ki_active=active)


def _ki_descent(x, y, step, p):
    # Channels are independent; each round moves one token per channel.
    t = x.shape[0]
    penalty = 1.0 / (t*t)
    initial_sse = (x-y).square().sum(0)
    cap = initial_sse * 1.05
    h = p.square().mean(0)[:, None] + penalty
    for _ in range(_KI_ROUNDS):
        e = x-y
        g = p.transpose(0, 1) @ (p @ e) / p.shape[0] + penalty*e
        budget = (cap-e.square().sum(0)).clamp_min(0)
        radius = (e.square()+budget[None]).sqrt()
        old = torch.round(y/step).clamp(-7, 7)
        lo = torch.minimum(torch.ceil((x-radius)/step).clamp(-7, 7), old)
        hi = torch.maximum(torch.floor((x+radius)/step).clamp(-7, 7), old)
        code = torch.maximum(lo, torch.minimum(hi, torch.round((y+g/h)/step)))
        delta = code*step-y
        change = h*delta.square()-2*g*delta
        value, index = change.min(0)
        columns = torch.arange(x.shape[1])
        y = y.clone()
        y[index, columns] += torch.where(value < 0, delta[index, columns], 0.)
    return y


def _ki_quant(x, base, st, kh, d):
    original = _deq_params(base).float()
    y = original.clone()
    step = _params_unit_flat(base).reshape_as(x)*.25
    probabilities = {}
    for head in torch.where(st['ki_active'])[0].tolist():
        sl = slice(head*d, (head+1)*d)
        p = torch.softmax(st['ki_probes'][head].float() @ x[:, sl].T / d**.5, -1)
        probabilities[head] = p
        y[:, sl] = _ki_descent(x[:, sl], y[:, sl], step[:, sl], p)
    result = _values_to_params(y, base)
    actual = _deq_params(result).float()
    keep = torch.zeros(x.shape[1], dtype=torch.bool)
    for head, p in probabilities.items():
        sl = slice(head*d, (head+1)*d)
        e0, e1 = x[:, sl]-original[:, sl], x[:, sl]-actual[:, sl]
        mse0, mse1 = e0.square().mean(0), e1.square().mean(0)
        c0 = (p@e0).square().mean(0)+mse0/x.shape[0]
        c1 = (p@e1).square().mean(0)+mse1/x.shape[0]
        keep[sl] = (c1 < c0) & (mse1 <= mse0*1.05)
    # Only codes change; retain all baseline scale fields exactly.
    safe = _values_to_params(torch.where(keep[None], actual, original), base)
    mask = keep.reshape(1, -1, 8, 2, 4)
    return {key: torch.where(mask, safe[key], value) if key in ('sign', 'mant') else value for key, value in base.items()}


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    parent = _ki_parent_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    KI_DEBUG.clear(); KI_DEBUG.update(eligible=0, accepted=0, error=False)
    qh, kh, d = q_num_heads, kv_num_heads, head_dim
    if len(calib_qkv_list)<2 or d%64 or not parent['q_state'].get('gq'):
        return parent
    hold = calib_qkv_list[-1]
    t = hold['v'][0].shape[0]
    if not 4 <= t <= 4096:
        return parent
    try:
        st = _ki_fit(calib_qkv_list, qh, kh, d)
        if st is None:
            return parent
        KI_DEBUG['eligible'] = int(st['ki_active'].sum())
        raw = [dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
        base = _ki_parent_v(*hold['v'], kh, d, parent['v_state'])
        trial = _ki_quant(raw[2], base, st, kh, d)
        ix = torch.linspace(0, t-1, min(16, t)).long()
        target = _joint_output(*raw, qh, kh, d, ix)
        q = _joint_quant_queries(hold, parent['q_state'], qh, kh, d, ix)
        k = _deq_params(hif4_dynamic_quantize_k(*hold['k'], kh, d, parent['k_state']))
        costs = []
        for params in (base, trial):
            error = (_joint_output(q, k, _deq_params(params), qh, kh, d, ix, query_subset=True)-target).square().mean((-1, -2))
            costs.append(torch.stack((error[::2].mean(0), error[1::2].mean(0))))
        active = st['ki_active'] & (costs[1] < costs[0]*.99).all(0)
        KI_DEBUG.update(accepted=int(active.sum()), ratios=(costs[1].mean(0)/costs[0].mean(0).clamp_min(1e-30)).tolist())
        if not bool(active.any()):
            return parent
        st.update(ki_active=active, ki_fallback=parent['v_state'])
        return dict(parent, v_state=st)
    except Exception as exc:
        KI_DEBUG.update(error=True, error_type=type(exc).__name__)
        return parent


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    if not isinstance(v_state, dict) or 'ki_probes' not in v_state:
        return _ki_parent_v(v_quant, v_scale, kv_num_heads, head_dim, v_state)
    base = _ki_parent_v(v_quant, v_scale, kv_num_heads, head_dim, v_state['ki_fallback'])
    if not 4 <= v_quant.shape[0] <= 4096:
        return base
    x = dequantize_nvfp4(v_quant, v_scale).float()
    return _ki_quant(x, base, v_state, kv_num_heads, head_dim)
