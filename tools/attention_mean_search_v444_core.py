"""Direct V mean-plus-SSE discrete risk; no conditional probability fitting."""

def _v444_fit(samples, state, query_heads, key_heads, head_dim):
    return {'active': torch.ones(key_heads, dtype=torch.bool), 'query_heads': query_heads}


def _v444_terms(error):
    tokens = error.shape[1]
    gradient = (error.sum(1, keepdim=True) + error) / tokens ** 2
    curvature = torch.full(error.shape[:2], 2.0 / tokens ** 2)
    return gradient, curvature


def _v444_refine(value, params, model, key_heads, head_dim):
    tokens = value.shape[0]
    query_heads = model['query_heads']
    if not 2 <= tokens <= 4096 or tokens * query_heads * head_dim > 16777216:
        return params
    active = torch.nonzero(model['active'], as_tuple=False).flatten()
    if active.numel() == 0:
        return params
    reconstructed = _deq_params(params).reshape(tokens, key_heads, head_dim).transpose(0, 1).contiguous()
    target = value.reshape(tokens, key_heads, head_dim).transpose(0, 1)[active].contiguous()
    step = (_params_unit_flat(params) * .25).reshape(tokens, key_heads, head_dim).transpose(0, 1)[active].contiguous()
    best = reconstructed[active].clone()
    error = target - best
    squared_error = error.square().sum(1)
    budget = squared_error.clamp_min(1e-30) * 1.05
    for _ in range(16):
        gradient, curvature = _v444_terms(error)
        codes = torch.round(best / step).clamp(-7, 7)
        radius = (budget[:, None] - squared_error[:, None] + error.square()).clamp_min(0).sqrt()
        lower = torch.ceil((target - radius) / step).clamp(-7, 7)
        upper = torch.floor((target + radius) / step).clamp(-7, 7)
        proposed_codes = torch.round(codes + gradient / (curvature[:, :, None] * step))
        proposed_codes = torch.maximum(lower, torch.minimum(upper, proposed_codes))
        proposed = proposed_codes * step
        delta = proposed - best
        increase = delta.square() - 2 * delta * error
        feasible = squared_error[:, None] + increase <= budget[:, None]
        cost = delta.square() * curvature[:, :, None] - 2 * delta * gradient
        cost = torch.where(feasible, cost, torch.full_like(cost, float('inf')))
        improvement, index = cost.min(1, keepdim=True)
        selected = torch.gather(delta, 1, index)
        selected = torch.where(improvement < 0, selected, torch.zeros_like(selected))
        if not bool(selected.any()):
            break
        best.scatter_add_(1, index, selected)
        error = target - best
        squared_error = error.square().sum(1)
    result = reconstructed.clone()
    result[active] = best
    packed = _values_to_params(result.transpose(0, 1).reshape(value.shape).contiguous(), params)
    decoded = _deq_params(packed).reshape(tokens, key_heads, head_dim).transpose(0, 1)[active]
    before = target - reconstructed[active]
    after = target - decoded
    g0, _ = _v444_terms(before)
    g1, _ = _v444_terms(after)
    c0 = (before.double() * g0.double()).sum(1)
    c1 = (after.double() * g1.double()).sum(1)
    valid = torch.isfinite(c1) & (c1 < c0)
    valid &= after.double().square().sum(1) <= before.double().square().sum(1).clamp_min(1e-30) * 1.05
    keep = torch.zeros(key_heads, head_dim, dtype=torch.bool)
    keep[active] = valid
    keep = keep[None].expand(tokens, -1, -1).reshape_as(params['mant'])
    answer = dict(params)
    for name in ('sign', 'mant'):
        answer[name] = torch.where(keep, packed[name], params[name]).contiguous()
    return answer

def _v444_calibrate(samples, parent, query_heads, key_heads, head_dim):
    if len(samples) < 3 or head_dim % 64:
        return parent
    holdouts = [i for i, sample in enumerate(samples) if 4 <= sample['v'][0].shape[0] <= 4096]
    if not holdouts:
        return parent
    holdout = holdouts[-1]
    model_samples = [sample for i, sample in enumerate(samples) if i != holdout] + [samples[holdout]]
    try:
        model = _v444_fit(model_samples, parent['v_state'], query_heads, key_heads, head_dim)
        if model is None:
            return parent
        sample = samples[holdout]
        value = dequantize_nvfp4(*sample['v']).float()
        baseline = _v444_parent_dynamic(*sample['v'], key_heads, head_dim, parent['v_state'])
        candidate = _v444_refine(value, baseline, model, key_heads, head_dim)
        if torch.equal(_deq_params(baseline), _deq_params(candidate)):
            return parent
        indices = torch.linspace(0, value.shape[0] - 1, min(64, value.shape[0])).long()
        if parent['q_state'].get('gq') and 'u' in parent['q_state']:
            query = _joint_quant_queries(sample, parent['q_state'], query_heads, key_heads, head_dim, indices)
        else:
            query = _deq_params(hif4_dynamic_quantize_q(
                *sample['q'], query_heads, head_dim, parent['q_state']))[indices]
        key = _deq_params(hif4_dynamic_quantize_k(*sample['k'], key_heads, head_dim, parent['k_state']))
        reference = _joint_output(
            dequantize_nvfp4(*sample['q']).float()[indices], dequantize_nvfp4(*sample['k']).float(),
            value, query_heads, key_heads, head_dim, indices, query_subset=True)
        losses = []
        for params in (baseline, candidate):
            output = _joint_output(query, key, _deq_params(params), query_heads, key_heads,
                                   head_dim, indices, query_subset=True)
            loss = (output - reference).square().mean((-1, -2))
            losses.append(torch.stack((loss[::2].mean(0), loss[1::2].mean(0))))
        model['active'] = (losses[1] < losses[0] * .99).all(0)
        if not bool(model['active'].any()):
            return parent
        state = dict(parent['v_state']) if isinstance(parent['v_state'], dict) else {}
        state['v444_key_risk'] = model
        return dict(parent, v_state=state)
    except (RuntimeError, ValueError):
        return parent

_v444_parent_calibration = hif4_calibration_attention
_v444_parent_dynamic = hif4_dynamic_quantize_v


@torch.inference_mode()
def hif4_calibration_attention(samples, qh, kh, d):
    parent = _v444_parent_calibration(samples, qh, kh, d)
    return _v444_calibrate(samples, parent, qh, kh, d)


@torch.inference_mode()
def hif4_dynamic_quantize_v(quant, scale, kh, d, state):
    params = _v444_parent_dynamic(quant, scale, kh, d, state)
    if not isinstance(state, dict) or 'v444_key_risk' not in state:
        return params
    try:
        value = dequantize_nvfp4(quant, scale).float()
        return _v444_refine(value, params, state['v444_key_risk'], kh, d)
    except (RuntimeError, ValueError):
        return params

