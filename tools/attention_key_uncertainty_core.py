"""Adapted from Mrxxx7/test V8; pinned source and changes in v441 report."""

def _v441_fit(samples, state, query_heads, key_heads, head_dim):
    training = samples[:-1]
    rows = sum(min(256, sample['v'][0].shape[0]) for sample in training)
    channels = key_heads * head_dim
    if len(training) < 2 or rows < 96 or rows > 1024:
        return None
    if rows * channels * channels > 536870912:
        return None
    if rows * 65 * query_heads * head_dim > 268435456:
        return None
    rank = min(64, head_dim)
    covariance_work = key_heads * head_dim ** 2 * (rows + head_dim)
    projection_work = query_heads * head_dim * rank * (rows + 65)
    metric_work = query_heads * 65 ** 2 * (rank + head_dim)
    if covariance_work + projection_work + metric_work > 536870912:
        return None
    if isinstance(state, dict) and 'v435_content_parent' in state:
        state = state['v435_content_parent']
    reuse_model = isinstance(state, dict) and 'kernel' in state
    if not reuse_model and channels ** 2 * (rows + channels) > 536870912:
        return None
    model = dict(state) if reuse_model else _vk_fit(
        samples, query_heads, key_heads, head_dim)
    arrays = {role: [] for role in 'qkv'}
    groups = []
    position = 0
    for sample in training:
        count = sample['v'][0].shape[0]
        indices = torch.linspace(0, count - 1, min(256, count)).long()
        groups.append((position, position + indices.numel()))
        position += indices.numel()
        for role in arrays:
            arrays[role].append(dequantize_nvfp4(*sample[role]).float()[indices])
    query, key, value = (torch.cat(arrays[role]) for role in 'qkv')
    features = _vk_features(value, model)
    identity = torch.eye(features.shape[1])
    gram = features.T @ features
    targets = torch.cat((query, key), 1)
    cross = features.T @ targets
    coefficients = torch.cholesky_solve(
        cross, torch.linalg.cholesky(gram + identity * (0.03 * rows)))
    residual = torch.empty_like(targets)
    for start, stop in groups:
        part = features[start:stop]
        train_gram = gram - part.T @ part + identity * (0.03 * (rows - stop + start))
        fitted = torch.cholesky_solve(
            cross - part.T @ targets[start:stop], torch.linalg.cholesky(train_gram))
        error = targets[start:stop] - part @ fitted
        # Key sequence bias cancels in softmax; query bias does not.
        query_width = query_heads * head_dim
        error[:, query_width:] -= error[:, query_width:].mean(0, keepdim=True)
        residual[start:stop] = error
    query_width = query_heads * head_dim
    query_error = residual[:, :query_width].reshape(rows, query_heads, head_dim).transpose(0, 1)
    key_error = residual[:, query_width:].reshape(rows, key_heads, head_dim).transpose(0, 1)
    denominator = max(rows - len(groups), 1)
    key_covariance = key_error.transpose(1, 2) @ key_error / denominator
    eigenvalues, basis = torch.linalg.eigh(key_covariance)
    eigenvalues = eigenvalues.clamp_min(0)
    tail = eigenvalues[:, :-rank].mean(1) if rank < head_dim else torch.zeros(key_heads)
    retained = (eigenvalues[:, -rank:] - tail[:, None]).clamp_min(0)
    repeat = query_heads // key_heads
    basis = basis[:, :, -rank:].repeat_interleave(repeat, 0)
    retained = retained.repeat_interleave(repeat, 0)
    tail = tail.repeat_interleave(repeat, 0)
    query_map = coefficients[:, :query_width].reshape(-1, query_heads, head_dim).transpose(0, 1)
    projected_map = query_map @ basis
    metric = (projected_map * retained[:, None]) @ projected_map.transpose(1, 2)
    # Preserve residual covariance trace with an isotropic discarded subspace.
    metric += tail[:, None, None] * (query_map @ query_map.transpose(1, 2))
    projected_error = query_error @ basis
    floor = (projected_error.square() * retained[:, None]).sum((1, 2))
    floor += tail * query_error.square().sum((1, 2))
    metric = metric / head_dim
    floor = floor / (rows * head_dim)
    if not bool(torch.isfinite(metric).all() & torch.isfinite(floor).all()):
        return None
    return {
        'center': model['center'], 'projection': model['projection'],
        'kernel': model['kernel'],
        **{k: model[k] for k in ('a430_temperature', 'a430_active') if k in model},
        **({'query_sigma_modes': model['query_sigma_modes']} if 'query_sigma_modes' in model else {}),
        'key_error_metric': metric.to(torch.bfloat16).contiguous(),
        'key_error_floor': floor.to(torch.bfloat16).contiguous(),
        'active': torch.ones(key_heads, dtype=torch.bool),
    }

def _v441_fields(value, model, key_heads):
    features = _vk_features(value, model)
    indices = torch.linspace(0, value.shape[0] - 1, min(_VK_QUERIES, value.shape[0])).long()
    queries = features[indices]
    variance = ((queries[None] @ model['key_error_metric'].float()) * queries[None]).sum(-1)
    variance = (variance + model['key_error_floor'].float()[:, None]).clamp_min(0)
    probability = _vk_probabilities(value, model, key_heads)
    base_count = variance.numel() // key_heads
    scenarios = probability.shape[1] // base_count
    variance = variance[:, :, None].expand(-1, -1, scenarios).reshape(key_heads, -1)
    return probability, variance

def _v441_terms(error, probability, variance):
    count = probability.shape[1]
    square_probability = probability.square()
    mass = square_probability.sum(-1)
    output = probability @ error
    squared_output = square_probability @ error
    diagonal = (variance[:, :, None] * square_probability).mean(1)
    multiplier = 1 + variance * mass
    gradient = probability.transpose(1, 2) @ (
        multiplier[:, :, None] * output - variance[:, :, None] * squared_output)
    gradient -= square_probability.transpose(1, 2) @ (variance[:, :, None] * output)
    gradient = gradient / count + diagonal[:, :, None] * error + error / error.shape[1] ** 2
    curvature = (square_probability * (
        multiplier[:, :, None] + variance[:, :, None] * (1 - 2 * probability))).mean(1)
    curvature = curvature + 1 / error.shape[1] ** 2
    return gradient, curvature

def _v441_refine(value, params, model, key_heads, head_dim):
    tokens = value.shape[0]
    query_heads = model['kernel'].shape[0]
    if not 2 <= tokens <= 4096 or tokens * query_heads * head_dim > 16777216:
        return params
    active = torch.nonzero(model['active'], as_tuple=False).flatten()
    if active.numel() == 0:
        return params
    probability, variance = _v441_fields(value, model, key_heads)
    if probability.shape[1] > 128:
        return params
    probability, variance = probability[active], variance[active]
    reconstructed = _deq_params(params).reshape(tokens, key_heads, head_dim).transpose(0, 1).contiguous()
    target = value.reshape(tokens, key_heads, head_dim).transpose(0, 1)[active].contiguous()
    step = (_params_unit_flat(params) * .25).reshape(tokens, key_heads, head_dim).transpose(0, 1)[active].contiguous()
    best = reconstructed[active].clone()
    error = target - best
    squared_error = error.square().sum(1)
    budget = squared_error.clamp_min(1e-30) * 1.05
    for _ in range(16):
        gradient, curvature = _v441_terms(error, probability, variance)
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
    g0, _ = _v441_terms(before, probability, variance)
    g1, _ = _v441_terms(after, probability, variance)
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

def _v441_calibrate(samples, parent, query_heads, key_heads, head_dim):
    if len(samples) < 3 or head_dim % 64:
        return parent
    holdouts = [i for i, sample in enumerate(samples) if 4 <= sample['v'][0].shape[0] <= 4096]
    if not holdouts:
        return parent
    holdout = holdouts[-1]
    model_samples = [sample for i, sample in enumerate(samples) if i != holdout] + [samples[holdout]]
    try:
        model = _v441_fit(model_samples, parent['v_state'], query_heads, key_heads, head_dim)
        if model is None:
            return parent
        sample = samples[holdout]
        value = dequantize_nvfp4(*sample['v']).float()
        baseline = _v441_parent_dynamic(*sample['v'], key_heads, head_dim, parent['v_state'])
        candidate = _v441_refine(value, baseline, model, key_heads, head_dim)
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
        state['v441_key_risk'] = model
        return dict(parent, v_state=state)
    except (RuntimeError, ValueError):
        return parent

_v441_parent_calibration = hif4_calibration_attention
_v441_parent_dynamic = hif4_dynamic_quantize_v


@torch.inference_mode()
def hif4_calibration_attention(samples, qh, kh, d):
    parent = _v441_parent_calibration(samples, qh, kh, d)
    return _v441_calibrate(samples, parent, qh, kh, d)


@torch.inference_mode()
def hif4_dynamic_quantize_v(quant, scale, kh, d, state):
    params = _v441_parent_dynamic(quant, scale, kh, d, state)
    if not isinstance(state, dict) or 'v441_key_risk' not in state:
        return params
    try:
        value = dequantize_nvfp4(quant, scale).float()
        return _v441_refine(value, params, state['v441_key_risk'], kh, d)
    except (RuntimeError, ValueError):
        return params

