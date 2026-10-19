"""v400: cross-sequence Q residual sigma scenarios for the existing V solver."""
_vu_parent_fit = _vk_fit
_vu_parent_probabilities = _vk_probabilities


def _vu_modes(features, queries, keys, groups, qh, kh, d):
    # Residuals are out-of-sequence for Q regression, conditional on the shared
    # feature map and K model. This is not a fully held-out end-to-end model.
    n, width = features.shape
    eye = torch.eye(width)
    h = features.T @ features
    fq = features.T @ queries
    chol = torch.linalg.cholesky(h + eye * (0.03 * n))
    bk = torch.cholesky_solve(features.T @ keys, chol)
    bk = bk.reshape(width, kh, d).permute(1, 0, 2).repeat_interleave(qh // kh, 0)
    residual = torch.empty_like(queries)
    for start, stop in groups:
        f = features[start:stop]
        a = h - f.T @ f + eye * (0.03 * (n - (stop - start)))
        bq = torch.cholesky_solve(fq - f.T @ queries[start:stop], torch.linalg.cholesky(a))
        residual[start:stop] = queries[start:stop] - f @ bq
    r = residual.reshape(n, qh, d).transpose(0, 1)
    coefficients = torch.bmm(r, bk.transpose(1, 2)) / (d ** 0.5)
    # The last feature is exactly one: this coefficient only shifts all key
    # logits equally. Remove that softmax-null direction before rank truncation.
    coefficients[:, :, -1] = 0
    # Centering discards mean model bias; this candidate isolates covariance.
    coefficients = coefficients - coefficients.mean(1, keepdim=True)
    cov = coefficients.transpose(1, 2) @ coefficients / max(n - 1, 1)
    eig, basis = torch.linalg.eigh(cov)
    return (basis[:, :, -2:] * eig[:, None, -2:].clamp_min(0).sqrt()).transpose(1, 2).contiguous()


def _vk_fit(samples, qh, kh, d):
    state = _vu_parent_fit(samples, qh, kh, d)
    if len(samples) < 3:
        return state
    try:
        arrays = {role: [] for role in 'qkv'}
        groups = []
        count = 0
        for sample in samples[:-1]:
            t = sample['v'][0].shape[0]
            ix = torch.linspace(0, t - 1, min(256, t)).long()
            groups.append((count, count + ix.numel()))
            count += ix.numel()
            for role in 'qkv':
                arrays[role].append(dequantize_nvfp4(*sample[role]).float()[ix])
        q, k, v = [torch.cat(arrays[role]) for role in 'qkv']
        modes = _vu_modes(_vk_features(v, state), q, k, groups, qh, kh, d)
        modes = modes.to(torch.bfloat16).contiguous()
        if bool(torch.isfinite(modes).all()):
            state['query_sigma_modes'] = modes
    except (RuntimeError, ValueError):
        pass
    return state


def _vk_probabilities(x, state, kh):
    modes = state.get('query_sigma_modes')
    if modes is None or x.shape[0] > 4096 or not bool(modes.any()):
        return _vu_parent_probabilities(x, state, kh)
    f = _vk_features(x, state)
    ix = torch.linspace(0, x.shape[0] - 1, min(_VK_QUERIES, x.shape[0])).long()
    logits = (f[ix].unsqueeze(0) @ state['kernel'].float()) @ f.T
    delta = (modes.float() @ f.T) * (2.0 ** 0.5)
    offsets = torch.cat((delta, -delta), 1)
    # Four equally weighted sigma points have the retained rank-two covariance.
    # Keep each P separately: E[||P e||^2], not ||E[P] e||^2.
    p = torch.softmax(logits[:, :, None, :] + offsets[:, None, :, :], -1)
    return p.reshape(kh, -1, x.shape[0]).contiguous()
