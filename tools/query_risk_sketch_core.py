"""v403: dense predicted queries, compressed second-moment risk, fixed solver width."""
_QR_QUERY_COUNT = 16
_qr_parent_probabilities = _vk_probabilities


def _qr_compress(probabilities, keep):
    rows = probabilities.shape[1]
    if rows <= keep:
        return probabilities
    gram = probabilities @ probabilities.transpose(-1, -2)
    _, basis = torch.linalg.eigh((gram + gram.transpose(-1, -2)) * .5)
    factors = basis[:, :, -keep:].transpose(-1, -2) @ probabilities
    # mean_keep ||F e||^2 = mean_rows ||U_keep^T P e||^2.
    return (factors * ((keep / rows) ** .5)).contiguous()


def _vk_probabilities(x, state, kh):
    if x.shape[0] <= _VK_QUERIES or x.shape[0] > 4096:
        return _qr_parent_probabilities(x, state, kh)
    try:
        f = _vk_features(x, state)
        ix = torch.linspace(0, x.shape[0]-1, min(_QR_QUERY_COUNT,x.shape[0])).long()
        logits = (f[ix][None] @ state['kernel'].float()) @ f.T
        modes = state.get('query_sigma_modes')
        if modes is not None and bool(modes.any()):
            delta = (modes.float() @ f.T) * (2.**.5)
            offsets = torch.cat((delta, -delta), 1)
        else:
            offsets = torch.zeros(logits.shape[0], 1, x.shape[0])
        probabilities = torch.softmax(logits[:,:,None,:]+offsets[:,None,:,:], -1)
        probabilities = probabilities.reshape(logits.shape[0], -1, x.shape[0])
        keep = _VK_QUERIES * offsets.shape[1]
        factors = _qr_compress(probabilities, keep)
        if not bool(torch.isfinite(factors).all()):
            return _qr_parent_probabilities(x,state,kh)
        return factors.reshape(kh,-1,x.shape[0]).contiguous()
    except (RuntimeError,ValueError):
        return _qr_parent_probabilities(x,state,kh)

def _vk_refine(x, p, state, kh, d):
    if x.shape[0] <= 1 or x.shape[0] > 4096 or state is None or not bool(state["active"].any()):
        return p
    probs = _vk_probabilities(x, state, kh)
    steps = _params_unit_flat(p) * 0.25
    baseline = _deq_params(p)
    xh = x.reshape(-1, kh, d).transpose(0, 1).contiguous()
    best = baseline.reshape(-1, kh, d).transpose(0, 1).contiguous()
    step = steps.reshape(-1, kh, d).transpose(0, 1).contiguous()
    penalty = 1.0 / x.shape[0]
    error = xh - best
    original_mse = error.square().mean(1).clamp_min(1e-30)
    best_cost = (probs @ error).square().mean(1) + penalty * original_mse
    mass = ((probs.shape[1] ** .5) * probs.square().sum(1).sqrt()).clamp_min(1e-12)[:, :, None]
    shift = torch.zeros_like(best)
    for _ in range(_VK_ROUNDS):
        residual = probs @ error
        gradient = probs.transpose(1, 2) @ residual
        shift = shift + 0.75 * gradient / mass
        proposal = torch.round((xh + shift) / step).clamp(-7, 7) * step
        error = xh - proposal
        mse = error.square().mean(1)
        cost = (probs @ error).square().mean(1) + penalty * mse
        accept = (cost < best_cost) & (mse <= original_mse * _VK_MSE_CAP)
        accept = accept & state["active"][:, None]
        best = torch.where(accept[:, None], proposal, best)
        best_cost = torch.where(accept, cost, best_cost)
    return _values_to_params(best.transpose(0, 1).reshape(x.shape).contiguous(), p)
