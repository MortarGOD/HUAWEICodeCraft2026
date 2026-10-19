"""Current-V content risk and full-grid reconstruction; embedded in v436."""


def _v436_probability(xh):
    # xh: heads, tokens, channels. No calibration peers or token positions.
    kh, t, d = xh.shape
    rows = torch.arange(d, device=xh.device)[:, None]
    columns = torch.tensor([1, 2, 4, 8, 16, 32, 3, 5], device=xh.device)[None]
    bits = rows.bitwise_and(columns)
    parity = torch.zeros_like(bits)
    for shift in range(6):
        parity = parity.bitwise_xor((bits >> shift).bitwise_and(1))
    directions = (1 - 2 * parity).to(xh.dtype) / (d ** .5)
    features = (xh - xh.mean(1, keepdim=True)) @ directions
    rms = features.square().mean(1, keepdim=True).sqrt()
    floor = rms.amax(-1, keepdim=True).clamp_min(1e-12) * 1e-4
    features = features / rms.clamp_min(floor)
    logits = features.transpose(1, 2)
    signed = torch.cat((logits, -logits), 1)
    probabilities = torch.softmax(torch.cat((signed * .5, signed * 2.), 1), -1)
    # Weighted rows: ||P e||^2 = .5 mean(e)^2 + .5 E_field[(p e)^2].
    probabilities = probabilities * ((.5 / probabilities.shape[1]) ** .5)
    uniform = torch.full((kh, 1, t), (.5 ** .5) / t,
                         dtype=xh.dtype, device=xh.device)
    return torch.cat((uniform, probabilities), 1).contiguous()


def _v436_block_risk(probability, error):
    # Public BF16 decoded values supplied by caller; double for acceptance.
    kh, t, d = error.shape
    e = error.double()
    sse = e.square().sum(1).reshape(kh, d // 64, 64).sum(-1)
    output = (probability.double() @ e).square().sum(1)
    cost = output.reshape(kh, d // 64, 64).sum(-1) + sse / (t * t)
    return cost, sse


def _v436_continuous(xh, y, dual, probability, chol, rho, diagonal):
    # Solve (P^T P + lambda I + rho I) z = H x + rho (y-u).
    displacement = y - dual - xh
    low = torch.cholesky_solve(probability @ displacement, chol)
    return xh + (rho / diagonal)[:, None, None] * (
        displacement - probability.transpose(1, 2) @ low)


def _v436_reconstruct(x, params, kh, d):
    t = x.shape[0]
    if (x.device.type != 'cpu' or t < 2 or t > 4096 or d % 64
            or x.numel() > 2 ** 20):
        return params
    xh = x.float().reshape(t, kh, d).transpose(0, 1).contiguous()
    probability = _v436_probability(xh)
    penalty = 1. / (t * t)
    rho = probability.square().sum((1, 2)) / t + penalty
    diagonal = rho + penalty
    gram = probability @ probability.transpose(1, 2)
    gram = gram + torch.eye(gram.shape[-1])[None] * diagonal[:, None, None]
    chol, info = torch.linalg.cholesky_ex(gram)
    if bool(info.any()) or not bool(torch.isfinite(chol).all()):
        return params
    # HiF4 decode includes the public BF16 round before objective evaluation.
    y = _deq_params(params).to(torch.bfloat16).float()
    y = y.reshape(t, kh, d).transpose(0, 1).contiguous()
    best = params
    best_cost, original_sse = _v436_block_risk(probability, y - xh)
    dual = torch.zeros_like(y)
    for _ in range(2):
        z = _v436_continuous(xh, y, dual, probability, chol, rho, diagonal)
        proposal_input = (z + dual).transpose(0, 1).reshape(t, kh * d).contiguous()
        if not bool(torch.isfinite(proposal_input).all()):
            break
        proposal = _dyn_table(proposal_input, None, False)
        decoded = _deq_params(proposal).to(torch.bfloat16).float()
        y = decoded.reshape(t, kh, d).transpose(0, 1).contiguous()
        cost, sse = _v436_block_risk(probability, y - xh)
        accept = (cost < best_cost) & (sse <= original_sse * 1.10)
        accept = accept & torch.isfinite(cost) & torch.isfinite(sse)
        if bool(accept.any()):
            block_mask = accept.reshape(1, -1)
            best = {key: torch.where(
                block_mask.reshape(*block_mask.shape, *([1] * (value.ndim - 2))),
                proposal[key], value).contiguous() for key, value in best.items()}
            best_cost = torch.where(accept, cost, best_cost)
        dual = dual + z - y
    # Recompute the exact merged result; never trust an unpacked/soft proposal.
    decoded = _deq_params(best).to(torch.bfloat16).float()
    error = decoded.reshape(t, kh, d).transpose(0, 1) - xh
    final_cost, final_sse = _v436_block_risk(probability, error)
    original = _deq_params(params).to(torch.bfloat16).float()
    old_cost, _ = _v436_block_risk(
        probability, original.reshape(t, kh, d).transpose(0, 1) - xh)
    valid = ((final_cost <= old_cost) & (final_sse <= original_sse * 1.10)
             & torch.isfinite(final_cost)).reshape(1, -1)
    return {key: torch.where(
        valid.reshape(*valid.shape, *([1] * (value.ndim - 2))),
        value, params[key]).contiguous() for key, value in best.items()}
