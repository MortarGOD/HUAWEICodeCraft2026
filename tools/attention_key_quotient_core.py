"""v426: K token-quotient lattice risk; appended to the standalone parent."""

_v426_parent_refine = _refine_attn_heads


def _v426_quotient_proposals(Ma, da, codes, heads, mean, gwflat, diag, tokens):
    """One top-1 proposal per row, with an exact coupled batch check per head.

    M is the uncentered error @ H. A simultaneous batch cannot be scored as
    a sum of independent centered single-row increments. If that batch fails,
    retain only its best single-row move. No tensor supplied here is mutated.
    """
    B, dh = diag.shape
    grad = Ma - mean[heads]
    costs = -2.0 * da * grad.abs() + (1.0 - 1.0 / tokens) * da.square() * diag[heads]
    up = grad < 0.0
    invalid = torch.where(up, codes >= 7.0, codes <= -7.0)
    costs.masked_fill_(invalid, float('inf'))
    idx = costs.argmin(1, keepdim=True)
    single = costs.gather(1, idx)[:, 0]
    active = torch.isfinite(single) & (single < 0.0)
    dr = torch.where(grad.gather(1, idx) < 0.0, 1.0, -1.0) * active[:, None]
    coef = dr * da.gather(1, idx)
    gb = gwflat.index_select(0, heads * dh + idx[:, 0])
    sum_d = torch.zeros((B * dh,), dtype=Ma.dtype, device=Ma.device)
    sum_d.scatter_add_(0, heads * dh + idx[:, 0], coef[:, 0])
    sum_d = sum_d.reshape(B, dh)
    sum_dh = torch.zeros_like(mean)
    sum_dh.index_add_(0, heads, coef * gb)
    own = coef[:, 0].square() * diag[heads, idx[:, 0]]
    linear = 2.0 * coef[:, 0] * grad.gather(1, idx)[:, 0]
    delta = torch.zeros((B,), dtype=Ma.dtype, device=Ma.device)
    delta.index_add_(0, heads, linear + own)
    delta -= (sum_d * sum_dh).sum(1) / tokens
    bad = (~torch.isfinite(delta)) | (delta >= 0.0)
    if bool((bad[heads] & active).any()):
        eligible = torch.where(active, single, float('inf'))
        best = torch.full_like(delta, float('inf'))
        best.scatter_reduce_(0, heads, eligible, reduce='amin', include_self=True)
        positions = torch.arange(heads.numel(), device=heads.device)
        chosen = torch.where(active & (single == best[heads]), positions, heads.numel())
        first = torch.full((B,), heads.numel(), dtype=torch.long, device=heads.device)
        first.scatter_reduce_(0, heads, chosen, reduce='amin', include_self=True)
        keep = (~bad[heads]) | (positions == first[heads])
        dr = dr * keep[:, None]
        coef = coef * keep[:, None]
    return idx, dr, coef, gb, active, delta


def _v426_quotient_refine(x, values, unit, H, num_heads, n_sweeps, total_rounds=None):
    T, C = x.shape
    if T <= 1 or C == 0 or n_sweeps <= 0 or num_heads <= 0 or H.shape[-1] == 0:
        return values
    B, dh = num_heads, H.shape[-1]
    hv = torch.arange(B) * H.shape[0] // B
    gw = H[hv].float().contiguous()
    d = 0.25 * unit
    codes = torch.round(values / unit * 4.0)
    v4f = codes.reshape(T, B, dh).permute(1, 0, 2).reshape(B * T, dh).contiguous()
    df = d.reshape(T, B, dh).permute(1, 0, 2).reshape(B * T, dh).contiguous()
    xf = x.reshape(T, B, dh).permute(1, 0, 2).reshape(B * T, dh).contiguous()
    M = torch.bmm((v4f * df - xf).reshape(B, T, dh), gw).reshape(B * T, dh)
    mean = M.reshape(B, T, dh).mean(1)
    flat = gw.reshape(B * dh, dh)
    diag = gw.diagonal(dim1=-2, dim2=-1)
    rows = torch.arange(B * T)
    heads = rows // T
    Ma, va, da = M, v4f, df
    local = False
    total = n_sweeps * (32 if T <= 128 else _REF_ATTN_ROUNDS) if total_rounds is None else total_rounds
    for _ in range(total):
        if rows.numel() == 0:
            break
        idx, dr, coef, gb, active, _ = _v426_quotient_proposals(
            Ma, da, va, heads, mean, flat, diag, T)
        va.scatter_add_(1, idx, dr)
        Ma.addcmul_(gb, coef)
        # Every accepted update changes the mean, including rows frozen later.
        mean.index_add_(0, heads, coef * gb / T)
        if bool(active.all()):
            continue
        if local:
            v4f[rows[~active]] = va[~active]
        rows, heads = rows[active], heads[active]
        Ma, va, da = Ma[active], va[active], da[active]
        local = True
    if local:
        v4f[rows] = va
    return v4f.reshape(B, T, dh).permute(1, 0, 2).reshape(T, C) * d


def _refine_attn_heads(x, values, unit, H, num_heads, n_sweeps,
                       total_rounds=None, quotient=False):
    if not quotient:
        return _v426_parent_refine(x, values, unit, H, num_heads, n_sweeps,
                                  total_rounds=total_rounds)
    return _v426_quotient_refine(x, values, unit, H, num_heads, n_sweeps,
                               total_rounds=total_rounds)
