"""Bounded multi-path block lattice search after the parent Q/K quantizer."""
_BEAM_WIDTH = 4
_BEAM_BLOCK = 8
_BEAM_PASSES = 1
_BEAM_MAX_T = 4096
_BEAM_CORRELATED = True


def _beam_quant(center, step, u, width):
    # center: [batch, rows, block]. Keep a true beam of partial assignments.
    batch, rows, d = center.shape
    work = center[:, :, None, :].clone()
    codes = torch.zeros_like(work)
    costs = torch.zeros(batch, rows, 1, dtype=center.dtype)
    for i in range(d):
        current = work[..., i]
        lower = torch.floor(current/step[:, :, None, i]).clamp(-7, 7)
        upper = (lower+1).clamp(-7, 7)
        # At saturation both branches may duplicate, but never leave the grid.
        proposed = torch.stack((lower, upper), -1)*step[:, :, None, None, i]
        err = (current[..., None]-proposed)/u[:, None, None, None, i, i]
        trial_cost = (costs[..., None]+err.square()).flatten(-2)
        keep = min(width, trial_cost.shape[-1])
        cost, selected = torch.topk(trial_cost, keep, dim=-1, largest=False, sorted=True)
        previous = selected//2
        selected_work = work.gather(2, previous[..., None].expand(-1, -1, -1, d))
        selected_codes = codes.gather(2, previous[..., None].expand(-1, -1, -1, d))
        err = err.flatten(-2).gather(-1, selected)
        q = proposed.flatten(-2).gather(-1, selected)
        selected_work -= err[..., None]*u[:, None, None, i, :]
        selected_codes[..., i] = q
        work, codes, costs = selected_work, selected_codes, cost
    return codes[:, :, 0]


def _beam_prepare(st, d):
    if not st.get('gq') or 'H' not in st:
        return None
    h = st['H'].double()
    h = (h+h.transpose(-1, -2))*0.5
    norm = h.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-20)
    h = h/norm[:, None, None] + torch.eye(d)*0.01
    perms = []
    for head in h:
        if not _BEAM_CORRELATED:
            perms.append(torch.arange(d)); continue
        diag = head.diagonal().clamp_min(1e-20).sqrt()
        corr = (head/(diag[:, None]*diag[None])).abs()
        remaining = torch.ones(d, dtype=torch.bool)
        order = []
        while len(order)<d:
            start = int(torch.where(remaining, head.diagonal(), -1).argmax())
            block = [start]; remaining[start] = False
            while len(block)<_BEAM_BLOCK:
                importance = corr[block].square().sum(0)
                index = int(importance.masked_fill(~remaining, -1).argmax())
                block.append(index); remaining[index] = False
            order.extend(block)
        perms.append(torch.tensor(order))
    perm = torch.stack(perms)
    hp = h.gather(1, perm[:, :, None].expand(-1, -1, d)).gather(2, perm[:, None, :].expand(-1, d, -1))
    blocks = torch.stack([hp[:, i:i+_BEAM_BLOCK, i:i+_BEAM_BLOCK] for i in range(0, d, _BEAM_BLOCK)], 1)
    ch, info = torch.linalg.cholesky_ex(blocks)
    if bool(info.any()):
        return None
    inv = torch.cholesky_solve(torch.eye(_BEAM_BLOCK, dtype=ch.dtype).expand_as(ch), ch)
    u = torch.linalg.cholesky(inv, upper=True)
    return dict(st, beam_h=hp.float().contiguous(), beam_perm=perm,
                beam_inv=inv.float().contiguous(), beam_u=u.float().contiguous())


def _beam_refine(x, params, st, heads, d, original_rows):
    if original_rows<=1 or original_rows>_BEAM_MAX_T or not isinstance(st, dict) or 'beam_h' not in st:
        return params
    rows = x.shape[0]; nb = d//_BEAM_BLOCK
    active = st.get('beam_active', torch.ones(heads, dtype=torch.bool))
    if not bool(active.any()):
        return params
    group = torch.arange(heads)*st['beam_h'].shape[0]//heads
    h, perm = st['beam_h'][group], st['beam_perm'][group]
    inv, u = st['beam_inv'][group], st['beam_u'][group]
    z = _deq_params(params).reshape(rows, heads, d).transpose(0, 1).contiguous()
    target = x.reshape(rows, heads, d).transpose(0, 1)
    step = _params_unit_flat(params).reshape(rows, heads, d).transpose(0, 1)*0.25
    index = perm[:, None, :].expand(heads, rows, d)
    z, target, step = [value.gather(-1, index) for value in (z, target, step)]
    for unused in range(_BEAM_PASSES):
        gradient = (z-target)@h
        # All blocks propose in parallel, then accept using the complete H.
        zb = z.reshape(heads, rows, nb, _BEAM_BLOCK).transpose(1, 2)
        gb = gradient.reshape(heads, rows, nb, _BEAM_BLOCK).transpose(1, 2)
        sb = step.reshape(heads, rows, nb, _BEAM_BLOCK).transpose(1, 2)
        center = zb-gb@inv
        trial = _beam_quant(center.flatten(0, 1), sb.flatten(0, 1), u.flatten(0, 1), _BEAM_WIDTH).reshape_as(zb)
        delta = trial-zb
        block_h = torch.stack([h[:, i:i+_BEAM_BLOCK, i:i+_BEAM_BLOCK] for i in range(0, d, _BEAM_BLOCK)], 1)
        change = (2*delta*gb+(delta@block_h)*delta).sum(-1)
        delta = torch.where((change<0)[..., None], delta, 0.0)
        full_delta = delta.transpose(1, 2).reshape_as(z)
        global_change = (2*full_delta*gradient+(full_delta@h)*full_delta).sum(-1)
        # Cross-block interaction can invalidate independent proposals. Fall
        # back to one best block, never accept a stale summed local objective.
        best = change.argmin(1)
        single = torch.zeros_like(delta)
        single.scatter_(1, best[:, None, :, None].expand(-1, 1, -1, _BEAM_BLOCK),
                        delta.gather(1, best[:, None, :, None].expand(-1, 1, -1, _BEAM_BLOCK)))
        single = single.transpose(1, 2).reshape_as(z)
        sd = (2*single*gradient+(single@h)*single).sum(-1)
        take_all = global_change<sd
        update = torch.where(take_all[..., None], full_delta, single)
        improvement = torch.where(take_all, global_change, sd)
        z += torch.where(((improvement < -1e-8)&active[:, None])[..., None], update, 0.0)
    values = torch.empty_like(z).scatter_(-1, index, z).transpose(0, 1).reshape(rows, heads*d)
    result = _values_to_params(values, params)
    mask = active.repeat_interleave(d//64)[None, :, None, None, None]
    return {key:torch.where(mask, result[key], params[key]) for key in params}
