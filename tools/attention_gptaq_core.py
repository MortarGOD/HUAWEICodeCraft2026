"""Independent asymmetric sequential quantization research, not a release.

All additions are calibration statistics; role interfaces remain independent.
Block boundaries propagate the same pre-update coefficients as a scalar sweep.
This deliberately avoids the reference repository's inconsistent block formula.
"""
_AQ_ALPHA = 0.25
_AQ_MODE = "triangular"
_AQ_ROWS = 192
AQ_DEBUG = {}


def _aq_sweep(x, unit, u, p, block=32):
    if not bool(p.any()):
        return _gptq_quantize_batched(x, unit, u)
    w = x.clone()
    out = torch.empty_like(w)
    n = w.shape[-1]
    for start in range(0, n, block):
        stop = min(start + block, n)
        local = w[..., start:stop].permute(2, 0, 1).contiguous()
        q = torch.empty_like(local)
        errors = torch.empty_like(local)
        coefficients = torch.empty_like(local)
        for i in range(stop-start):
            col = start+i
            current = local[i]
            coefficients[i] = current
            q[i] = (current / unit[..., col] * 4).round().clamp(-7, 7) * (unit[..., col] * 0.25)
            errors[i] = (current-q[i]) / u[:, col, col, None].clamp_min(1e-30)
            if col+1 < stop:
                local[i+1:] -= errors[i][None] * u[:, col, col+1:stop].T[:, :, None]
                local[i+1:] += coefficients[i][None] * p[:, col, col+1:stop].T[:, :, None]
        out[..., start:stop] = q.permute(1, 2, 0)
        if stop < n:
            w[..., stop:] -= errors.permute(1, 2, 0).contiguous() @ u[:, start:stop, stop:]
            w[..., stop:] += coefficients.permute(1, 2, 0).contiguous() @ p[:, start:stop, stop:]
    return out


def _aq_transform(x, state, nh, kh, d, role):
    y = x.reshape(-1, nh, d)
    sv = state.get('qs' if role == 'q' else 'ks')
    if sv is not None:
        y = y * sv.float().repeat_interleave(nh//kh, 0)[None]
    if 'joint_transform' in state:
        y = (y.transpose(0, 1) @ state['joint_transform'].float().repeat_interleave(nh//kh, 0)).transpose(0, 1)
    elif state.get('rot'):
        y = y @ _make_R(d, int(state.get('rseed') or 0))
    if role == 'q' and 'temperature_gain' in state:
        y = y * state['temperature_gain'].float()[None, :, None]
    return y.reshape(x.shape).contiguous()


def _aq_quant(x, state, nh, d, rows):
    params = _dyn_table(x, None, has_scale=False)
    unit = _params_unit_flat(params)
    def expand(t):
        return t.float().repeat_interleave(nh//t.shape[0], 0)
    xb = x.reshape(-1, nh, d).transpose(0, 1).contiguous()
    ub = unit.reshape(-1, nh, d).transpose(0, 1).contiguous()
    center = xb @ expand(state['aq_center'])
    if state['aq_mode'] == 'center':
        values = _gptq_quantize_batched(center, ub, expand(state['u']))
    else:
        values = _aq_sweep(xb, ub, expand(state['u']), expand(state['aq_p']))
    values = values.transpose(0, 1).reshape(x.shape).contiguous()
    rf = int(state.get('rf') or 0)
    if rf and rows <= ATTN_REFINE_MAX_T and 'H' in state:
        rounds = rf * (32 if rows <= 128 else (4 if rows > 1024 else _REF_ATTN_ROUNDS))
        values = _refine_attn_heads(center.transpose(0, 1).reshape(x.shape).contiguous(),
                                   values, unit, state['H'], nh, rf, total_rounds=rounds)
    return _values_to_params(values, params)


def _aq_fit(samples, states, qh, kh, d):
    AQ_DEBUG.clear()
    if not all(states[r+'_state'].get('gq') for r in 'qk'):
        AQ_DEBUG['skipped'] = 'parent GPTQ unavailable'
        return None
    grams = {r: torch.zeros(kh, d, d) for r in 'qk'}
    differences = {r: torch.zeros(kh, d, d) for r in 'qk'}
    count = 0
    for sample in samples[:-1]:
        rows = sample['q'][0].shape[0]
        if rows <= 1:
            continue
        ix = torch.linspace(0, rows-1, min(rows, _AQ_ROWS)).long()
        for role, nh in (('q', qh), ('k', kh)):
            st = states[role+'_state']
            raw = _aq_transform(dequantize_nvfp4(sample[role][0][ix], sample[role][1][ix]).float(), st, nh, kh, d, role)
            if role == 'q':
                quant = _joint_quant_queries(sample, st, qh, kh, d, ix)
            else:
                # K rows are independent, but preserve the original T budget.
                pseudo = {'q': sample['k'], 'v': sample['v']}
                qst = dict(st)
                if 'ks' in qst:
                    qst['qs'] = qst.pop('ks')
                quant = _joint_quant_queries(pseudo, qst, kh, kh, d, ix)
            a = raw.reshape(-1, kh, nh//kh, d).permute(1, 0, 2, 3).reshape(kh, -1, d)
            b = quant.reshape(-1, kh, nh//kh, d).permute(1, 0, 2, 3).reshape(kh, -1, d)
            grams[role] += b.transpose(1, 2) @ b
            differences[role] += (a-b).transpose(1, 2) @ b
        count += ix.numel()
    if count == 0:
        return None
    answer = dict(states)
    for role, other in (('q', 'k'), ('k', 'q')):
        h, diff = grams[other].double(), differences[other].double()
        norm = h.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-20)
        h, diff = h/norm[:, None, None], diff/norm[:, None, None]
        hd = h + GPTQ_DAMP * torch.eye(d, dtype=torch.float64)
        chol = torch.linalg.cholesky(hd)
        inv = torch.cholesky_solve(torch.eye(d, dtype=torch.float64).expand(kh, d, d), chol)
        u = torch.linalg.cholesky(inv, upper=True)
        alpha = 0.0 if _AQ_MODE == 'metric' else _AQ_ALPHA
        p = alpha * torch.triu(diff @ u.transpose(1, 2), diagonal=1) @ u
        center = torch.eye(d, dtype=torch.float64) + alpha * diff @ inv
        assert bool(torch.isfinite(p).all() and torch.isfinite(center).all())
        AQ_DEBUG[role+'_cross_relative_norm'] = float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(h).clamp_min(1e-30))
        AQ_DEBUG[role+'_center_relative_change'] = float(torch.linalg.vector_norm(center-torch.eye(d))/((kh*d)**0.5))
        answer[role+'_state'] = dict(states[role+'_state'], u=u.float().contiguous(),
                                    H=hd.float().contiguous(), aq_p=p.float().contiguous(),
                                    aq_center=center.float().contiguous(), aq_mode=_AQ_MODE)
    return answer


def _aq_guard(hold, states, proposed, qh, kh, d):
    rows = hold['v'][0].shape[0]
    ix = torch.linspace(0, rows-1, min(rows, 32)).long()
    raw = [dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    target = _joint_output(*raw, qh, kh, d, ix)
    v = _deq_params(hif4_dynamic_quantize_v(*hold['v'], kh, d, states['v_state']))
    qs = [_joint_quant_queries(hold, st['q_state'], qh, kh, d, ix) for st in (states, proposed)]
    ks = [_deq_params(hif4_dynamic_quantize_k(*hold['k'], kh, d, st['k_state'])) for st in (states, proposed)]
    costs = torch.empty(2, 2, 2, kh)
    for qi in range(2):
        for ki in range(2):
            err = (_joint_output(qs[qi], ks[ki], v, qh, kh, d, ix, query_subset=True)-target).square().mean((-1, -2))
            costs[qi, ki] = torch.stack((err[::2].mean(0), err[1::2].mean(0)))
    assert bool(torch.isfinite(costs).all())
    masks = {r: torch.zeros(kh, dtype=torch.bool) for r in 'qk'}
    for hv in range(kh):
        best = costs[0, 0, :, hv]
        for qi, ki in ((1, 0), (0, 1), (1, 1)):
            trial = costs[qi, ki, :, hv]
            if bool((trial < best*0.995).all()):
                masks['q'][hv], masks['k'][hv] = bool(qi), bool(ki)
                best = trial
    answer = dict(states)
    for role in 'qk':
        mask = masks[role]
        if not bool(mask.any()):
            continue
        old, st = states[role+'_state'], dict(proposed[role+'_state'])
        for key in ('H', 'u'):
            # H may be absent when the existing refinement budget is zero.
            # In that case it is never read by either inactive or active rows.
            parent = old[key].float() if key in old else st[key]
            st[key] = torch.where(mask[:, None, None], st[key], parent).contiguous()
        st['aq_p'] = torch.where(mask[:, None, None], st['aq_p'], 0.0).contiguous()
        st['aq_center'] = torch.where(mask[:, None, None], st['aq_center'], torch.eye(d)).contiguous()
        answer[role+'_state'] = st
    AQ_DEBUG.update(q_groups=int(masks['q'].sum()), k_groups=int(masks['k'].sum()),
                    ratios=(costs.mean((2, 3))/costs[0, 0].mean().clamp_min(1e-30)).tolist())
    return answer


_aq_parent_calibration = hif4_calibration_attention


def hif4_calibration_attention(samples, qh, kh, d):
    states = _aq_parent_calibration(samples, qh, kh, d)
    AQ_DEBUG.clear()
    if len(samples)<2 or not all(states[r+'_state'].get('gq') for r in 'qk'):
        return states
    rows = samples[-1]['v'][0].shape[0]
    if rows < 4 or rows > 4096:
        return states
    proposed = _aq_fit(samples, states, qh, kh, d)
    return states if proposed is None else _aq_guard(samples[-1], states, proposed, qh, kh, d)
