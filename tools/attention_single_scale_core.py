"""v394: replace Q metadata once; do not duplicate GPTQ or refinement."""
SP_DEBUG = {}
_sp_parent_calibration = hif4_calibration_attention


def _sp_weighted(x, weight):
    # Weights were normalized per block during calibration. Do not renormalize
    # across selected heads: their acceptance mask must not change the weights.
    rows, channels = x.shape
    blocks = channels//64
    weight = weight.clamp(.25, 4.) if USE_WEIGHTS else torch.ones_like(weight)
    weight = weight.expand(rows, channels)
    fn = _quant_chunk_vec if x.numel() >= 2_000_000 else _quant_chunk
    pieces = {}
    for start in range(0, rows, ROW_CHUNK):
        p = fn(x[start:start+ROW_CHUNK].reshape(-1, blocks, 8, 2, 4),
               weight[start:start+ROW_CHUNK].reshape(-1, blocks, 8, 2, 4), CAND_GRID)
        for key, value in p.items():
            pieces.setdefault(key, []).append(value)
    return {key: torch.cat(value, 0) for key, value in pieces.items()}


def _sp_params(x, st, nh, d, total_rows):
    if (not isinstance(st, dict) or 'sp_active' not in st or
            not 2 <= total_rows <= 4096 or d%64):
        return _dyn_table(x, None, has_scale=False)
    active = st['sp_active'].repeat_interleave(nh//st['sp_active'].numel())
    if not bool(active.any()):
        return _dyn_table(x, None, has_scale=False)
    weights = st['sp_weight'].float().repeat_interleave(nh//st['sp_weight'].shape[0], 0)
    ids = torch.arange(nh*d//64).reshape(nh, d//64)
    result = None
    for enabled in (False, True):
        selected = torch.where(active == enabled)[0]
        if not selected.numel():
            continue
        columns = (selected[:, None]*d+torch.arange(d)[None]).reshape(-1)
        block_ids = ids[selected].reshape(-1)
        values = x[:, columns]
        params = (_sp_weighted(values, weights[selected].reshape(1, -1)) if enabled
                  else _dyn_table(values, None, has_scale=False))
        if result is None:
            result = {k: v.new_empty((x.shape[0], nh*d//64, *v.shape[2:])) for k,v in params.items()}
        for key in result:
            result[key][:, block_ids] = params[key]
    return result


def _sp_guard(samples, parent, qh, kh, d):
    SP_DEBUG.clear(); SP_DEBUG.update(accepted=0, error=False)
    st = parent['q_state']
    if not samples or d%64 or not st.get('gq') or 'H' not in st:
        return parent
    hold = samples[-1]; t = hold['q'][0].shape[0]
    if not 4 <= t <= 4096:
        return parent
    h = st['H'].float()
    weight = h.diagonal(dim1=-2, dim2=-1).clamp_min(1e-20).reshape(h.shape[0], -1, 64)
    weight = weight/weight.mean(-1, keepdim=True).clamp_min(1e-20)
    weight = weight.reshape(h.shape[0], d).contiguous()
    if not bool(torch.isfinite(weight).all()):
        return parent
    trial = dict(st, sp_weight=weight, sp_active=torch.ones(kh, dtype=torch.bool))
    ix = torch.linspace(0, t-1, min(t, 32)).long()
    raw = [dequantize_nvfp4(hold['q'][0][ix], hold['q'][1][ix]).float()]
    raw += [dequantize_nvfp4(*hold[r]).float() for r in 'kv']
    target = _joint_output(*raw, qh, kh, d, ix, query_subset=True)
    k = _deq_params(hif4_dynamic_quantize_k(*hold['k'], kh, d, parent['k_state']))
    v = _deq_params(hif4_dynamic_quantize_v(*hold['v'], kh, d, parent['v_state']))
    costs = []
    for state in (st, trial):
        q = _joint_quant_queries(hold, state, qh, kh, d, ix)
        error = (_joint_output(q, k, v, qh, kh, d, ix, query_subset=True)-target).square().mean((-1, -2))
        costs.append(torch.stack((error[::2].mean(0), error[1::2].mean(0))))
    take = (costs[1] < costs[0]*.995).all(0) & torch.isfinite(costs[1]).all(0)
    SP_DEBUG.update(accepted=int(take.sum()), ratios=(costs[1].mean(0)/costs[0].mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):
        return parent
    trial['sp_active'] = take
    return dict(parent, q_state=trial)


def hif4_calibration_attention(samples, qh, kh, d):
    parent = _sp_parent_calibration(samples, qh, kh, d)
    try:
        return _sp_guard(samples, parent, qh, kh, d)
    except (RuntimeError, ValueError) as exc:
        SP_DEBUG.update(error=True, error_type=type(exc).__name__)
        return parent
