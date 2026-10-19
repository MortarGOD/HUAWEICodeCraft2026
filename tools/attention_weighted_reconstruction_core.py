"""v396: sequence-separated mass-weighted nonlinear output reconstruction."""
WR_DEBUG = {}
_WR_POINTS = 64
_WR_ITEMS = 256
_WR_ROUNDS = 2
_wr_parent_calibration = hif4_calibration_attention
_wr_parent_q = hif4_dynamic_quantize_q
_wr_parent_queries = _joint_quant_queries


def _wr_cluster(rawk, key, rawv, value, count):
    # [heads, tokens, dim]; paired assignments preserve raw/deployed coupling.
    heads, tokens, d = rawk.shape
    count = min(count, tokens)
    feature = torch.cat((rawk, key), -1)
    # Farthest-point initialization avoids duplicate seeds missing rare modes.
    norm = feature.square().sum(-1)
    seed = norm.argmax(-1)
    nearest = torch.full((heads,tokens), float('inf'))
    centers = []
    for _ in range(count):
        atom = feature[torch.arange(heads),seed]
        centers.append(atom)
        distance = (norm+atom.square().sum(-1)[:,None]-2*torch.bmm(feature,atom[:,:,None]).squeeze(-1)).clamp_min(0)
        nearest = torch.minimum(nearest,distance)
        seed = nearest.argmax(-1)
    centers = torch.stack(centers,1)
    for _ in range(3):
        distance = (feature.square().sum(-1, keepdim=True)
                    + centers.square().sum(-1)[:, None]
                    - 2*torch.bmm(feature, centers.transpose(1, 2)))
        assignment = distance.argmin(-1)
        mass = torch.zeros(heads, count)
        mass.scatter_add_(1, assignment, torch.ones(heads, tokens))
        total = torch.zeros_like(centers)
        total.scatter_add_(1, assignment[..., None].expand_as(feature), feature)
        centers = torch.where(mass[..., None]>0, total/mass.clamp_min(1)[..., None], centers)
    result = {}
    for name, tensor in (('raw', rawk), ('key', key), ('raw_value', rawv), ('value', value)):
        total = torch.zeros(heads, count, d)
        total.scatter_add_(1, assignment[..., None].expand_as(tensor), tensor)
        result[name] = total/mass.clamp_min(1)[..., None]
    result['logmass'] = torch.where(mass>0, mass.clamp_min(1).log(), -80.)
    return result


def _wr_fit(samples, parent, qh, kh, d):
    banks = []
    for sample in samples[:-1][-2:]:
        rawk, rawv = [dequantize_nvfp4(*sample[r]).float().reshape(-1, kh, d).transpose(0, 1) for r in 'kv']
        key = _deq_params(hif4_dynamic_quantize_k(*sample['k'], kh, d, parent['k_state'])).float().reshape(-1, kh, d).transpose(0, 1)
        value = _deq_params(hif4_dynamic_quantize_v(*sample['v'], kh, d, parent['v_state'])).float().reshape(-1, kh, d).transpose(0, 1)
        center = rawv.mean(1, keepdim=True)
        banks.append(_wr_cluster(rawk/(d**.5), key/(d**.5), rawv-center, value-center, _WR_POINTS))
    # Pad representatives only, never concatenate independent softmax sequences.
    width = max(b['key'].shape[1] for b in banks)
    for bank in banks:
        pad = width-bank['key'].shape[1]
        if pad:
            for name in bank:
                shape = (kh, pad) if name=='logmass' else (kh, pad, d)
                tail = torch.full(shape, -80.) if name=='logmass' else torch.zeros(shape)
                bank[name] = torch.cat((bank[name], tail), 1)
    bank = {name:torch.stack([b[name] for b in banks], 1).contiguous() for name in banks[0]}
    bank['active'] = torch.ones(qh, dtype=torch.bool)
    return bank


def _wr_output(q, key, value, logmass):
    probability = torch.softmax(torch.einsum('nd,nsmd->nsm', q, key)+logmass, -1)
    return torch.einsum('nsm,nsmd->nsd', probability, value), probability


def _wr_cost_gradient(q, target, bank, gradient=False):
    output, probability = _wr_output(q, bank['key'], bank['value'], bank['logmass'])
    residual = output-target
    cost = residual.square().sum(-1).mean(-1)
    if not gradient:
        return cost
    influence = torch.einsum('nsd,nsmd->nsm', residual, bank['value'])
    dz = 2*probability*(influence-(probability*influence).sum(-1, keepdim=True))
    grad = torch.einsum('nsm,nsmd->nsd', dz, bank['key']).mean(1)
    return cost, grad


def _wr_refine(raw, transformed, base, bank, qh, d):
    rows = raw.shape[0]
    if not 2<=rows<=4096 or d>256 or qh>128 or not bool(bank['active'].any()):
        return base
    value = _deq_params(base).float().reshape(rows, qh, d).transpose(0, 1)
    x = transformed.reshape(rows, qh, d).transpose(0, 1)
    score = (value-x).square().sum(-1)/x.square().sum(-1).clamp_min(1e-20)
    quota = min(max(1, rows//8), max(1, _WR_ITEMS//qh))
    rid = score.topk(quota, dim=1).indices[bank['active']].reshape(-1)
    hid = torch.arange(qh)[:, None].expand(qh, quota)[bank['active']].reshape(-1)
    kh = bank['key'].shape[0]
    selected = {name:bank[name][hid//(qh//kh)] for name in ('key','raw','value','raw_value','logmass')}
    target, _ = _wr_output(raw.reshape(rows, qh, d)[rid,hid], selected['raw'], selected['raw_value'], selected['logmass'])
    bid = hid[:, None]*(d//64)+torch.arange(d//64)[None]
    params = {name:p[rid[:, None], bid] for name,p in base.items()}
    current = _deq_params(params).float()
    step = _params_unit_flat(params)*.25
    initial = current.clone()
    best = _wr_cost_gradient(current, target, selected)
    for _ in range(_WR_ROUNDS):
        _, grad = _wr_cost_gradient(current, target, selected, True)
        direction = torch.where(grad<0, 1., -1.)
        level = (current/step).round()
        estimate = (grad*direction*step).masked_fill((level+direction).abs()>7, float('inf'))
        indices = estimate.topk(min(4,d), largest=False, dim=-1).indices
        origin = current
        changed = False
        for width in (1,min(4,d)):
            coord = indices[:, :width]
            change = (direction*step).gather(1, coord)
            change = torch.where(torch.isfinite(estimate.gather(1,coord)), change, 0.)
            trial = origin.clone().scatter_add_(1, coord, change)
            # Accept the actual packed/deployed values, not continuous proposals.
            trial = _deq_params(_values_to_params(trial, params)).float()
            cost = _wr_cost_gradient(trial, target, selected)
            take = (cost<best)&torch.isfinite(cost)
            current = torch.where(take[:,None], trial, current)
            best = torch.where(take, cost, best)
            changed = changed or bool(take.any())
        if not changed:
            break
    take = (current!=initial).any(-1)
    if not bool(take.any()):
        return base
    packed = _values_to_params(current, params)
    answer = {name:p.clone() for name,p in base.items()}
    for name in answer:
        answer[name][rid[take,None],bid[take]] = packed[name][take]
    return answer


def _wr_guard(samples, parent, qh, kh, d):
    WR_DEBUG.clear(); WR_DEBUG.update(accepted=0, changed_queries=0, error=False)
    if len(samples)<3 or d%64 or d>256 or qh>128:
        return parent
    hold = samples[-1]; rows = hold['q'][0].shape[0]
    if not 4<=rows<=4096:
        return parent
    bank = _wr_fit(samples, parent, qh, kh, d)
    st = parent['q_state']
    rawq = dequantize_nvfp4(*hold['q']).float()
    base = _wr_parent_q(*hold['q'], qh, d, st)
    trial = _wr_refine(rawq, _kt_input(rawq, st, qh, d, 'q'), base, bank, qh, d)
    before, after = _deq_params(base).float(), _deq_params(trial).float()
    ix = torch.where((before!=after).any(-1))[0]
    WR_DEBUG['changed_queries'] = ix.numel()
    if ix.numel()<2:
        return parent
    rawk, rawv = [dequantize_nvfp4(*hold[r]).float() for r in 'kv']
    target = _joint_output(rawq[ix], rawk, rawv, qh, kh, d, ix, query_subset=True)
    key = _deq_params(hif4_dynamic_quantize_k(*hold['k'], kh, d, parent['k_state']))
    value = _deq_params(hif4_dynamic_quantize_v(*hold['v'], kh, d, parent['v_state']))
    costs = []
    for q in (before[ix],after[ix]):
        error = (_joint_output(q,key,value,qh,kh,d,ix,query_subset=True)-target).reshape(ix.numel(),qh,d).square().mean(-1)
        costs.append(torch.stack((error[::2].mean(0),error[1::2].mean(0))))
    take = (costs[1]<costs[0]*.995).all(0)&torch.isfinite(costs[1]).all(0)
    WR_DEBUG.update(accepted=int(take.sum()), ratios=(costs[1].mean(0)/costs[0].mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):
        return parent
    bank['active'] = take
    return dict(parent, q_state=dict(st, wr=bank))


def hif4_calibration_attention(samples, qh, kh, d):
    parent = _wr_parent_calibration(samples, qh, kh, d)
    try:
        return _wr_guard(samples, parent, qh, kh, d)
    except (RuntimeError, ValueError) as exc:
        WR_DEBUG.update(error=True,error_type=type(exc).__name__)
        return parent


def hif4_dynamic_quantize_q(quant, scale, qh, d, state):
    base = _wr_parent_q(quant, scale, qh, d, state)
    if not isinstance(state,dict) or 'wr' not in state:
        return base
    raw = dequantize_nvfp4(quant,scale).float()
    return _wr_refine(raw,_kt_input(raw,state,qh,d,'q'),base,state['wr'],qh,d)


def _joint_quant_queries(hold, state, qh, kh, d, ix):
    if 'wr' not in state:
        return _wr_parent_queries(hold,state,qh,kh,d,ix)
    return _deq_params(hif4_dynamic_quantize_q(*hold['q'],qh,d,state))[ix]
