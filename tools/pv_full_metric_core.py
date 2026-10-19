"""Full-key, deployed-QKV output metrics; offline research, statically embeddable.

Unlike the earlier PVGN probe, neither fitting nor guarding subsamples keys.
Only query rows and output-Jacobian columns are sampled. GQA heads are separate
for Q and grouped for K. This changes calibration metrics, not interface access.
"""
_PVFULL_BLEND = 0.5
_PVFULL_QUERIES = 8
_PVFULL_RANK = 32
_PVFULL_MAX_ROWS = 4096
_PVFULL_GUARD_QUERIES = 64
_PVFULL_SAFE_RECOVERY = True
PVFULL_DEBUG = {}


def _pvfull_head_metrics(q, k, v, sketch):
    """Q:[queries,d], K/V:[all_keys,d]; raw Gauss--Newton blocks."""
    d = q.shape[-1]
    p = torch.softmax(q @ k.T / (d**0.5), -1)
    projected = v @ sketch
    ov = p @ projected
    weighted = p[:, :, None] * (projected[None] - ov[:, None])
    # Jq's output dimension is sketched, never its key dimension.
    jf = k.T @ weighted.permute(1, 0, 2).reshape(k.shape[0], -1)
    gq = jf @ jf.T / (d * q.shape[0])
    ofull = p @ v
    distances = (v.square().sum(-1)[None] - 2 * (ofull @ v.T)
                 + ofull.square().sum(-1)[:, None]).clamp_min(0)
    weights = (p.square() * distances).sum(-1)
    gk = q.T @ (weights[:, None] * q) / (d * q.shape[0])
    return gq, gk


def _pvfull_normalize(h):
    tr = h.diagonal(dim1=-2, dim2=-1).sum(-1)
    return h / tr.clamp_min(1e-30)[..., None, None]


def _pvfull_fit(samples, states, qh, kh, d):
    gq = torch.zeros(qh, d, d)
    gk = torch.zeros(kh, d, d)
    # A fixed Rademacher output sketch has E[S S.T]=I; it does not remove keys.
    generator = torch.Generator().manual_seed(0x315F + d)
    rank = min(d, _PVFULL_RANK)
    sketch = (torch.randint(0, 2, (d, rank), generator=generator).float()*2-1) / rank**0.5
    count = 0
    for sample in samples[:-1][-2:]:
        rows = sample['q'][0].shape[0]
        if rows < 2 or rows > _PVFULL_MAX_ROWS:
            continue
        ix = torch.linspace(0, rows-1, min(rows, _PVFULL_QUERIES)).long()
        q = _joint_quant_queries(sample, states['q_state'], qh, kh, d, ix)
        k = _deq_params(hif4_dynamic_quantize_k(*sample['k'], kh, d, states['k_state']))
        v = _deq_params(hif4_dynamic_quantize_v(*sample['v'], kh, d, states['v_state']))
        q = q.reshape(-1, qh, d).transpose(0, 1)
        k = k.reshape(-1, kh, d).transpose(0, 1)
        v = v.reshape(-1, kh, d).transpose(0, 1)
        cq, ck = torch.zeros_like(gq), torch.zeros_like(gk)
        for h in range(qh):
            hv = h // (qh//kh)
            cq[h], hk = _pvfull_head_metrics(q[h], k[hv], v[hv], sketch)
            ck[hv] += hk
        # Equal contribution per calibration sample; avoid short-T trace dominance.
        gq += _pvfull_normalize(cq)
        gk += _pvfull_normalize(ck)
        count += 1
    PVFULL_DEBUG['fit_samples'] = count
    return gq, gk


def _pvfull_proposal(state, metric, nh):
    u = state['u'].float()
    if _PVFULL_SAFE_RECOVERY:
        inverse_u = torch.linalg.solve_triangular(u,torch.eye(u.shape[-1]).expand_as(u),upper=True)
        damped = inverse_u @ inverse_u.transpose(-1,-2)
    else:
        damped = torch.cholesky_inverse(u, upper=True)
    ridge = damped.diagonal(dim1=-2, dim2=-1).mean(-1) * (GPTQ_DAMP/(1+GPTQ_DAMP))
    raw = damped - ridge[:, None, None] * torch.eye(damped.shape[-1])
    raw = ((raw + raw.transpose(-1, -2))*0.5).repeat_interleave(nh//raw.shape[0], 0)
    trace = raw.diagonal(dim1=-2, dim2=-1).sum(-1)
    norm = _pvfull_normalize(metric) * trace[:, None, None]
    valid = torch.isfinite(norm).all((-1, -2)) & (metric.diagonal(dim1=-2, dim2=-1).sum(-1)>1e-20)
    blended = torch.where(valid[:, None, None],
                          (1-_PVFULL_BLEND)*raw + _PVFULL_BLEND*norm, raw)
    new_u = _upper_cholesky_inv(blended)
    if new_u is None:
        PVFULL_DEBUG.setdefault('rejected_metrics',[]).append(dict(
            heads=nh, min_eigenvalue=float(torch.linalg.eigvalsh(blended).min()),
            max_diagonal=float(blended.diagonal(dim1=-2,dim2=-1).max())))
        return None
    answer = dict(state, u=new_u.contiguous())
    if 'H' in state:
        answer['H'] = blended.bfloat16().contiguous()
    return answer


def _pvfull_guard(hold, states, proposed, qh, kh, d):
    rows = hold['q'][0].shape[0]
    ix = torch.linspace(0, rows-1, min(rows, _PVFULL_GUARD_QUERIES)).long()
    raw = [dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    target = _joint_output(*raw, qh, kh, d, ix).reshape(-1, qh, d)
    value = _deq_params(hif4_dynamic_quantize_v(*hold['v'], kh, d, states['v_state']))
    queries = [_joint_quant_queries(hold, s['q_state'], qh, kh, d, ix) for s in (states, proposed)]
    keys = [_deq_params(hif4_dynamic_quantize_k(*hold['k'], kh, d, s['k_state'])) for s in (states, proposed)]
    costs = torch.empty(2, 2, 2, qh)
    for qi in range(2):
        for ki in range(2):
            out = _joint_output(queries[qi], keys[ki], value, qh, kh, d, ix, query_subset=True).reshape(-1,qh,d)
            err = (out-target).square().mean(-1)
            costs[qi,ki] = torch.stack((err[::2].mean(0),err[1::2].mean(0)))
    if not bool(torch.isfinite(costs).all()):
        PVFULL_DEBUG.update(nonfinite_guard=True,
                            query_finite=[bool(torch.isfinite(x).all()) for x in queries],
                            key_finite=[bool(torch.isfinite(x).all()) for x in keys],
                            value_finite=bool(torch.isfinite(value).all()),
                            target_finite=bool(torch.isfinite(target).all()))
        return states
    qm, km = torch.zeros(qh,dtype=torch.bool), torch.zeros(kh,dtype=torch.bool)
    for hv in range(kh):
        sl = slice(hv*(qh//kh),(hv+1)*(qh//kh))
        q0 = (costs[1,0,:,sl]<costs[0,0,:,sl]*.995).all(0)
        best = torch.where(q0[None],costs[1,0,:,sl],costs[0,0,:,sl])
        q1 = (costs[1,1,:,sl]<costs[0,1,:,sl]*.995).all(0)
        other = torch.where(q1[None],costs[1,1,:,sl],costs[0,1,:,sl])
        take = bool((other.sum(-1)<best.sum(-1)*.995).all() and (other<=best*1.01).all())
        km[hv],qm[sl] = take, q1 if take else q0
    answer = dict(states)
    for role,mask in (('q',qm),('k',km)):
        if not bool(mask.any()):
            continue
        old,new = states[role+'_state'],proposed[role+'_state']
        out = dict(old)
        for field in ('u','H'):
            if field in old and field in new:
                before = old[field].repeat_interleave(mask.numel()//old[field].shape[0],0)
                out[field] = torch.where(mask[:,None,None],new[field],before).contiguous()
        answer[role+'_state'] = out
    PVFULL_DEBUG.update(q_heads=int(qm.sum()), k_heads=int(km.sum()),
                        hold_ratios=(costs.mean((2,3))/costs[0,0].mean().clamp_min(1e-30)).tolist())
    return answer


_pvfull_parent_calibration = hif4_calibration_attention


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    states = _pvfull_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    PVFULL_DEBUG.clear()
    PVFULL_DEBUG.update(q_heads=0,k_heads=0,error=False)
    if (len(calib_qkv_list)<2 or not states['q_state'].get('gq') or not states['k_state'].get('gq')
            or not 4<=calib_qkv_list[-1]['q'][0].shape[0]<=_PVFULL_MAX_ROWS):
        return states
    try:
        gq,gk = _pvfull_fit(calib_qkv_list,states,q_num_heads,kv_num_heads,head_dim)
        if not PVFULL_DEBUG.get('fit_samples'):
            return states
        pq = _pvfull_proposal(states['q_state'],gq,q_num_heads)
        pk = _pvfull_proposal(states['k_state'],gk,kv_num_heads)
        if pq is None or pk is None:
            return states
        return _pvfull_guard(calib_qkv_list[-1],states,dict(states,q_state=pq,k_state=pk),
                             q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        PVFULL_DEBUG.update(error=True, error_type=type(exc).__name__)
        return states
