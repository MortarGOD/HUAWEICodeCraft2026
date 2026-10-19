"""Output-calibrated per-query-head gain; no changes to the scoring softmax.

The gain acts on quantizer input before legal HiF4 metadata/rounding. We never
multiply dequantized output or E6M2 metadata by an arbitrary illegal factor.
"""
_TEMP_FIT_QUERIES = 16
_TEMP_GUARD_QUERIES = 64
_TEMP_STRENGTHS = (0.5, 1.0)
_TEMP_CENTER_KEYS = True
TEMP_DEBUG = {}


def _temp_heads(x, nh, d):
    return x.reshape(-1, nh, d).transpose(0, 1).contiguous()


def _temp_fit(samples, states, qh, kh, d):
    gradient = torch.zeros(qh)
    curvature = torch.zeros(qh)
    for sample in samples[:-1][-2:]:
        rows = sample["v"][0].shape[0]
        if rows <= 1 or rows > 4096:
            continue
        ix = torch.linspace(0, rows-1, min(rows, _TEMP_FIT_QUERIES)).long()
        raw = [dequantize_nvfp4(*sample[r]).float() for r in "qkv"]
        reference = _joint_output(*raw, qh, kh, d, ix).reshape(ix.numel(), qh, d).transpose(0, 1)
        q = _temp_heads(_joint_quant_queries(sample, states["q_state"], qh, kh, d, ix), qh, d)
        k = _temp_heads(_deq_params(hif4_dynamic_quantize_k(*sample["k"], kh, d, states["k_state"])), kh, d)
        v = _temp_heads(_deq_params(hif4_dynamic_quantize_v(*sample["v"], kh, d, states["v_state"])), kh, d)
        k = k.repeat_interleave(qh//kh, 0)
        v = v.repeat_interleave(qh//kh, 0)
        logits = q @ k.transpose(1, 2) / (d**0.5)
        p = torch.softmax(logits, -1)
        out = p @ v
        derivative = (p * logits) @ v - out * (p * logits).sum(-1, keepdim=True)
        gradient += ((out-reference)*derivative).mean((1, 2))
        curvature += derivative.square().mean((1, 2))
    valid = torch.isfinite(gradient) & torch.isfinite(curvature) & (curvature > 1e-20)
    delta = torch.where(valid, -gradient/curvature.clamp_min(1e-20), torch.zeros_like(gradient))
    return delta.clamp(-0.2, 0.25)


def _temp_guard(hold, states, delta, qh, kh, d):
    rows = hold["v"][0].shape[0]
    ix = torch.linspace(0, rows-1, min(rows, _TEMP_GUARD_QUERIES)).long()
    raw = [dequantize_nvfp4(*hold[r]).float() for r in "qkv"]
    reference = _joint_output(*raw, qh, kh, d, ix).reshape(-1, qh, d)
    k = _deq_params(hif4_dynamic_quantize_k(*hold["k"], kh, d, states["k_state"]))
    v = _deq_params(hif4_dynamic_quantize_v(*hold["v"], kh, d, states["v_state"]))
    parent = states["q_state"]
    gains = [torch.ones(qh)] + [1.0+s*delta for s in _TEMP_STRENGTHS]
    queries = []
    for gain in gains:
        st = dict(parent, temperature_gain=gain)
        queries.append(_joint_quant_queries(hold, st, qh, kh, d, ix))
    keys = [k]
    if _TEMP_CENTER_KEYS:
        centered = dict(states["k_state"], key_center_mask=torch.ones(kh, dtype=torch.bool))
        keys.append(_deq_params(hif4_dynamic_quantize_k(*hold["k"],kh,d,centered)))
    costs = torch.empty(len(keys),len(gains),2,qh)
    for ci,key in enumerate(keys):
        for gi,q in enumerate(queries):
            output = _joint_output(q,key,v,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d)
            err = (output-reference).square().mean(-1)
            costs[ci,gi]=torch.stack((err[::2].mean(0),err[1::2].mean(0)))
    if not bool(torch.isfinite(costs).all()):
        return states
    winners=[]
    for ci in range(len(keys)):
        best=costs[ci,0]
        gain=gains[0].clone()
        for i in range(1,len(gains)):
            accept=(costs[ci,i]<best*0.995).all(0)
            gain=torch.where(accept,gains[i],gain)
            best=torch.where(accept[None],costs[ci,i],best)
        winners.append((best,gain))
    best,gain=winners[0]
    mask=torch.zeros(kh,dtype=torch.bool)
    if len(winners)>1:
        candidate,other_gain=winners[1]
        for hv in range(kh):
            sl=slice(hv*(qh//kh),(hv+1)*(qh//kh))
            take=bool((candidate[:,sl].sum(-1)<best[:,sl].sum(-1)*0.995).all()
                      and (candidate[:,sl]<=best[:,sl]*1.02).all())
            mask[hv]=take
            if take:gain[sl]=other_gain[sl]
    TEMP_DEBUG.update(heads=int((gain!=1).sum()), centered_groups=int(mask.sum()),gains=gain.tolist(),
                      fitted_delta=delta.tolist(),
                      hold_ratios=(costs.mean((2,3))/costs[0,0].mean().clamp_min(1e-30)).tolist())
    answer=dict(states)
    if bool((gain!=1).any()):
        answer['q_state']=dict(parent,temperature_gain=gain.contiguous())
    if bool(mask.any()):
        answer['k_state']=dict(states['k_state'],key_center_mask=mask)
    return answer


_temp_parent_calibration = hif4_calibration_attention


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    states = _temp_parent_calibration(calib_qkv_list, q_num_heads, kv_num_heads, head_dim)
    TEMP_DEBUG.clear()
    TEMP_DEBUG.update(heads=0, error=False)
    if (len(calib_qkv_list)<2 or head_dim%64 or not states["q_state"].get("gq")):
        return states
    rows = calib_qkv_list[-1]["v"][0].shape[0]
    if rows<4 or rows>4096:
        return states
    try:
        delta = _temp_fit(calib_qkv_list,states,q_num_heads,kv_num_heads,head_dim)
        return _temp_guard(calib_qkv_list[-1],states,delta,q_num_heads,kv_num_heads,head_dim)
    except Exception:
        TEMP_DEBUG["error"]=True
        return states
