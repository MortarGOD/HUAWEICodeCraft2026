"""v397: bounded cross-sequence selection of existing Q/K combinations."""
CS_DEBUG = {}
_CS_WORK_LIMIT = 4_000_000_000
_cs_parent_calibration = hif4_calibration_attention
_cs_parent_base = _oj_parent_calibration
_cs_parent_kt = _kt_guard


def _oj_parent_calibration(samples, qh, kh, d):
    parent = _cs_parent_base(samples, qh, kh, d)
    return dict(parent, _cs_logit_q=parent['q_state'])


def _kt_guard(samples, parent, qh, kh, d):
    answer = _cs_parent_kt(samples, parent, qh, kh, d)
    return dict(answer, _cs_plain_k=parent['k_state'])


def _cs_clean(parent):
    return {name:parent[name] for name in ('q_state','k_state','v_state')}


def _cs_candidates(captured):
    parent = _cs_clean(captured)
    q = dict(parent['q_state'])
    oldq = captured.get('_cs_logit_q',q)
    for name in ('H','u'):
        if name in oldq:
            q[name] = oldq[name]
        else:
            q.pop(name,None)
    key = dict(captured.get('_cs_plain_k',parent['k_state']))
    if 'key_center_mask' in parent['k_state']:
        key['key_center_mask'] = parent['k_state']['key_center_mask']
    return [parent,dict(parent,q_state=q),dict(parent,k_state=key),dict(parent,q_state=q,k_state=key)]


def _cs_pick(costs):
    # candidates x distinct sequences. Parent is always candidate zero.
    if costs.ndim!=2 or costs.shape[1]<3 or not bool(torch.isfinite(costs[0]).all()):
        return 0
    if bool((costs[0]<=1e-30).any()):
        return 0
    ratio = costs/costs[0].clamp_min(1e-30)[None]
    valid = torch.isfinite(ratio).all(1)&(ratio.mean(1)<.995)&(ratio.max(1).values<=1.01)&((ratio<1.).sum(1)>=2)
    valid[0] = False
    log = ratio.clamp_min(1e-30).log()
    risk = log.mean(1)+.5*log.max(1).values.clamp_min(0)
    risk = risk.masked_fill(~valid,float('inf'))
    return int(risk.argmin()) if bool(valid.any()) else 0


def _cs_full_cost(rawq,rawk,rawv,query,key,value,qh,kh,d):
    total = torch.zeros((),dtype=torch.float64)
    rows = rawq.shape[0]
    for start in range(0,rows,128):
        end = min(rows,start+128);ix=torch.arange(start,end)
        target = _joint_output(rawq[start:end],rawk,rawv,qh,kh,d,ix,query_subset=True)
        output = _joint_output(query[start:end],key,value,qh,kh,d,ix,query_subset=True)
        total += (output-target).square().double().sum()
    return total/(rows*qh*d)


def _cs_select(samples,captured,qh,kh,d):
    parent = _cs_clean(captured)
    CS_DEBUG.clear();CS_DEBUG.update(chosen=0,screen_choice=0,error=False,reason='ineligible')
    if len(samples)<3 or d%64 or not all(parent[r+'_state'].get('gq') for r in 'qk'):
        return parent
    eligible = [i for i,s in enumerate(samples) if 4<=s['q'][0].shape[0]<=4096]
    if len(eligible)<3 or eligible[-1]!=len(samples)-1:
        return parent
    indices = [eligible[0],eligible[len(eligible)//2],eligible[-1]]
    work = sum(qh*d*samples[i]['q'][0].shape[0]**2 for i in indices)
    CS_DEBUG.update(indices=indices,work=work)
    if work>_CS_WORK_LIMIT:
        CS_DEBUG['reason']='work_budget'
        return parent
    candidates = _cs_candidates(captured)
    costs = torch.zeros(4,3,dtype=torch.float64)
    records = []
    for j,si in enumerate(indices):
        sample = samples[si];t=sample['q'][0].shape[0]
        raw = [dequantize_nvfp4(*sample[r]).float() for r in 'qkv']
        ix=torch.linspace(0,t-1,min(t,32)).long()
        target=_joint_output(*raw,qh,kh,d,ix)
        # Each distinct operand is quantized once per sequence.
        queries=[_joint_quant_queries(sample,candidates[i]['q_state'],qh,kh,d,ix) for i in (0,1)]
        keys=[_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,candidates[i]['k_state'])).float() for i in (0,2)]
        value=_deq_params(hif4_dynamic_quantize_v(*sample['v'],kh,d,parent['v_state'])).float()
        for ci in range(4):
            output=_joint_output(queries[ci%2],keys[ci//2],value,qh,kh,d,ix,query_subset=True)
            costs[ci,j]=(output-target).square().double().mean()
        records.append((sample,raw,keys,value))
    chosen = _cs_pick(costs)
    CS_DEBUG.update(screen_choice=chosen,screen_ratios=(costs/costs[0].clamp_min(1e-30)).tolist(),reason='screen_reject')
    if not chosen:
        return parent
    full = torch.empty(2,3,dtype=torch.float64)
    for j,(sample,raw,keys,value) in enumerate(records):
        oldq=_deq_params(hif4_dynamic_quantize_q(*sample['q'],qh,d,parent['q_state'])).float()
        newq=oldq if chosen%2==0 else _deq_params(hif4_dynamic_quantize_q(*sample['q'],qh,d,candidates[chosen]['q_state'])).float()
        full[0,j]=_cs_full_cost(*raw,oldq,keys[0],value,qh,kh,d)
        full[1,j]=_cs_full_cost(*raw,newq,keys[chosen//2],value,qh,kh,d)
    take = _cs_pick(full)==1
    CS_DEBUG.update(chosen=chosen if take else 0,full_ratios=(full[1]/full[0].clamp_min(1e-30)).tolist(),reason='accepted' if take else 'full_reject')
    return candidates[chosen] if take else parent


@torch.inference_mode()
def hif4_calibration_attention(samples,qh,kh,d):
    captured = _cs_parent_calibration(samples,qh,kh,d)
    try:
        return _cs_select(samples,captured,qh,kh,d)
    except (RuntimeError,ValueError) as exc:
        CS_DEBUG.update(error=True,error_type=type(exc).__name__)
        return _cs_clean(captured)
