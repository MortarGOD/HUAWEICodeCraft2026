"""Value feedback stratified by current-token energy, with parent-MSE caps."""
_STRATA_BINS = 8
_STRATA_ROUNDS = 8
_STRATA_CAP = 1.10
STRATA_DEBUG = {}
_strata_parent_calibration = hif4_calibration_attention
_strata_parent_v = hif4_dynamic_quantize_v


def _strata_proposal(x, params, kh, d, bins):
    rows=x.shape[0]
    if rows<=1 or rows>4096 or bins>rows:
        return params
    raw=x.reshape(rows,kh,d).transpose(0,1).contiguous()
    value=_deq_params(params).reshape(rows,kh,d).transpose(0,1).contiguous()
    step=(_params_unit_flat(params)*.25).reshape(rows,kh,d).transpose(0,1).contiguous()
    energy=raw.square().mean(-1)
    order=energy.argsort(-1)
    index=order[:,:,None].expand(kh,rows,d)
    raw,value,step=[v.gather(1,index) for v in (raw,value,step)]
    error=raw-value
    reference=error.square().sum(1).clamp_min(1e-30)
    # Dynamic quantiles, not calibration-token identities or cached answers.
    offsets=[rows*i//bins for i in range(bins+1)]
    for i in range(bins):
        start,stop=offsets[i],offsets[i+1]
        xg,sg,vg=raw[:,start:stop],step[:,start:stop],value[:,start:stop]
        eg=xg-vg
        n=stop-start
        score=eg.sum(1).square()+eg.square().sum(1)
        shift=eg.mean(1)
        best=vg.clone()
        for unused in range(_STRATA_ROUNDS):
            trial=((xg+shift[:,None,:])/sg).round().clamp(-7,7)*sg
            te=xg-trial
            next_score=te.sum(1).square()+te.square().sum(1)
            take=next_score<score
            best=torch.where(take[:,None,:],trial,best)
            score=torch.minimum(score,next_score)
            shift+=te.mean(1)*.75
        value[:,start:stop]=best
    mse=(raw-value).square().sum(1)
    valid=(mse<=reference*_STRATA_CAP+1e-20)
    original=_deq_params(params).reshape(rows,kh,d).transpose(0,1).gather(1,index)
    value=torch.where(valid[:,None,:],value,original)
    value=torch.empty_like(value).scatter_(1,index,value).transpose(0,1).reshape(rows,kh*d)
    return _values_to_params(value,params)


def _strata_guard(samples,states,qh,kh,d):
    hold=samples[-1];rows=hold['v'][0].shape[0]
    if rows<4 or rows>4096 or not states['q_state'].get('gq'):
        return states
    raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    ix=torch.linspace(0,rows-1,min(64,rows)).long()
    target=_joint_output(*raw,qh,kh,d,ix)
    q=_joint_quant_queries(hold,states['q_state'],qh,kh,d,ix)
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,states['k_state']))
    parent=_strata_parent_v(*hold['v'],kh,d,states['v_state'])
    trial=_strata_proposal(raw[2],parent,kh,d,_STRATA_BINS)
    costs=[]
    for params in (parent,trial):
        err=(_joint_output(q,k,_deq_params(params),qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        costs.append(torch.stack((err[::2].mean(0),err[1::2].mean(0))))
    mask=(costs[1]<costs[0]*.995).all(0)&torch.isfinite(costs[1]).all(0)
    STRATA_DEBUG.update(heads=int(mask.sum()),ratios=(costs[1].mean(0)/costs[0].mean(0).clamp_min(1e-30)).tolist())
    if not bool(mask.any()):return states
    return dict(states,v_state=dict(strata_mask=mask,strata_bins=_STRATA_BINS,parent=states['v_state']))


def hif4_calibration_attention(samples,qh,kh,d):
    states=_strata_parent_calibration(samples,qh,kh,d)
    STRATA_DEBUG.clear()
    return _strata_guard(samples,states,qh,kh,d) if len(samples)>1 else states


def hif4_dynamic_quantize_v(q,s,kh,d,state):
    if not isinstance(state,dict) or 'strata_mask' not in state:
        return _strata_parent_v(q,s,kh,d,state)
    parent=_strata_parent_v(q,s,kh,d,state['parent'])
    trial=_strata_proposal(dequantize_nvfp4(q,s).float(),parent,kh,d,int(state['strata_bins']))
    mask=state['strata_mask'].repeat_interleave(d//64)[None,:,None,None,None]
    return {key:torch.where(mask,trial[key],parent[key]) for key in parent}
