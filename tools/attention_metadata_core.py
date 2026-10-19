"""Bounded legal HiF4 block re-encoding under a calibrated quadratic metric.

Independent implementation: conditional Newton proposal + exact incremental
quadratic acceptance. It is not AQLM's codebook or BoA's weight quantizer.
"""
_META_MAX_T = 4096
_META_MARGIN = .005
_META_RERANK = True
META_DEBUG = {}
_meta_parent_calibration = hif4_calibration_attention


def _meta_prepare(state, d):
    if not isinstance(state, dict) or not state.get('gq') or 'H' not in state:
        return None
    h = state['H'].float()
    h = (h+h.transpose(-1,-2))*.5
    norm = h.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-20)
    h = h/norm[:,None,None] + .01*torch.eye(d)[None]
    if not bool(torch.isfinite(h).all()):
        return None
    blocks = torch.stack([h[:,i:i+64,i:i+64] for i in range(0,d,64)],1)
    try:
        chol, info = torch.linalg.cholesky_ex(blocks)
        if bool(info.any()):
            return None
        eye = torch.eye(64).expand_as(chol)
        inverse = torch.cholesky_solve(eye, chol)
    except Exception:
        return None
    if not bool(torch.isfinite(inverse).all()):
        return None
    return dict(state, meta_h=h.contiguous(), meta_inv=inverse.contiguous())


def _meta_refine(x, params, state, heads, d, original_rows):
    if (original_rows<=1 or original_rows>_META_MAX_T or
            not isinstance(state,dict) or 'meta_h' not in state):
        return params
    active = state.get('meta_active', torch.ones(heads,dtype=torch.bool))
    if not bool(active.any()):
        return params
    rows=x.shape[0]; nb=d//64
    group=torch.arange(heads)*state['meta_h'].shape[0]//heads
    h=state['meta_h'][group]; inverse=state['meta_inv'][group]
    z=_deq_params(params).reshape(rows,heads,d).transpose(0,1).contiguous()
    reference=x.reshape(rows,heads,d).transpose(0,1)
    gradient=(z-reference)@h
    result={key:value.clone() for key,value in params.items()}
    # Each step preserves the accepted previous blocks and updates their effect
    # on all remaining blocks. No frozen stale gradient acceptance.
    for block in range(nb):
        start=block*64; stop=start+64
        old=z[:,:,start:stop]
        target=old-gradient[:,:,start:stop]@inverse[:,block]
        target=target.transpose(0,1).reshape(rows,heads*64)
        weight=h[:,start:stop,start:stop].diagonal(dim1=-2,dim2=-1).reshape(1,heads*64)
        proposal=None; change=None; values=None
        grids=[(candidate,) for candidate in CAND_GRID] if _META_RERANK else [CAND_GRID]
        for grid in grids:
            trial=_quantize_weighted(target,weight,grid=grid)
            tv=_deq_params(trial).reshape(rows,heads,64).transpose(0,1)
            td=tv-old
            tc=(2*td*gradient[:,:,start:stop]+(td@h[:,start:stop,start:stop])*td).sum(-1)
            if proposal is None:
                proposal,change,values=trial,tc,tv
            else:
                take=tc<change
                proposal={key:torch.where(take.T[:,:,None,None,None],trial[key],proposal[key]) for key in trial}
                change=torch.where(take,tc,change)
                values=torch.where(take[:,:,None],tv,values)
        delta=values-old
        accept=(change < -1e-8)&torch.isfinite(change)&active[:,None]
        delta=torch.where(accept[:,:,None],delta,torch.zeros_like(delta))
        z[:,:,start:stop]=old+delta
        gradient+=delta@h[:,start:stop,:]
        mask=accept.T[:,:,None,None,None]
        for key in result:
            result[key][:,block::nb]=torch.where(mask,proposal[key],result[key][:,block::nb])
    return result


def _meta_guard(samples, states, qh, kh, d):
    hold=samples[-1]; rows=hold['q'][0].shape[0]
    if not 4<=rows<=_META_MAX_T:
        return states
    proposed={role:_meta_prepare(states[role+'_state'],d) for role in 'qk'}
    if all(value is None for value in proposed.values()):
        return states
    ix=torch.linspace(0,rows-1,min(rows,64)).long()
    raw=[dequantize_nvfp4(*hold[role]).float() for role in 'qkv']
    target=_joint_output(*raw,qh,kh,d,ix).reshape(-1,qh,d)
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,states['v_state']))
    qs=[];ks=[]
    for choice in (0,1):
        qst=proposed['q'] if choice and proposed['q'] is not None else states['q_state']
        kst=proposed['k'] if choice and proposed['k'] is not None else states['k_state']
        q=(_joint_quant_queries(hold,qst,qh,kh,d,ix) if qst.get('gq') else
           _deq_params(hif4_dynamic_quantize_q(*hold['q'],qh,d,qst))[ix])
        qs.append(q)
        ks.append(_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,kst)))
    costs=torch.empty(2,2,2,kh)
    for qi in (0,1):
        for ki in (0,1):
            out=_joint_output(qs[qi],ks[ki],v,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d)
            error=(out-target).square().reshape(-1,kh,(qh//kh)*d).mean(-1)
            costs[qi,ki]=torch.stack((error[::2].mean(0),error[1::2].mean(0)))
    qmask=torch.zeros(kh,dtype=torch.bool);kmask=torch.zeros_like(qmask)
    best=costs[0,0].clone()
    for qi,ki in ((1,0),(0,1),(1,1)):
        take=torch.isfinite(costs[qi,ki]).all(0)&(costs[qi,ki]<best*(1-_META_MARGIN)).all(0)
        best=torch.where(take[None],costs[qi,ki],best)
        qmask=torch.where(take,bool(qi),qmask);kmask=torch.where(take,bool(ki),kmask)
    answer=dict(states)
    for role,mask,rep in (('q',qmask,qh//kh),('k',kmask,1)):
        if proposed[role] is not None and bool(mask.any()):
            answer[role+'_state']=dict(proposed[role],meta_active=mask.repeat_interleave(rep))
    META_DEBUG.update(q_heads=int(qmask.sum())*(qh//kh),k_heads=int(kmask.sum()),costs=costs.tolist())
    return answer


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    states=_meta_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    META_DEBUG.clear();META_DEBUG.update(q_heads=0,k_heads=0,error=False)
    if len(calib_qkv_list)<2 or head_dim%64:
        return states
    try:
        return _meta_guard(calib_qkv_list,states,q_num_heads,kv_num_heads,head_dim)
    except Exception:
        META_DEBUG['error']=True
        return states
