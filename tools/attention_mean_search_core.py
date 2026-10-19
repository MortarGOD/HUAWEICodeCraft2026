"""Uniform-attention quadratic search with exact two-coordinate exchanges."""
MS_DEBUG={}
_MS_PAIRS=2
_MS_PENALTY=.01
_ms_parent_calibration=hif4_calibration_attention
_ms_parent_v=hif4_dynamic_quantize_v


def _ms_search(x,base,step,penalty,pairs,singles=8):
    # T^2 times the uniform-risk objective: (sum e)^2 + penalty * sum(e^2).
    value=base.clone();error=value-x;total=error.sum(1)
    for _ in range(singles):
        g=total[:,None]+penalty*error
        proposal=((value-g/(1+penalty))/step).round().clamp(-7,7)*step
        delta=proposal-value;cost=2*g*delta+(1+penalty)*delta.square()
        best,index=cost.min(1);ii=index[:,None]
        change=delta.gather(1,ii).squeeze(1)*(best<0)
        value.scatter_(1,ii,(value.gather(1,ii).squeeze(1)+change)[:,None])
        error=value-x;total=error.sum(1)
    for _ in range(pairs):
        g=total[:,None]+penalty*error
        best=torch.zeros_like(total);bi=torch.zeros_like(total,dtype=torch.int64);bj=bi.clone()
        di=torch.zeros_like(total);dj=di.clone()
        # A small anchor shortlist; each anchor's partner is searched over all tokens.
        for direction in (-1.,1.):
            move=((value/step).round()+direction).clamp(-7,7)*step-value
            cost=2*g*move+(1+penalty)*move.square()
            anchors=cost.topk(min(2,value.shape[1]),dim=1,largest=False).indices
            for rank in range(anchors.shape[1]):
                ai=anchors[:,rank,:];ii=ai[:,None]
                da=move.gather(1,ii).squeeze(1);ca=cost.gather(1,ii).squeeze(1)
                proposal=((value-(g+da[:,None])/(1+penalty))/step).round().clamp(-7,7)*step
                db=proposal-value
                combined=ca[:,None]+2*(g+da[:,None])*db+(1+penalty)*db.square()
                combined.scatter_(1,ii,float('inf'))
                cb,aj=combined.min(1);take=cb<best
                best=torch.where(take,cb,best);bi=torch.where(take,ai,bi);bj=torch.where(take,aj,bj)
                di=torch.where(take,da,di);dj=torch.where(take,db.gather(1,aj[:,None]).squeeze(1),dj)
        value.scatter_add_(1,bi[:,None],di[:,None]);value.scatter_add_(1,bj[:,None],dj[:,None])
        error=value-x;total=error.sum(1)
    return value


def _ms_quant(raw,base,st,kh,d):
    t=raw.shape[0]
    x=raw.reshape(t,kh,d).transpose(0,1)
    values=_deq_params(base).reshape(t,kh,d).transpose(0,1)
    step=(_params_unit_flat(base)*.25).reshape(t,kh,d).transpose(0,1)
    result=_ms_search(x,values,step,st['penalty'],st['pairs'])
    out=_values_to_params(result.transpose(0,1).reshape(raw.shape).contiguous(),base)
    active=st.get('active',torch.ones(kh,dtype=torch.bool)).repeat_interleave(d//64)
    return {k:torch.where(active.reshape(1,-1,*([1]*(p.ndim-2))),out[k],p) for k,p in base.items()}


def _ms_guard(samples,parent,qh,kh,d):
    MS_DEBUG.clear();MS_DEBUG.update(error=False,adopted=0)
    if len(samples)<3 or d%64 or not parent['q_state'].get('gq'):return parent
    hold=samples[-1];t=hold['v'][0].shape[0]
    if not 4<=t<=4096 or hold['v'][0].numel()>4194304:return parent
    ix=torch.linspace(0,t-1,min(32,t)).long()
    rawq=dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()
    rawk=dequantize_nvfp4(*hold['k']).float();rawv=dequantize_nvfp4(*hold['v']).float()
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    p=_vr_probs(q,k,qh,kh,d);truth=_vr_probs(rawq,rawk,qh,kh,d)
    target=truth@rawv.reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    base=_ms_parent_v(*hold['v'],kh,d,parent['v_state'])
    st=dict(penalty=_MS_PENALTY,pairs=_MS_PAIRS)
    trial=_ms_quant(rawv,base,st,kh,d)
    def loss(params):
        value=_deq_params(params).reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        e=(p@value-target).square().mean(-1).reshape(kh,qh//kh,-1).mean(1)
        return torch.stack((e[:,::2].mean(1),e[:,1::2].mean(1)))
    old=loss(base);new=loss(trial);take=(new<old*.995).all(0)&torch.isfinite(new).all(0)
    MS_DEBUG.update(adopted=int(take.sum()),hold_ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):return parent
    st.update(active=take,parent=parent['v_state'])
    return dict(parent,v_state=dict(mean_search=st))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_ms_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _ms_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        MS_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    if not isinstance(v_state,dict) or 'mean_search' not in v_state:
        return _ms_parent_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)
    st=v_state['mean_search'];base=_ms_parent_v(v_quant,v_scale,kv_num_heads,head_dim,st['parent'])
    if not 4<=v_quant.shape[0]<=4096 or v_quant.numel()>4194304:return base
    return _ms_quant(dequantize_nvfp4(v_quant,v_scale).float(),base,st,kv_num_heads,head_dim)
