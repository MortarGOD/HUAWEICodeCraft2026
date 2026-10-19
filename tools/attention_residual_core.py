"""Conditional task regression: fit correction through the deployed attention operator.

Research mixin; no other current role is read by the dynamic interface.
"""
_VR_RIDGE = .1
_VR_TARGET = 'cross'
_VR_FIT_SEQUENCES = 4
_VR_ROUTE = False
VR_DEBUG = {}
_vr_parent_calibration = hif4_calibration_attention
_vr_parent_dynamic = hif4_dynamic_quantize_v


def _vr_probs(q, k, qh, kh, d):
    qb = q.reshape(-1,qh,d).transpose(0,1)
    kb = k.reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    return torch.softmax(qb @ kb.transpose(1,2)/(d**.5),-1)


def _vr_features(v, st, kh, d):
    x=v.reshape(-1,kh,d).transpose(0,1)
    f=(x-st['center'][:,None]) / st['rms'][:,None]
    return torch.cat((f,torch.ones(kh,x.shape[1],1)),2)


def _vr_quant(v, parent, st, kh, d, gain):
    if not bool((gain!=0).any()):return parent
    f=_vr_features(v,st,kh,d)
    delta=f @ st['beta']
    # Trust region limits extrapolation relative to observable quantization noise.
    base=_deq_params(parent)
    cap=(base-v).square().mean(0).sqrt().reshape(kh,1,d)*2
    delta=delta.clamp(-cap,cap)*gain[:,None,None]
    target=v+delta.transpose(0,1).reshape(v.shape)
    step=_params_unit_flat(parent)*.25
    best=base;error=target-best
    cost=error.mean(0).square()+8/v.shape[0]*error.square().mean(0)
    shift=error.mean(0)
    for _ in range(8):
        candidate=torch.round((target+shift)/step).clamp(-7,7)*step
        e=target-candidate
        cc=e.mean(0).square()+8/v.shape[0]*e.square().mean(0)
        take=cc<cost
        best=torch.where(take[None],candidate,best)
        cost=torch.minimum(cost,cc)
        shift=shift+.75*e.mean(0)
    active=gain.repeat_interleave(d)!=0
    best=torch.where(active[None],best,base)
    result=_values_to_params(best,parent)
    blocks=(gain!=0).repeat_interleave(d//64)
    return {key:torch.where(blocks.reshape(1,-1,*([1]*(value.ndim-2))),value,parent[key])
            for key,value in result.items()}


def _vr_fit(samples, states, qh, kh, d):
    selected_indices=[i for i,s in enumerate(samples[:-1]) if 4<=s['v'][0].shape[0]<=4096][-_VR_FIT_SEQUENCES:]
    selected=[samples[i] for i in selected_indices]
    if len(selected)<2:return None
    vs=[dequantize_nvfp4(*s['v']).float() for s in selected]
    summary=torch.cat([v[torch.linspace(0,v.shape[0]-1,min(128,v.shape[0])).long()] for v in vs])
    vh=summary.reshape(-1,kh,d).transpose(0,1)
    center=vh.mean(1);rms=(vh-center[:,None]).square().mean(1).sqrt().clamp_min(1e-8)
    st=dict(center=center.contiguous(),rms=rms.contiguous())
    aa=torch.zeros(kh,d+1,d+1);ab=torch.zeros(kh,d+1,d)
    cache={i:k for i,k in states.get('_vr_cache',[])}
    for si,s,v in zip(selected_indices,selected,vs):
        t=v.shape[0];ix=torch.linspace(0,t-1,min(32,t)).long()
        rawq=dequantize_nvfp4(s['q'][0][ix],s['q'][1][ix]).float()
        rawk=dequantize_nvfp4(*s['k']).float()
        q=_joint_quant_queries(s,states['q_state'],qh,kh,d,ix)
        k=cache.get(si)
        if k is None:k=_deq_params(hif4_dynamic_quantize_k(*s['k'],kh,d,states['k_state']))
        p=_vr_probs(q,k,qh,kh,d)
        pr=_vr_probs(rawq,rawk,qh,kh,d)
        v0=_deq_params(_vr_parent_dynamic(*s['v'],kh,d,states['v_state']))
        vb=v.reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        v0b=v0.reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        residual=pr@vb-p@(vb if _VR_TARGET=='cross' else v0b)
        a=p@_vr_features(v,st,kh,d).repeat_interleave(qh//kh,0)
        a=a.reshape(kh,-1,d+1);b=residual.reshape(kh,-1,d)
        # Per-sequence/head normalization prevents high-energy sequences dominating.
        noise=(pr@vb-p@v0b).square().mean((1,2)).reshape(kh,-1).mean(1).clamp_min(1e-12)
        w=1/(noise*a.shape[1])
        aa+=(a.transpose(1,2)@a)*w[:,None,None]
        ab+=(a.transpose(1,2)@b)*w[:,None,None]
    reg=aa.diagonal(dim1=-2,dim2=-1).mean(1).clamp_min(1e-10)*_VR_RIDGE
    st['beta']=torch.linalg.solve(aa+reg[:,None,None]*torch.eye(d+1),ab).contiguous()
    if _VR_TARGET=='zero':st['beta'].zero_()
    return st


def _vr_guard(samples,parent,qh,kh,d):
    VR_DEBUG.clear();VR_DEBUG.update(accepted=0,error=False,target=_VR_TARGET)
    if d%64 or not parent['q_state'].get('gq'):return parent
    hold=samples[-1];t=hold['v'][0].shape[0]
    if not 4<=t<=4096:return parent
    eligible=_vm_eligible(hold,qh,kh,d) if _VR_ROUTE else torch.ones(kh,dtype=torch.bool)
    VR_DEBUG['eligible_heads']=int(eligible.sum())
    if not bool(eligible.any()):return parent
    st=_vr_fit(samples,parent,qh,kh,d)
    if st is None:return parent
    raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    p0=_vr_parent_dynamic(*hold['v'],kh,d,parent['v_state'])
    ix=torch.linspace(0,t-1,min(64,t)).long()
    target=_joint_output(*raw,qh,kh,d,ix)
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    def loss(p):
        e=(_joint_output(q,k,_deq_params(p),qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    original=loss(p0);best=original.clone();gains=torch.zeros(kh)
    ratios=[]
    for gain in (.25,.5,1.):
        trial=_vr_quant(raw[2],p0,st,kh,d,torch.full((kh,),gain))
        cost=loss(trial)
        take=(cost<original*.99).all(0)&(cost.mean(0)<best.mean(0))&eligible
        best=torch.where(take[None],cost,best);gains=torch.where(take,gain,gains)
        ratios.append((cost.mean(0)/original.mean(0).clamp_min(1e-30)).tolist())
    VR_DEBUG.update(accepted=int((gains!=0).sum()),gains=gains.tolist(),ratios=ratios)
    if not bool((gains!=0).any()):return parent
    st.update(gain=gains,fallback=parent['v_state'])
    return dict(parent,v_state=dict(vr=st))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_vr_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:
        answer=_vr_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        VR_DEBUG.update(error=True,error_type=type(exc).__name__)
        answer=parent
    # Calibration-local K reuse never reaches frozen state or a dynamic role.
    return {key:value for key,value in answer.items() if key!='_vr_cache'}


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    if not isinstance(v_state,dict) or 'vr' not in v_state:
        return _vr_parent_dynamic(v_quant,v_scale,kv_num_heads,head_dim,v_state)
    st=v_state['vr'];p=_vr_parent_dynamic(v_quant,v_scale,kv_num_heads,head_dim,st['fallback'])
    t=v_quant.shape[0]
    if not 4<=t<=4096:return p
    return _vr_quant(dequantize_nvfp4(v_quant,v_scale).float(),p,st,kv_num_heads,head_dim,st['gain'])
