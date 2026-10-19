"""Value-only conditional attention moments and cross-error gradient student."""
AM_DEBUG={}
_AM_UNCERTAINTY=True
_AM_GRADIENT_GAIN=0.
_AM_ROUNDS=8
_am_parent_calibration=hif4_calibration_attention
_am_parent_v=hif4_dynamic_quantize_v


def _am_features(v,st,kh,d):
    x=v.reshape(-1,kh,d).transpose(0,1)
    z=((x-st['center'][:,None])/st['scale'][:,None]).clamp(-8.,8.)
    small=z.gather(2,st['index'][:,None].expand(kh,z.shape[1],-1))
    pos=torch.arange(z.shape[1])
    extra=torch.stack((z.square().mean(-1).clamp_min(1e-8).log(),z.abs().mean(-1),
        (pos==0).float()[None].expand(kh,-1),(pos<8).float()[None].expand(kh,-1)),-1)
    return torch.cat((torch.ones(kh,z.shape[1],1),small,small.square(),extra),-1)


def _am_cal_data(sample,parent,qh,kh,d,count):
    t=sample['v'][0].shape[0];ix=torch.linspace(0,t-1,min(count,t)).long()
    rawq=dequantize_nvfp4(sample['q'][0][ix],sample['q'][1][ix]).float()
    rawk=dequantize_nvfp4(*sample['k']).float();v=dequantize_nvfp4(*sample['v']).float()
    q=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
    prob=_vr_probs(q,k,qh,kh,d);truth=_vr_probs(rawq,rawk,qh,kh,d)
    vb=v.reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    cross=(prob-truth)@vb
    gradient=(prob.transpose(1,2)@cross)/ix.numel()
    gradient=gradient.reshape(kh,qh//kh,t,d).mean(1)
    byhead=prob.reshape(kh,qh//kh,ix.numel(),t)
    moments=torch.stack((byhead.mean((1,2)),byhead.square().mean((1,2))),-1)
    return v,moments,gradient,(q,k,truth@vb,ix)


def _am_fit(samples,parent,qh,kh,d):
    pool=[s for s in samples[:-1] if 4<=s['v'][0].shape[0]<=2048 and s['v'][0].numel()<=4194304][-3:]
    if len(pool)<2:return None
    if len(pool)==3:pool=[pool[0],pool[-1]]
    summaries=[]
    for s in pool:
        t=s['v'][0].shape[0];ix=torch.linspace(0,t-1,min(256,t)).long()
        summaries.append(dequantize_nvfp4(s['v'][0][ix],s['v'][1][ix]).float().reshape(-1,kh,d).transpose(0,1))
    summary=torch.cat(summaries,1);center=summary.mean(1)
    scale=(summary-center[:,None]).square().mean(1).sqrt().clamp_min(1e-8)
    st=dict(center=center.contiguous(),scale=scale.contiguous(),index=scale.argsort(-1,descending=True)[:,:min(16,d)].contiguous())
    xs=[];ys=[];gs=[];weights=[]
    for s in pool:
        v,moments,gradient,_=_am_cal_data(s,parent,qh,kh,d,16)
        t=v.shape[0];f=_am_features(v,st,kh,d)
        y=(moments*torch.tensor([float(t),float(t*t)])[None,None]).clamp_min(1e-8).log()
        rms=v.reshape(t,kh,d).square().mean(0).sqrt().clamp_min(1e-8)
        xs.append(f);ys.append(y);gs.append(gradient*t/rms[:,None])
        weights.append((moments[:,:,0]*t).sqrt().clamp(.1,10.))
    x=torch.cat(xs,1);y=torch.cat(ys,1);g=torch.cat(gs,1);w=torch.cat(weights,1)
    fs=x.square().mean(1).sqrt().clamp_min(1e-6);z=x/fs[:,None]
    h=z.transpose(1,2)@(z*w[:,:,None])/z.shape[1]+torch.eye(z.shape[-1])*.05
    target=torch.cat((y,g),-1)
    beta=torch.linalg.solve(h,z.transpose(1,2)@(target*w[:,:,None])/z.shape[1])
    st.update(feature_scale=fs.contiguous(),beta=beta.contiguous())
    return st


def _am_moments(v,st,kh,d):
    t=v.shape[0];f=_am_features(v,st,kh,d)/st['feature_scale'][:,None]
    prediction=f@st['beta']
    a=torch.softmax(prediction[:,:,0].clamp(-12.,12.),1)
    b=prediction[:,:,1].clamp(-12.,12.).exp()/(t*t)
    b=torch.maximum(a.square(),torch.minimum(a,b))
    variance=(b-a.square()) if st['uncertainty'] else torch.zeros_like(a)
    variance=variance+.01/(t*t)
    rms=v.reshape(t,kh,d).square().mean(0).sqrt().clamp_min(1e-8)
    gradient=prediction[:,:,2:]*rms[:,None]/t
    return a,variance,gradient


def _am_search(x,base,step,a,variance,gradient,rounds):
    value=base.clone();error=value-x
    mean=(a[:,:,None]*error).sum(1)
    curvature=(a.square()+variance)[:,:,None].clamp_min(1e-30)
    for _ in range(rounds):
        g=a[:,:,None]*mean[:,None]+variance[:,:,None]*error+gradient
        candidate=torch.round((value-g/curvature)/step).clamp(-7.,7.)*step
        delta=candidate-value
        cost=2*g*delta+curvature*delta.square()
        best,index=cost.min(1);take=best<0
        ii=index[:,None,:]
        change=delta.gather(1,ii).squeeze(1)*take
        previous=value.gather(1,ii).squeeze(1)
        value.scatter_(1,ii,(previous+change)[:,None])
        error=value-x
        mean=mean+a.gather(1,index)*change
    return value


def _am_quant(v,parent,st,kh,d):
    t=v.shape[0];x=v.reshape(t,kh,d).transpose(0,1)
    base=_deq_params(parent).reshape(t,kh,d).transpose(0,1)
    step=(_params_unit_flat(parent)*.25).reshape(t,kh,d).transpose(0,1)
    a,var,g=_am_moments(v,st,kh,d)
    # The student cross-gradient is uncertain: bound it by observable quantization noise.
    noise=(base-x).square().mean(1).sqrt()
    cap=a[:,:,None]*noise[:,None]
    g=g.clamp(-cap,cap)*st['gradient_gain']
    result=_am_search(x,base,step,a,var,g,_AM_ROUNDS)
    active=st.get('active',torch.ones(kh,dtype=torch.bool)).repeat_interleave(d//64)
    out=_values_to_params(result.transpose(0,1).reshape(v.shape).contiguous(),parent)
    return {k:torch.where(active.reshape(1,-1,*([1]*(p.ndim-2))),out[k],p) for k,p in parent.items()}


def _am_guard(samples,parent,qh,kh,d,model=None):
    AM_DEBUG.clear();AM_DEBUG.update(error=False,adopted=0)
    if len(samples)<3 or d%64 or not parent['q_state'].get('gq'):return parent
    hold=samples[-1];t=hold['v'][0].shape[0]
    if not 4<=t<=4096 or hold['v'][0].numel()>4194304:return parent
    st=_am_fit(samples,parent,qh,kh,d) if model is None else dict(model)
    if st is None:return parent
    st=dict(st,uncertainty=_AM_UNCERTAINTY,gradient_gain=_AM_GRADIENT_GAIN)
    v,moments,_,data=_am_cal_data(hold,parent,qh,kh,d,32)
    q,k,target,ix=data
    p=_vr_probs(q,k,qh,kh,d)
    base=_am_parent_v(*hold['v'],kh,d,parent['v_state'])
    trial=_am_quant(v,base,st,kh,d)
    def loss(params):
        values=_deq_params(params).reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        e=(p@values-target).square().mean(-1).reshape(kh,qh//kh,-1).mean(1)
        return torch.stack((e[:,::2].mean(1),e[:,1::2].mean(1)))
    old=loss(base);new=loss(trial);take=(new<old*.995).all(0)&torch.isfinite(new).all(0)
    predicted,_,_=_am_moments(v,st,kh,d)
    AM_DEBUG.update(adopted=int(take.sum()),hold_ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist(),
        probability_l1=(predicted-moments[:,:,0]).abs().sum(1).tolist(),
        uniform_l1=(1./t-moments[:,:,0]).abs().sum(1).tolist())
    if not bool(take.any()):return parent
    st.update(active=take,parent=parent['v_state'])
    return dict(parent,v_state=dict(moment=st))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_am_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _am_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        AM_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    if not isinstance(v_state,dict) or 'moment' not in v_state:
        return _am_parent_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)
    st=v_state['moment'];base=_am_parent_v(v_quant,v_scale,kv_num_heads,head_dim,st['parent'])
    if not 4<=v_quant.shape[0]<=4096 or v_quant.numel()>4194304:return base
    return _am_quant(dequantize_nvfp4(v_quant,v_scale).float(),base,st,kv_num_heads,head_dim)
