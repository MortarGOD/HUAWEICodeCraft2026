"""Explicit leading positions plus an exchangeable background risk model."""
PR_DEBUG={}
_PR_ANCHORS=1
_PR_RESCALE=True
_pr_parent_calibration=hif4_calibration_attention
_pr_parent_v=hif4_dynamic_quantize_v


def _pr_moments(prob,anchors):
    # prob: [KV heads, sampled GQA queries, tokens].
    n=prob.shape[-1]-anchors
    mass=torch.cat((prob[:,:,:anchors],prob[:,:,anchors:].sum(-1,keepdim=True)),-1)
    gram=mass.transpose(1,2)@mass/mass.shape[1]
    squares=prob[:,:,anchors:].square().sum(-1).mean(1)
    kappa=(n*squares/gram[:,-1,-1].clamp_min(1e-30)).clamp(1.,float(n))
    return gram.contiguous(),kappa.contiguous()


def _pr_odds(prob,anchors):
    n=prob.shape[-1]-anchors;mass=prob[:,:,anchors:].sum(-1)
    odds=(n*prob[:,:,:anchors]/mass[:,:,None].clamp_min(1e-12)).clamp_min(0.)
    kappa=(n*prob[:,:,anchors:].square().sum(-1)/mass.square().clamp_min(1e-30)).clamp(1.,float(n))
    return odds.contiguous(),kappa.contiguous()


def _pr_resolve(model,t,anchors):
    n=t-anchors;odds=model['odds'];denominator=n+odds.sum(-1,keepdim=True)
    mass=torch.cat((odds/denominator,n/denominator),-1)
    gram=mass.transpose(1,2)@mass/mass.shape[1]
    weights=mass[:,:,-1].square()
    kappa=(weights*model['background_kappa']).sum(1)/weights.sum(1).clamp_min(1e-30)
    return dict(gram=gram,kappa=kappa.clamp(1.,float(n)))


def _pr_gradient(error,gram,kappa,anchors):
    n=error.shape[1]-anchors
    background=error[:,anchors:];mean=background.mean(1,keepdim=True)
    summary=torch.cat((error[:,:anchors],mean),1)
    weighted=gram@summary
    beta=gram[:,-1,-1]*(kappa.clamp(1.,float(n))-1)/(n*max(1,n-1))
    ridge=.001/(error.shape[1]**2)
    gradient=torch.cat((weighted[:,:anchors],weighted[:,-1:]/n+beta[:,None,None]*(background-mean)),1)+ridge*error
    diagonal=torch.cat((gram.diagonal(dim1=1,dim2=2)[:,:anchors],
        (gram[:,-1,-1]/(n*n)+beta*(1-1/n))[:,None].expand(-1,n)),1)+ridge
    objective=(summary*weighted).sum((1,2))+beta*(background-mean).square().sum((1,2))+ridge*error.square().sum((1,2))
    return gradient,diagonal,objective


def _pr_search(raw,base,step,model,anchors,rounds=8):
    value=base.clone();gram=model['gram'];kappa=model['kappa']
    for _ in range(rounds):
        gradient,diagonal,_=_pr_gradient(value-raw,gram,kappa,anchors)
        curvature=diagonal[:,:,None].clamp_min(1e-30)
        candidate=((value-gradient/curvature)/step).round().clamp(-7,7)*step
        delta=candidate-value;cost=2*gradient*delta+curvature*delta.square()
        best,index=cost.min(1);ii=index[:,None]
        move=delta.gather(1,ii).squeeze(1)*(best<0)
        value.scatter_(1,ii,(value.gather(1,ii).squeeze(1)+move)[:,None])
    return value


def _pr_fit(samples,parent,qh,kh,d,anchors):
    models=[]
    for sample in samples[-3:-1]:
        t=sample['k'][0].shape[0]
        if not max(4,anchors+2)<=t<=2048 or sample['v'][0].numel()>4194304:continue
        ix=torch.linspace(0,t-1,min(16,t)).long()
        q=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,ix)
        k=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
        prob=_vr_probs(q,k,qh,kh,d).reshape(kh,-1,t)
        gram,kappa=_pr_moments(prob,anchors)
        odds,background_kappa=_pr_odds(prob,anchors)
        models.append(dict(length=t,gram=gram,kappa=kappa,odds=odds,background_kappa=background_kappa))
    return models


def _pr_quant(raw,base,st,kh,d):
    t=raw.shape[0];anchors=st['anchors']
    if t<=anchors+1:return base
    model=min(st['models'],key=lambda item:max(t,item['length'])/min(t,item['length']))
    if st.get('rescale',False):model=_pr_resolve(model,t,anchors)
    shape=(t,kh,d)
    result=_pr_search(raw.reshape(shape).transpose(0,1),_deq_params(base).reshape(shape).transpose(0,1),
        (_params_unit_flat(base)*.25).reshape(shape).transpose(0,1),model,anchors)
    out=_values_to_params(result.transpose(0,1).reshape(raw.shape).contiguous(),base)
    active=st.get('active',torch.ones(kh,dtype=torch.bool)).repeat_interleave(d//64)
    return {k:torch.where(active.reshape(1,-1,*([1]*(v.ndim-2))),out[k],v) for k,v in base.items()}


def _pr_guard(samples,parent,qh,kh,d):
    PR_DEBUG.clear();PR_DEBUG.update(error=False,adopted=0)
    anchors=_PR_ANCHORS
    if len(samples)<3 or d%64 or not parent['q_state'].get('gq'):return parent
    hold=samples[-1];t=hold['v'][0].shape[0]
    if not max(4,anchors+2)<=t<=4096 or hold['v'][0].numel()>4194304:return parent
    models=_pr_fit(samples,parent,qh,kh,d,anchors)
    if not models:return parent
    st=dict(anchors=anchors,models=models,rescale=_PR_RESCALE)
    ix=torch.linspace(0,t-1,min(32,t)).long()
    rawq=dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()
    rawk=dequantize_nvfp4(*hold['k']).float();rawv=dequantize_nvfp4(*hold['v']).float()
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    p=_vr_probs(q,k,qh,kh,d);target=_vr_probs(rawq,rawk,qh,kh,d)@rawv.reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    base=_pr_parent_v(*hold['v'],kh,d,parent['v_state']);trial=_pr_quant(rawv,base,st,kh,d)
    def loss(params):
        values=_deq_params(params).reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        e=(p@values-target).square().mean(-1).reshape(kh,qh//kh,-1).mean(1)
        return torch.stack((e[:,::2].mean(1),e[:,1::2].mean(1)))
    old=loss(base);new=loss(trial);active=(new<old*.995).all(0)&torch.isfinite(new).all(0)
    PR_DEBUG.update(adopted=int(active.sum()),hold_ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist(),
        model_diagonals=[item['gram'].diagonal(dim1=1,dim2=2).tolist() for item in models])
    if not bool(active.any()):return parent
    st.update(active=active,parent=parent['v_state'])
    return dict(parent,v_state=dict(position_risk=st))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_pr_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _pr_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        PR_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    if not isinstance(v_state,dict) or 'position_risk' not in v_state:
        return _pr_parent_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)
    st=v_state['position_risk'];base=_pr_parent_v(v_quant,v_scale,kv_num_heads,head_dim,st['parent'])
    if not 4<=v_quant.shape[0]<=4096 or v_quant.numel()>4194304:return base
    return _pr_quant(dequantize_nvfp4(v_quant,v_scale).float(),base,st,kv_num_heads,head_dim)
