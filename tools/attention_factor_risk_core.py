"""Overrides for the moment research harness: V-conditioned probability factors.

Requires attention_moment_core, retaining its independent-sequence PV guard.
No test Q/K. Frozen calibration query prototypes are absorbed into a regression.
"""
_AF_RIDGE=.1
_AF_SEARCH_RIDGE=.1
_AF_RANK=16


def _af_features(x,st):
    z=((x-st['center'][:,None])/st['scale'][:,None]).clamp(-8.,8.)
    return torch.cat((torch.ones_like(z[:,:,:1]),z),-1)


def _am_fit(samples,parent,qh,kh,d):
    pool=[s for s in samples[:-1] if 4<=s['v'][0].shape[0]<=2048
          and s['v'][0].numel()<=4194304][-2:]
    if len(pool)<2:return None
    proto=[];xs=[];ks=[]
    for sample in pool:
        t=sample['v'][0].shape[0]
        ix=torch.linspace(0,t-1,min(16,t)).long()
        q=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,ix)
        q=q.reshape(-1,kh,qh//kh,d).permute(1,0,2,3).reshape(kh,-1,d)
        proto.append(q)
        xs.append(dequantize_nvfp4(*sample['v']).float().reshape(t,kh,d).transpose(0,1))
        ks.append(_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
                  .reshape(t,kh,d).transpose(0,1))
    proto=torch.cat(proto,1)
    ix=torch.linspace(0,proto.shape[1]-1,min(_AF_RANK,proto.shape[1])).long()
    proto=proto[:,ix];x=torch.cat(xs,1);k=torch.cat(ks,1)
    center=x.mean(1);scale=(x-center[:,None]).square().mean(1).sqrt().clamp_min(1e-6)
    st=dict(center=center.contiguous(),scale=scale.contiguous())
    features=_af_features(x,st)
    target=(k@proto.transpose(1,2))/(d**.5)
    h=features.transpose(1,2)@features/x.shape[1]
    h=h+torch.eye(h.shape[-1])*_AF_RIDGE
    beta=torch.linalg.solve(h,features.transpose(1,2)@target/x.shape[1])
    st.update(beta=beta.contiguous(),search_ridge=_AF_SEARCH_RIDGE)
    return st


def _af_probability(v,st,kh,d):
    x=v.reshape(-1,kh,d).transpose(0,1)
    logits=(_af_features(x,st)@st['beta']).transpose(1,2)
    return logits.clamp(-40.,40.).softmax(-1)


def _am_moments(v,st,kh,d):
    p=_af_probability(v,st,kh,d)
    return p.mean(1),p.var(1,unbiased=False),torch.zeros(kh,v.shape[0],d)


def _af_search(raw,base,step,p,ridge,rounds):
    value=base.clone();error=value-raw
    residual=p@error;rank=p.shape[1];lam=ridge/(raw.shape[1]**2)
    diagonal=p.square().mean(1)[:,:,None]+lam
    for _ in range(rounds):
        gradient=p.transpose(1,2)@residual/rank+lam*error
        candidate=((value-gradient/diagonal.clamp_min(1e-30))/step).round().clamp(-7,7)*step
        delta=candidate-value;cost=2*gradient*delta+diagonal*delta.square()
        best,index=cost.min(1);ii=index[:,None]
        move=delta.gather(1,ii).squeeze(1)*(best<0)
        value.scatter_add_(1,ii,move[:,None]);error=value-raw
        residual=residual+p.gather(2,index[:,None].expand(-1,rank,-1))*move[:,None]
    return value


def _am_quant(v,parent,st,kh,d):
    t=v.shape[0];raw=v.reshape(t,kh,d).transpose(0,1)
    base=_deq_params(parent).reshape(t,kh,d).transpose(0,1)
    step=(_params_unit_flat(parent)*.25).reshape(t,kh,d).transpose(0,1)
    result=_af_search(raw,base,step,_af_probability(v,st,kh,d),st['search_ridge'],8)
    out=_values_to_params(result.transpose(0,1).reshape(v.shape).contiguous(),parent)
    active=st.get('active',torch.ones(kh,dtype=torch.bool)).repeat_interleave(d//64)
    return {k:torch.where(active.reshape(1,-1,*([1]*(p.ndim-2))),out[k],p) for k,p in parent.items()}
