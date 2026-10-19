"""Current-Key conditional sensitivity weights for Softmax-null translation."""
SS_DEBUG={}
_SS_MODE=1
_SS_PROB_GATE=False
_ss_parent_calibration=hif4_calibration_attention
_ss_parent_k=hif4_dynamic_quantize_k


def _ss_weights(prob,mode):
    if mode==1:score=prob.square()
    elif mode==2:score=prob*(1-prob)
    else:
        distance=((1-prob).square()+(prob.square().sum(-1,keepdim=True)-prob.square()).clamp_min(0))
        score=prob.square()*distance
    mass=score.mean(1)
    total=mass.sum(-1,keepdim=True)
    weight=torch.where(total>1e-30,mass/total.clamp_min(1e-30),torch.full_like(mass,1/prob.shape[-1]))
    return .9*weight+.1/prob.shape[-1]


def _ss_quant(raw,st,kh,d,mode):
    y=_kt_input(raw,st,kh,d,'k').reshape(-1,kh,d)
    key=raw.reshape(-1,kh,d).transpose(0,1)
    prob=torch.softmax(st['kt_probe'].float()@key.transpose(1,2),-1)
    weight=(_ss_weights(prob,mode) if mode<=3 else .9*prob.mean(1)+.1/raw.shape[0]).transpose(0,1)[:,:,None]
    center=(y*weight).sum(0,keepdim=True)
    if mode in (4,5):
        anchor=prob.mean(1).argmax(-1)
        center=y.gather(0,anchor[None,:,None].expand(1,kh,d))
    elif mode==6:
        values,order=y.sort(dim=0)
        cumulative=weight.expand_as(y).gather(0,order).cumsum(0)
        index=(cumulative<.5).sum(0).clamp_max(raw.shape[0]-1)
        center=values.gather(0,index[None])
    elif mode==7:center=y.median(0,keepdim=True).values
    rounds={4:0,8:4,9:6,10:10,11:12,12:16}.get(mode,8)
    rate=.5 if mode==13 else (1.5 if mode==14 else 1.)
    for _ in range(rounds):
        z=y-center
        error=_deq_params(_dyn_table(z.reshape(raw.shape),None,has_scale=False)).reshape(y.shape)-z
        center=center-rate*(error*weight).sum(0,keepdim=True)
    plain={k:v for k,v in st.items() if k not in ('joint_transform','ks','rot','hc_matrix')}
    return _kt_fixed_grid((y-center).reshape(raw.shape).contiguous(),plain,kh,d,'k')


def _ss_guard(samples,parent,qh,kh,d):
    SS_DEBUG.clear();SS_DEBUG.update(accepted=0,error=False)
    st=parent['k_state'];hold=samples[-1];t=hold['k'][0].shape[0]
    if not 4<=t<=4096 or not st.get('gq') or 'u' not in st:return parent
    proposed=dict(st)
    if 'kt_probe' not in proposed:
        probes=[]
        for sample in samples[:-1][-2:]:
            count=sample['q'][0].shape[0];ixq=torch.linspace(0,count-1,min(count,16)).long()
            qr=dequantize_nvfp4(sample['q'][0][ixq],sample['q'][1][ixq]).float().reshape(-1,kh,qh//kh,d)
            probes.append(qr.permute(1,0,2,3).reshape(kh,-1,d))
        proposed['kt_probe']=torch.cat(probes,1)/(d**.5)
    ix=torch.linspace(0,t-1,min(t,32)).long()
    raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
    raw.extend(dequantize_nvfp4(*hold[r]).float() for r in 'kv')
    target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(_ss_parent_k(*hold['k'],kh,d,st))
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
    def cost(key):
        e=(_joint_output(q,key,v,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    trial=_deq_params(_ss_quant(raw[1],proposed,kh,d,_SS_MODE))
    old=cost(k);new=cost(trial)
    take=(new<old*.995).all(0)
    if _SS_PROB_GATE:
        def probability(query,key):
            query=query.reshape(-1,kh,qh//kh,d).permute(1,2,0,3)
            key=key.reshape(-1,kh,d).permute(1,2,0)[:,None]
            return torch.softmax((query@key)/(d**.5),-1)
        reference=probability(raw[0],raw[1])
        def pcost(key):
            error=(probability(q,key)-reference).square().sum(-1).mean(1).transpose(0,1)
            return torch.stack((error[::2].mean(0),error[1::2].mean(0)))
        before=pcost(k);after=pcost(trial)
        extra=(after<before*.95).all(0)&(new<=old*1.01).all(0)
        take=take|extra
        SS_DEBUG.update(probability_ratio=(after.mean(0)/before.mean(0).clamp_min(1e-30)).tolist(),probability_accepted=int(extra.sum()))
    SS_DEBUG.update(accepted=int(take.sum()),ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):return parent
    return dict(parent,k_state=dict(proposed,ss_mode=_SS_MODE,ss_active=take,ss_fallback=st))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_ss_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _ss_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        SS_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    if not isinstance(k_state,dict) or 'ss_mode' not in k_state or not 4<=k_quant.shape[0]<=4096:
        fallback=k_state.get('ss_fallback',k_state) if isinstance(k_state,dict) else k_state
        return _ss_parent_k(k_quant,k_scale,kv_num_heads,head_dim,fallback)
    raw=dequantize_nvfp4(k_quant,k_scale).float()
    new=_ss_quant(raw,k_state,kv_num_heads,head_dim,k_state['ss_mode'])
    if bool(k_state['ss_active'].all()):return new
    old=_ss_parent_k(k_quant,k_scale,kv_num_heads,head_dim,k_state['ss_fallback'])
    mask=k_state['ss_active'].repeat_interleave(head_dim//64)
    return {name:torch.where(mask.reshape(1,-1,*([1]*(a.ndim-2))),a,old[name]) for name,a in new.items()}
