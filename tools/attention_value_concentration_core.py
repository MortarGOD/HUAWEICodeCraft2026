"""Calibrated exchangeable attention risk for Value quantization."""
VC_DEBUG={}
_VC_MODE=1
_vc_parent_calibration=hif4_calibration_attention
_vc_parent_v=hif4_dynamic_quantize_v


def _vc_fit(samples,qh,kh,d):
    counts=[];concentrations=[]
    for s in samples[:-1][-2:]:
        t=s['k'][0].shape[0]
        if t<=1:continue
        ix=torch.linspace(0,t-1,min(t,32)).long()
        q=dequantize_nvfp4(s['q'][0][ix],s['q'][1][ix]).float().reshape(-1,qh,d).transpose(0,1)
        k=dequantize_nvfp4(*s['k']).float().reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        p=torch.softmax(q@k.transpose(1,2)/(d**.5),-1)
        c=(t*p.square().sum(-1).mean(-1)).reshape(kh,qh//kh).mean(-1).clamp(1.,float(t))
        counts.append(t);concentrations.append(c)
    if len(counts)<2:return None
    c0,c1=concentrations;t0,t1=counts
    power=((c1-1).clamp_min(1e-6).log()-(c0-1).clamp_min(1e-6).log())/max(float(torch.log(torch.tensor(t1/t0))),1e-6)
    power=power.clamp(0.,1.)
    return dict(vc_count=int(t1),vc_excess=(c1-1).contiguous(),vc_power=power.contiguous())


def _vc_quant(raw,parent,st,kh,d,mode):
    if mode==3:return _dyn_table(raw,None,has_scale=False)
    t=raw.shape[0];p=parent
    c=(1+st['vc_excess'].float()*(t/st['vc_count'])**st['vc_power'].float()).clamp(1.,float(t))
    penalty=(c-1)/(t-c).clamp_min(1e-6)
    if mode==2:penalty=(c/t).clamp_min(1./t)
    penalty=penalty.repeat_interleave(d)[None]
    base=_deq_params(p);best=base;err=raw-base
    cost=err.mean(0,keepdim=True).square()+penalty*err.square().mean(0,keepdim=True)
    step=_params_unit_flat(p)*.25;shift=err.mean(0,keepdim=True)
    for _ in range(8):
        value=torch.round((raw+shift)/step).clamp(-7,7)*step
        error=raw-value
        trial=error.mean(0,keepdim=True).square()+penalty*error.square().mean(0,keepdim=True)
        take=trial<cost
        best=torch.where(take,value,best);cost=torch.minimum(cost,trial)
        shift=shift+.75*error.mean(0,keepdim=True)
    return _values_to_params(best,p)


def _vc_guard(samples,parent,qh,kh,d):
    VC_DEBUG.clear();VC_DEBUG.update(accepted=0,error=False)
    hold=samples[-1];t=hold['v'][0].shape[0]
    if len(samples)<3 or d%64 or not 4<=t<=4096:return parent
    fit=_vc_fit(samples,qh,kh,d)
    if fit is None:return parent
    ix=torch.linspace(0,t-1,min(t,64)).long()
    raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
    raw.extend(dequantize_nvfp4(*hold[r]).float() for r in 'kv')
    target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
    q=_deq_params(hif4_dynamic_quantize_q(*hold['q'],qh,d,parent['q_state']))[ix]
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    base=_vc_parent_v(*hold['v'],kh,d,parent['v_state'])
    def cost(value):
        e=(_joint_output(q,k,_deq_params(value),qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    trial=_vc_quant(raw[2],base,fit,kh,d,_VC_MODE)
    old=cost(base);new=cost(trial);take=(new<old*.995).all(0)
    VC_DEBUG.update(accepted=int(take.sum()),concentration=(1+fit['vc_excess']).tolist(),ratios=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):return parent
    return dict(parent,v_state=dict(fit,vc_mode=_VC_MODE,vc_active=take,vc_fallback=parent['v_state']))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_vc_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _vc_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        VC_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    if not isinstance(v_state,dict) or 'vc_mode' not in v_state or not 4<=v_quant.shape[0]<=4096:
        fallback=v_state.get('vc_fallback',v_state) if isinstance(v_state,dict) else v_state
        return _vc_parent_v(v_quant,v_scale,kv_num_heads,head_dim,fallback)
    parent=_vc_parent_v(v_quant,v_scale,kv_num_heads,head_dim,v_state['vc_fallback'])
    raw=dequantize_nvfp4(v_quant,v_scale).float()
    new=_vc_quant(raw,parent,v_state,kv_num_heads,head_dim,v_state['vc_mode'])
    mask=v_state['vc_active'].repeat_interleave(head_dim//64)
    return {name:torch.where(mask.reshape(1,-1,*([1]*(a.ndim-2))),a,parent[name]) for name,a in new.items()}
