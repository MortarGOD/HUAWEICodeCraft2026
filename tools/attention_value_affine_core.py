"""Calibration output affine reconstruction, realized through legal V input quantization."""
_VA_MARGIN = .005
VA_DEBUG = {}
_va_parent_calibration = hif4_calibration_attention
_va_parent_v = _dyn_v


def _va_quantize(x):
    p=_dyn_table(x,None,has_scale=False)
    if x.shape[0]<=1:return p
    unit=_params_unit_flat(p);best=_deq_params(p);penalty=1./x.shape[0]
    error=x-best;best_cost=error.mean(0).square()+penalty*error.square().mean(0)
    shift=error.mean(0)
    for _ in range(8):
        proposal=torch.round((x+shift)/unit*4.).clamp(-7.,7.)*(unit*.25)
        error=x-proposal;mean=error.mean(0)
        cost=mean.square()+penalty*error.square().mean(0)
        accept=cost<best_cost
        best=torch.where(accept[None],proposal,best)
        best_cost=torch.minimum(cost,best_cost);shift=shift+mean*.75
    return _values_to_params(best,p)


def _va_apply(x,gain,bias):
    return _va_quantize(x*gain.reshape(1,-1)+bias.reshape(1,-1))


def _dyn_v(quant,scale,state,kvh,dh):
    if not isinstance(state,dict) or 'value_affine_gain' not in state or quant.shape[0]<=1 or quant.shape[0]>4096:
        return _va_parent_v(quant,scale,state,kvh,dh)
    x=dequantize_nvfp4(quant,scale).float()
    candidate=_va_apply(x,state['value_affine_gain'],state['value_affine_bias'])
    active=state['value_affine_active']
    if bool(active.all()):return candidate
    parent=_va_parent_v(quant,scale,state,kvh,dh)
    mask=active.repeat_interleave(dh//64)[None,:,None,None,None]
    return {key:torch.where(mask,candidate[key],parent[key]) for key in parent}


def _va_fit(samples,states,qh,kh,d):
    targets=[];outputs=[];vrms=[]
    for sample in samples[:-1][-2:]:
        rows=sample['v'][0].shape[0]
        if not 2<=rows<=4096:continue
        ix=torch.linspace(0,rows-1,min(rows,32)).long()
        raw=[dequantize_nvfp4(*sample[r]).float() for r in 'qkv']
        target=_joint_output(*raw,qh,kh,d,ix)
        q=(_joint_quant_queries(sample,states['q_state'],qh,kh,d,ix) if states['q_state'].get('gq') else
           _deq_params(hif4_dynamic_quantize_q(*sample['q'],qh,d,states['q_state']))[ix])
        k=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,states['k_state']))
        v=_deq_params(_va_parent_v(*sample['v'],states['v_state'],kh,d))
        out=_joint_output(q,k,v,qh,kh,d,ix,query_subset=True)
        targets.append(target.permute(0,2,1,3).reshape(-1,kh,d))
        outputs.append(out.permute(0,2,1,3).reshape(-1,kh,d))
        vrms.append(raw[2].square().mean(0).reshape(kh,d))
    if not targets:return None
    target=torch.cat(targets);out=torch.cat(outputs)
    tm,om=target.mean(0),out.mean(0)
    centered=out-om;variance=centered.square().mean(0)
    ridge=variance.mean(-1,keepdim=True).clamp_min(1e-20)*.1
    gain=(((target-tm)*centered).mean(0)+ridge)/(variance+ridge)
    gain=gain.clamp(.9,1.1)
    bound=torch.stack(vrms).mean(0).sqrt()*.1
    bias=(tm-gain*om).clamp(-bound,bound)
    bias_only=(tm-om).clamp(-bound,bound)
    if not bool(torch.isfinite(gain).all() and torch.isfinite(bias).all()):return None
    return gain,bias,bias_only


def _va_guard(samples,states,qh,kh,d):
    hold=samples[-1];rows=hold['v'][0].shape[0]
    if not 4<=rows<=4096:return states
    fit=_va_fit(samples,states,qh,kh,d)
    if fit is None:return states
    gain,bias,bias_only=fit
    options=[(torch.ones_like(gain),torch.zeros_like(bias)),
             (torch.ones_like(gain),.5*bias_only),(1+.5*(gain-1),.5*bias),
             (gain,bias)]
    ix=torch.linspace(0,rows-1,min(rows,64)).long()
    raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    target=_joint_output(*raw,qh,kh,d,ix)
    q=(_joint_quant_queries(hold,states['q_state'],qh,kh,d,ix) if states['q_state'].get('gq') else
       _deq_params(hif4_dynamic_quantize_q(*hold['q'],qh,d,states['q_state']))[ix])
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,states['k_state']))
    costs=[]
    for i,(g,b) in enumerate(options):
        vp=_va_parent_v(*hold['v'],states['v_state'],kh,d) if i==0 else _va_apply(raw[2],g,b)
        out=_joint_output(q,k,_deq_params(vp),qh,kh,d,ix,query_subset=True)
        e=(out-target).square().mean((-1,-2))
        costs.append(torch.stack((e[::2].mean(0),e[1::2].mean(0))))
    costs=torch.stack(costs);best=costs[0];choice=torch.zeros(kh,dtype=torch.int64)
    for i in range(1,len(options)):
        take=torch.isfinite(costs[i]).all(0)&(costs[i]<best*(1-_VA_MARGIN)).all(0)
        choice=torch.where(take,i,choice);best=torch.where(take[None],costs[i],best)
    VA_DEBUG.update(choice=choice.tolist(),costs=costs.tolist())
    if not bool(choice.any()):return states
    finalg=options[0][0].clone();finalb=options[0][1].clone()
    for i,(g,b) in enumerate(options[1:],1):
        finalg=torch.where((choice==i)[:,None],g,finalg)
        finalb=torch.where((choice==i)[:,None],b,finalb)
    answer=dict(states);old=states['v_state'] if isinstance(states['v_state'],dict) else {}
    answer['v_state']=dict(old,value_affine_gain=finalg.contiguous(),value_affine_bias=finalb.contiguous(),
                           value_affine_active=(choice!=0))
    return answer


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    states=_va_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    VA_DEBUG.clear();VA_DEBUG.update(error=False)
    if len(calib_qkv_list)<2 or head_dim%64:return states
    try:return _va_guard(calib_qkv_list,states,q_num_heads,kv_num_heads,head_dim)
    except Exception:
        VA_DEBUG['error']=True
        return states
