"""Per-channel nonlinear feature student of calibrated final-output residuals."""
RS_DEBUG={}
_RS_FIT_QUERIES=16
_RS_GUARD_QUERIES=32
_rs_parent_calibration=hif4_calibration_attention
_rs_parent_v=hif4_dynamic_quantize_v


def _rs_features(v,p,kh,d):
    x=v.reshape(-1,kh,d).transpose(0,1)
    b=_deq_params(p).reshape_as(v).reshape(-1,kh,d).transpose(0,1)
    step=(_params_unit_flat(p)*.25).reshape(-1,kh,d).transpose(0,1)
    rms=x.square().mean(1).sqrt().clamp_min(1e-8)
    z=x/rms[:,None];e=(x-b)/rms[:,None]
    token=z.square().mean(2,keepdim=True).clamp_max(16.)
    em=e.mean(1,keepdim=True).expand_as(e)
    f=torch.stack((e,z,z.sign(),em,token*e,token*z,z*z.abs(),
                   step/rms[:,None]*z.sign()),-1)
    return f,rms,step,b


def _rs_quant(v,p,st,kh,d,gain):
    f,rms,step,base=_rs_features(v,p,kh,d)
    delta=(f*st['beta'][:,None]).sum(-1)*rms[:,None]
    delta=delta.clamp(-2*step,2*step)*gain[:,None,None]
    value=torch.round((base+delta)/step).clamp(-7,7)*step
    value=torch.where((gain!=0)[:,None,None],value,base)
    out=_values_to_params(value.transpose(0,1).reshape(v.shape).contiguous(),p)
    active=(gain!=0).repeat_interleave(d//64)
    return {k:torch.where(active.reshape(1,-1,*([1]*(x.ndim-2))),out[k],x) for k,x in p.items()}


def _rs_fit(samples,parent,qh,kh,d):
    aa=torch.zeros(kh,d,8,8);ab=torch.zeros(kh,d,8,1)
    eligible=[s for s in samples[:-1] if 4<=s['v'][0].shape[0]<=2048 and s['v'][0].numel()<=4194304][-3:]
    if len(eligible)==3:eligible=[eligible[0],eligible[-1]]
    if len(eligible)<2:return None
    for s in eligible:
        t=s['v'][0].shape[0];ix=torch.linspace(0,t-1,min(_RS_FIT_QUERIES,t)).long()
        qraw=dequantize_nvfp4(s['q'][0][ix],s['q'][1][ix]).float()
        kraw=dequantize_nvfp4(*s['k']).float();v=dequantize_nvfp4(*s['v']).float()
        q=_joint_quant_queries(s,parent['q_state'],qh,kh,d,ix)
        k=_deq_params(hif4_dynamic_quantize_k(*s['k'],kh,d,parent['k_state']))
        p=_vr_probs(q,k,qh,kh,d)
        base=_rs_parent_v(*s['v'],kh,d,parent['v_state'])
        f,rms,_,b=_rs_features(v,base,kh,d)
        target=_joint_output(qraw,kraw,v,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d).transpose(0,1)
        residual=(target-p@b.repeat_interleave(qh//kh,0))/rms.repeat_interleave(qh//kh,0)[:,None]
        a=torch.einsum('hqt,htcf->hqcf',p,f.repeat_interleave(qh//kh,0))
        a=a.reshape(kh,qh//kh,ix.numel(),d,8).permute(0,3,1,2,4).reshape(kh,d,-1,8)
        y=residual.reshape(kh,qh//kh,ix.numel(),d).permute(0,3,1,2).reshape(kh,d,-1,1)
        aa+=(a.transpose(-1,-2)@a)/a.shape[2]
        ab+=(a.transpose(-1,-2)@y)/a.shape[2]
    ridge=aa.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-10)*.05
    beta=torch.linalg.solve(aa+ridge[:,:,None,None]*torch.eye(8),ab).squeeze(-1)
    return dict(beta=beta.contiguous())


def _rs_guard(samples,parent,qh,kh,d):
    RS_DEBUG.clear();RS_DEBUG.update(error=False,adopted=0)
    if len(samples)<3 or d%64 or not parent['q_state'].get('gq'):return parent
    hold=samples[-1];t=hold['v'][0].shape[0]
    if not 4<=t<=4096 or hold['v'][0].numel()>4194304:return parent
    st=_rs_fit(samples,parent,qh,kh,d)
    if st is None:return parent
    ix=torch.linspace(0,t-1,min(_RS_GUARD_QUERIES,t)).long()
    qr=dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()
    kr=dequantize_nvfp4(*hold['k']).float();v=dequantize_nvfp4(*hold['v']).float()
    target=_joint_output(qr,kr,v,qh,kh,d,ix,query_subset=True)
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    base=_rs_parent_v(*hold['v'],kh,d,parent['v_state'])
    def loss(p):
        e=(_joint_output(q,k,_deq_params(p),qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    old=loss(base);best=old.clone();gains=torch.zeros(kh);ratios=[]
    for gain in (.5,1.,2.):
        trial=_rs_quant(v,base,st,kh,d,torch.full((kh,),gain))
        cost=loss(trial)
        take=(cost<old*.995).all(0)&(cost.mean(0)<best.mean(0))&torch.isfinite(cost).all(0)
        best=torch.where(take[None],cost,best);gains=torch.where(take,gain,gains)
        ratios.append((cost.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    RS_DEBUG.update(adopted=int((gains!=0).sum()),gains=gains.tolist(),ratios=ratios)
    if not bool((gains!=0).any()):return parent
    st.update(gain=gains,parent=parent['v_state'])
    return dict(parent,v_state=dict(residual_student=st))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_rs_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _rs_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        RS_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    if not isinstance(v_state,dict) or 'residual_student' not in v_state:
        return _rs_parent_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)
    st=v_state['residual_student'];base=_rs_parent_v(v_quant,v_scale,kv_num_heads,head_dim,st['parent'])
    if not 4<=v_quant.shape[0]<=4096 or v_quant.numel()>4194304:return base
    return _rs_quant(dequantize_nvfp4(v_quant,v_scale).float(),base,st,kv_num_heads,head_dim,st['gain'])
