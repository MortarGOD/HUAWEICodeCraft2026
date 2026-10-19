"""Learn final-output risk of complete native Value blocks during calibration."""
import math
RK_DEBUG={}
_RK_CONTEXT=True
_RK_RIDGE=.1
_RK_ORACLE_DIAGNOSTIC=False
_RK_FIT_QUERIES=16
_RK_GUARD_QUERIES=32
_rk_parent_calibration=hif4_calibration_attention
_rk_parent_v=hif4_dynamic_quantize_v


def _rk_candidates(v,parent):
    step=_params_unit_flat(parent)*.25
    nearest=torch.round(v/step).clamp(-7,7)*step
    yield _values_to_params(nearest,parent)
    bias=(v-_deq_params(parent)).mean(0)
    yield _values_to_params(torch.round((v+bias)/step).clamp(-7,7)*step,parent)
    for factor in (.7,1.,1.3):
        p=_dyn_table(v*factor,None,has_scale=False)
        step=_params_unit_flat(p)*.25
        yield _values_to_params(torch.round(v/step).clamp(-7,7)*step,p)


def _rk_features(v,value,kh,d,context):
    t=v.shape[0];x=v.reshape(t,kh,d//64,64);e=(value-v).reshape_as(x)
    energy=x.square().mean((0,3)).clamp_min(1e-12)
    row=x.square().mean(3)/energy[None]
    weight=row.clamp_max(16.);weight=weight/weight.mean(0).clamp_min(1e-12)[None]
    mean=e.mean(0);xm=x.mean(0)
    f=torch.stack((e.square().mean((0,3)),mean.square().mean(2),
        (e*x).mean((0,3)),(mean*xm).mean(2),
        (e.square().mean(3)*weight).mean(0),
        (e*weight[:,:,:,None]).mean(0).square().mean(2),
        (e.abs().mean(0)).square().mean(2),
        (e*x.sign()).mean((0,3))*energy.sqrt()),-1)/energy[:,:,None]
    if context:
        # A bounded observed-length interaction, not a guessed online length distribution.
        interaction=128./max(1,t) if context=='inverse' else min(12.,math.log2(max(2,t)))/8.-1.
        f=torch.cat((f,f*interaction),2)
    return f


def _rk_choose(v,parent,st,kh,d):
    reference=_rk_features(v,_deq_params(parent),kh,d,st['context'])
    best=torch.zeros(kh,d//64);out={k:x.clone() for k,x in parent.items()}
    active=st.get('active',torch.ones(kh,dtype=torch.bool))
    choices=torch.zeros(kh,d//64,dtype=torch.int32)
    for index,p in enumerate(_rk_candidates(v,parent),1):
        f=(_rk_features(v,_deq_params(p),kh,d,st['context'])-reference)/st['scale'][:,None]
        predicted=(f*st['beta'][:,None]).sum(-1)
        take=(predicted<best)&active[:,None]
        best=torch.where(take,predicted,best);choices=torch.where(take,index,choices)
        flat=take.flatten()
        for k,x in out.items():out[k]=torch.where(flat.reshape(1,-1,*([1]*(x.ndim-2))),p[k],x)
    return out,choices


def _rk_fit(samples,parent,qh,kh,d):
    features=[];labels=[];sizes=[]
    eligible=[s for s in samples[:-1] if 4<=s['v'][0].shape[0]<=2048 and s['v'][0].numel()<=4194304][-3:]
    if len(eligible)==3:eligible=[eligible[0],eligible[-1]]
    if len(eligible)<2:return None
    for s in eligible:
        t=s['v'][0].shape[0];ix=torch.linspace(0,t-1,min(_RK_FIT_QUERIES,t)).long()
        rawq=dequantize_nvfp4(s['q'][0][ix],s['q'][1][ix]).float()
        rawk=dequantize_nvfp4(*s['k']).float();v=dequantize_nvfp4(*s['v']).float()
        base=_rk_parent_v(*s['v'],kh,d,parent['v_state']);b=_deq_params(base)
        target=_joint_output(rawq,rawk,v,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d).transpose(0,1)
        q=_joint_quant_queries(s,parent['q_state'],qh,kh,d,ix)
        k=_deq_params(hif4_dynamic_quantize_k(*s['k'],kh,d,parent['k_state']))
        prob=_vr_probs(q,k,qh,kh,d)
        bh=b.reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        residual=prob@bh-target
        initial=_rk_features(v,b,kh,d,_RK_CONTEXT)
        energy=v.reshape(t,kh,d//64,64).square().mean((0,3)).clamp_min(1e-12)
        xf=[];yf=[]
        for p in _rk_candidates(v,base):
            value=_deq_params(p)
            delta=(value-b).reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
            change=prob@delta
            loss=(2*residual*change+change.square()).reshape(kh,qh//kh,ix.numel(),d//64,64).mean((1,2,4))/energy
            xf.append(_rk_features(v,value,kh,d,_RK_CONTEXT)-initial);yf.append(loss)
        features.append(torch.stack(xf,1).reshape(kh,-1,initial.shape[-1]))
        labels.append(torch.stack(yf,1).reshape(kh,-1));sizes.append(t)
    x=torch.cat(features,1);y=torch.cat(labels,1)
    scale=x.square().mean(1).sqrt().clamp_min(1e-8);z=x/scale[:,None]
    h=z.transpose(1,2)@z/z.shape[1]+torch.eye(z.shape[2])*_RK_RIDGE
    beta=torch.linalg.solve(h,(z.transpose(1,2)@y[:,:,None])/z.shape[1]).squeeze(-1)
    RK_DEBUG.update(fit_sequences=len(eligible),fit_rows=sizes,training_mae=float(((z*beta[:,None]).sum(-1)-y).abs().mean()))
    return dict(beta=beta.contiguous(),scale=scale.contiguous(),context=_RK_CONTEXT)


def _rk_guard(samples,parent,qh,kh,d):
    RK_DEBUG.clear();RK_DEBUG.update(error=False,adopted=0)
    if len(samples)<3 or d%64 or not parent['q_state'].get('gq'):return parent
    hold=samples[-1];t=hold['v'][0].shape[0]
    if not 4<=t<=4096 or hold['v'][0].numel()>4194304:return parent
    st=_rk_fit(samples,parent,qh,kh,d)
    if st is None:return parent
    ix=torch.linspace(0,t-1,min(_RK_GUARD_QUERIES,t)).long()
    rawq=dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()
    rawk=dequantize_nvfp4(*hold['k']).float();v=dequantize_nvfp4(*hold['v']).float()
    target=_joint_output(rawq,rawk,v,qh,kh,d,ix,query_subset=True)
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    base=_rk_parent_v(*hold['v'],kh,d,parent['v_state'])
    trial,choices=_rk_choose(v,base,st,kh,d)
    def loss(p):
        e=(_joint_output(q,k,_deq_params(p),qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    old=loss(base);new=loss(trial)
    if _RK_ORACLE_DIAGNOSTIC:
        prob=_vr_probs(q,k,qh,kh,d)
        truth=target.reshape(-1,qh,d).transpose(0,1)
        def block_loss(p):
            value=_deq_params(p).reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
            return (prob@value-truth).square().reshape(kh,qh//kh,ix.numel(),d//64,64).mean((1,2,4))
        origin=block_loss(base);oracle=origin.clone()
        for p in _rk_candidates(v,base):oracle=torch.minimum(oracle,block_loss(p))
        RK_DEBUG.update(calibration_oracle_ratio=(oracle.mean(-1)/origin.mean(-1).clamp_min(1e-30)).tolist())
    take=(new<old*.995).all(0)&torch.isfinite(new).all(0)
    RK_DEBUG.update(adopted=int(take.sum()),choices=choices.tolist(),hold_ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):return parent
    st.update(active=take,parent=parent['v_state'])
    return dict(parent,v_state=dict(risk_rank=st))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_rk_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _rk_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        RK_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    if not isinstance(v_state,dict) or 'risk_rank' not in v_state:
        return _rk_parent_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)
    st=v_state['risk_rank'];base=_rk_parent_v(v_quant,v_scale,kv_num_heads,head_dim,st['parent'])
    if not 4<=v_quant.shape[0]<=4096 or v_quant.numel()>4194304:return base
    v=dequantize_nvfp4(v_quant,v_scale).float()
    return _rk_choose(v,base,st,kv_num_heads,head_dim)[0]
