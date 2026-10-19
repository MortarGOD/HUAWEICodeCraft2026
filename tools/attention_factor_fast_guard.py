"""Equivalent factor-only calibration guard, omitting unused moment diagnostics."""
def _am_guard(samples,parent,qh,kh,d,model=None):
    AM_DEBUG.clear();AM_DEBUG.update(error=False,adopted=0)
    if len(samples)<3 or d%64 or not parent['q_state'].get('gq'):return parent
    hold=samples[-1];t=hold['v'][0].shape[0]
    if not 4<=t<=4096 or hold['v'][0].numel()>4194304:return parent
    st=_am_fit(samples,parent,qh,kh,d) if model is None else dict(model)
    if st is None:return parent
    st=dict(st,uncertainty=_AM_UNCERTAINTY,gradient_gain=_AM_GRADIENT_GAIN)
    ix=torch.linspace(0,t-1,min(32,t)).long()
    rawq=dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()
    rawk=dequantize_nvfp4(*hold['k']).float();v=dequantize_nvfp4(*hold['v']).float()
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    p=_vr_probs(q,k,qh,kh,d)
    truth=_vr_probs(rawq,rawk,qh,kh,d)
    target=truth@v.reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    base=_am_parent_v(*hold['v'],kh,d,parent['v_state'])
    trial=_am_quant(v,base,st,kh,d)
    def loss(params):
        values=_deq_params(params).reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        e=(p@values-target).square().mean(-1).reshape(kh,qh//kh,-1).mean(1)
        return torch.stack((e[:,::2].mean(1),e[:,1::2].mean(1)))
    old=loss(base);new=loss(trial);take=(new<old*.995).all(0)&torch.isfinite(new).all(0)
    AM_DEBUG.update(adopted=int(take.sum()),hold_ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):return parent
    st.update(active=take,parent=parent['v_state'])
    return dict(parent,v_state=dict(moment=st))
