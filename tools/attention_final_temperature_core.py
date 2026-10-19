"""Recalibrate Query temperature against the final deployed Key/Value paths."""
TG_DEBUG={}
_TG_MODE=1
_tg_parent_calibration=hif4_calibration_attention


def _tg_guard(samples,parent,qh,kh,d):
    TG_DEBUG.clear();TG_DEBUG.update(accepted=0,error=False)
    if len(samples)<3 or d%64 or not parent['q_state'].get('gq'):return parent
    hold=samples[-1];t=hold['q'][0].shape[0]
    if not 4<=t<=4096:return parent
    delta,_=_temp_fit(samples,parent,qh,kh,d)
    oldgain=parent['q_state'].get('temperature_gain',torch.ones(qh)).float()
    factors=(.5,1.) if _TG_MODE==1 else (1.,2.,4.)
    gains=[oldgain]+[oldgain*(1.+s*delta).clamp(.25,2.) for s in factors]
    ix=torch.linspace(0,t-1,min(t,64)).long()
    raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
    raw.extend(dequantize_nvfp4(*hold[r]).float() for r in 'kv')
    target=_joint_output(*raw,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d)
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
    costs=[]
    for gain in gains:
        state=dict(parent['q_state'],temperature_gain=gain)
        q=_joint_quant_queries(hold,state,qh,kh,d,ix)
        error=(_joint_output(q,k,v,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d)-target).square().mean(-1)
        costs.append(torch.stack((error[::2].mean(0),error[1::2].mean(0))))
    if not bool(torch.isfinite(torch.stack(costs)).all()):return parent
    best=costs[0];chosen=oldgain.clone()
    for gain,cost in zip(gains[1:],costs[1:]):
        take=(cost<costs[0]*.995).all(0)&(cost.mean(0)<best.mean(0))
        chosen=torch.where(take,gain,chosen);best=torch.where(take[None],cost,best)
    TG_DEBUG.update(accepted=int((chosen!=oldgain).sum()),gains=chosen.tolist(),ratios=(best.mean(0)/costs[0].mean(0).clamp_min(1e-30)).tolist())
    if torch.equal(chosen,oldgain):return parent
    return dict(parent,q_state=dict(parent['q_state'],temperature_gain=chosen.contiguous()))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_tg_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _tg_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        TG_DEBUG.update(error=True,error_type=type(exc).__name__);return parent
