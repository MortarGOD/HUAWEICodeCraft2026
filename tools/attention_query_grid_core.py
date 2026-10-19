"""Native Query scale-tree refinement with full per-head channel curvature."""
QG_DEBUG={}
_QG_MODE=4
_qg_parent_calibration=hif4_calibration_attention
_qg_parent_q=hif4_dynamic_quantize_q


def _qg_quant(raw,st,qh,kh,d,total_rows):
    x=_kt_input(raw,st,qh,d,'q')
    return _kg_grid(x,st,qh,d,st['qg_mode'],total_rows)


def _qg_guard(samples,parent,qh,kh,d):
    QG_DEBUG.clear();QG_DEBUG.update(accepted=0,error=False)
    st=parent['q_state'];hold=samples[-1];t=hold['q'][0].shape[0]
    if len(samples)<3 or d%64 or not 4<=t<=4096 or not st.get('gq') or 'u' not in st:return parent
    u=st['u'].float()
    inv=torch.linalg.solve_triangular(u,torch.eye(d).expand_as(u),upper=True)
    h=inv@inv.transpose(-1,-2)
    h=h/h.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None].clamp_min(1e-30)
    proposed=dict(st,kg_h=h.repeat_interleave(qh//h.shape[0],0).contiguous(),qg_mode=_QG_MODE)
    ix=torch.linspace(0,t-1,min(t,64)).long()
    raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
    raw.extend(dequantize_nvfp4(*hold[r]).float() for r in 'kv')
    target=_joint_output(*raw,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d)
    q=_joint_quant_queries(hold,st,qh,kh,d,ix)
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
    def cost(query):
        e=(_joint_output(query,k,v,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d)-target).square().mean(-1)
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    trial=_deq_params(_qg_quant(raw[0],proposed,qh,kh,d,t))
    old=cost(q);new=cost(trial)
    take=(new<old*.995).all(0)
    QG_DEBUG.update(accepted=int(take.sum()),ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):return parent
    return dict(parent,q_state=dict(proposed,qg_active=take,qg_fallback=st))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_qg_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _qg_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        QG_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    if not isinstance(q_state,dict) or 'qg_mode' not in q_state or not 4<=q_quant.shape[0]<=4096:
        fallback=q_state.get('qg_fallback',q_state) if isinstance(q_state,dict) else q_state
        return _qg_parent_q(q_quant,q_scale,q_num_heads,head_dim,fallback)
    raw=dequantize_nvfp4(q_quant,q_scale).float()
    new=_qg_quant(raw,q_state,q_num_heads,q_state['kvh'],head_dim,q_quant.shape[0])
    if bool(q_state['qg_active'].all()):return new
    old=_qg_parent_q(q_quant,q_scale,q_num_heads,head_dim,q_state['qg_fallback'])
    mask=q_state['qg_active'].repeat_interleave(head_dim//64)
    return {name:torch.where(mask.reshape(1,-1,*([1]*(a.ndim-2))),a,old[name]) for name,a in new.items()}
