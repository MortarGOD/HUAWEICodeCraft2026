"""Query GPTQ with calibrated native scale-tree reconstruction."""
QG_DEBUG={}
_QG_MODE=4
_QG_PASSES=2
_qg_parent_calibration=hif4_calibration_attention
_qg_parent_q=hif4_dynamic_quantize_q


def _qg_quant(raw,st,qh,kh,d,total_rows):
    x=_kt_input(raw,st,qh,d,'q')
    p=_dyn_table(x,None,has_scale=False)
    u=st['u'].float().repeat_interleave(qh//st['u'].shape[0],0)
    y=x.reshape(-1,qh,d)
    initial=_gptq_quantize_batched(y.transpose(0,1).contiguous(),
        _params_unit_flat(p).reshape_as(y).transpose(0,1).contiguous(),u)
    p=_values_to_params(initial.transpose(0,1).reshape(x.shape).contiguous(),p)
    active=st.get('qg_active',torch.ones(qh,dtype=torch.bool))
    blocks=active.repeat_interleave(d//64)
    selected={name:value[:,blocks].clone() for name,value in p.items()}
    nh=int(active.sum())
    reference=y[:,active]
    current=_deq_params(selected).reshape(-1,nh,d)
    h=st['kg_h'].float()[active]
    for sweep in range(_QG_PASSES):
        for block in range(d//64):
            a=block*64;b=a+64
            gradient=torch.einsum('thd,hde->the',current-reference,h)[:,:,a:b]
            old=current[:,:,a:b].clone();subh=h[:,a:b,a:b]
            hh=subh.clone()
            hh.diagonal(dim1=-2,dim2=-1).add_(.01*subh.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None])
            proposal=old-torch.linalg.solve(hh,gradient.transpose(0,1).transpose(1,2)).transpose(1,2).transpose(0,1)
            cols=torch.arange(nh)*(d//64)+block
            best_params={name:value[:,cols].clone() for name,value in selected.items()}
            best=torch.zeros_like(gradient[:,:,0])
            for sf in _vm_neighbor_scales(best_params['scale_factor']):
                trial=_vm_tree_at_sf(proposal.reshape(x.shape[0],-1),sf)
                value=_deq_params(trial).reshape(x.shape[0],nh,64)
                delta=value-old
                cost=2*(delta*gradient).sum(-1)+torch.einsum('thc,hce,the->th',delta,subh,delta)
                take=cost<best;best=torch.minimum(best,cost)
                for name,tensor in best_params.items():
                    best_params[name]=torch.where(take.reshape(*take.shape,*([1]*(tensor.ndim-2))),trial[name],tensor)
            current[:,:,a:b]=_deq_params(best_params).reshape(x.shape[0],nh,64)
            for name in selected:selected[name][:,cols]=best_params[name]
    for name in p:p[name][:,blocks]=selected[name]
    unit=_params_unit_flat(p);value=_deq_params(p)
    rf=int(st.get('rf') or 0)
    if rf and total_rows<=ATTN_REFINE_MAX_T and 'H' in st:
        rounds=rf*(32 if total_rows<=128 else (4 if total_rows>1024 else _REF_ATTN_ROUNDS))
        value=_refine_attn_heads(x,value,unit,st['H'],qh,rf,total_rounds=rounds)
    return _values_to_params(value,p)


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_qg_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _qg_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        QG_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    if not isinstance(q_state,dict) or 'qg_mode' not in q_state or not 4<=q_quant.shape[0]<=4096:
        fallback=q_state.get('qg_fallback',q_state) if isinstance(q_state,dict) else q_state
        return _qg_parent_q(q_quant,q_scale,q_num_heads,head_dim,fallback)
    if not bool(q_state['qg_active'].any()):
        return _qg_parent_q(q_quant,q_scale,q_num_heads,head_dim,q_state['qg_fallback'])
    raw=dequantize_nvfp4(q_quant,q_scale).float()
    return _qg_quant(raw,q_state,q_num_heads,q_state['kvh'],head_dim,q_quant.shape[0])
