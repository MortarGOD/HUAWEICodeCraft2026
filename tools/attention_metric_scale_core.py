"""Joint metadata/code proposal scored by complete channel Gram, Q only."""
HS_DEBUG={}
_hs_parent_calibration=hif4_calibration_attention
_hs_parent_q=hif4_dynamic_quantize_q


def _hs_trial(raw,st,qh,d,parent,total_rows):
    x=_kt_input(raw,st,qh,d,'q')
    h=_joint_raw_metric(st).float()
    h=(h+h.transpose(-1,-2))*.5
    hf=h.repeat_interleave(qh//h.shape[0],0)
    importance=hf.diagonal(dim1=-2,dim2=-1).reshape(1,-1).clamp_min(1e-20)
    candidate=_quantize_weighted(x,importance)
    unit=_params_unit_flat(candidate)
    u=st['u'].float().repeat_interleave(qh//st['u'].shape[0],0)
    values=_gptq_quantize_batched(x.reshape(-1,qh,d).transpose(0,1).contiguous(),
        unit.reshape(-1,qh,d).transpose(0,1).contiguous(),u).transpose(0,1).reshape(x.shape).contiguous()
    rf=int(st.get('rf') or 0)
    if rf and total_rows<=ATTN_REFINE_MAX_T and 'H' in st:
        rounds=rf*(32 if total_rows<=128 else (4 if total_rows>1024 else _REF_ATTN_ROUNDS))
        values=_refine_attn_heads(x,values,unit,st['H'],qh,rf,total_rounds=rounds)
    candidate=_values_to_params(values,candidate)
    def cost(params):
        e=(_deq_params(params)-x).reshape(-1,qh,d).transpose(0,1)
        return ((e@hf)*e).sum(-1).transpose(0,1)
    take=cost(candidate)<cost(parent)
    if 'hs_active' in st:
        take=take & st['hs_active'].repeat_interleave(qh//st['hs_active'].numel())[None]
    mask=take.repeat_interleave(d//64,-1)
    return {key:torch.where(mask.reshape(*mask.shape,*([1]*(value.ndim-2))),value,parent[key])
        for key,value in candidate.items()}


@torch.inference_mode()
def hif4_calibration_attention(samples,q_num_heads,kv_num_heads,head_dim):
    parent=_hs_parent_calibration(samples,q_num_heads,kv_num_heads,head_dim)
    qh,kh,d=q_num_heads,kv_num_heads,head_dim
    st=parent['q_state'];HS_DEBUG.clear();HS_DEBUG.update(accepted=0,error=False)
    if not st.get('gq') or 'u' not in st or d%64:return parent
    hold=samples[-1];n=hold['q'][0].shape[0]
    if not 4<=n<=4096:return parent
    ix=torch.linspace(0,n-1,min(n,32)).long()
    raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
    raw += [dequantize_nvfp4(*hold[r]).float() for r in 'kv']
    # Fixed-grid query helper preserves the parent's full-length refine budget.
    original=_joint_quant_queries(hold,st,qh,kh,d,ix)
    source=_kt_input(raw[0],st,qh,d,'q')
    pp=_values_to_params(original,_dyn_table(source,None,has_scale=False))
    trial=_deq_params(_hs_trial(raw[0],st,qh,d,pp,n))
    key=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    val=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
    target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
    def cost(query):
        e=(_joint_output(query,key,val,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    old,new=cost(original),cost(trial)
    take=(new<old*.995).all(0)&torch.isfinite(new).all(0)
    HS_DEBUG.update(accepted=int(take.sum()),ratios=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):return parent
    return dict(parent,q_state=dict(st,hs_active=take))


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    parent=_hs_parent_q(q_quant,q_scale,q_num_heads,head_dim,q_state)
    if not isinstance(q_state,dict) or 'hs_active' not in q_state or not 4<=q_quant.shape[0]<=4096:
        return parent
    return _hs_trial(dequantize_nvfp4(q_quant,q_scale).float(),q_state,q_num_heads,head_dim,parent,q_quant.shape[0])
