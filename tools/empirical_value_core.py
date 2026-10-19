"""Expected-query Value optimization when calibration verifies exact K=V.

No current Q/K is accessed by the dynamic V interface. The state contains only
calibration query samples; current K is inferred as V only on validated groups.
"""
_EMPIRICAL_QUERIES=16
EMPIRICAL_DEBUG={}
_empirical_parent_probabilities=_vk_probabilities
_empirical_parent_calibration=hif4_calibration_attention


def _vk_probabilities(x,state,kh):
    if 'empirical_queries' not in state:
        return _empirical_parent_probabilities(x,state,kh)
    d=state['empirical_queries'].shape[-1]
    q=state['empirical_queries'].float()
    k=x.reshape(-1,kh,d).transpose(0,1).repeat_interleave(q.shape[0]//kh,0)
    p=torch.softmax(q@k.transpose(1,2)/d**.5,-1)
    return p.reshape(kh,-1,x.shape[0]).contiguous()


def _empirical_fit(samples,qh,kh,d):
    queries=[]
    for sample in samples[:-1]:
        k=dequantize_nvfp4(*sample['k'])
        v=dequantize_nvfp4(*sample['v'])
        if not torch.equal(k,v):
            return None
        q=dequantize_nvfp4(*sample['q'])
        ix=torch.linspace(0,q.shape[0]-1,min(q.shape[0],512)).long()
        queries.append(q[ix].reshape(-1,qh,d))
    if not queries:
        return None
    q=torch.cat(queries)
    ix=torch.linspace(0,q.shape[0]-1,min(q.shape[0],_EMPIRICAL_QUERIES)).long()
    return dict(empirical_queries=q[ix].transpose(0,1).contiguous(),
                active=torch.ones(kh,dtype=torch.bool),heads=qh)


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    states=_empirical_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    EMPIRICAL_DEBUG.clear();EMPIRICAL_DEBUG.update(identity=False,accepted=0,error=False)
    qh,kh,d=q_num_heads,kv_num_heads,head_dim
    if len(calib_qkv_list)<2 or d%64 or not states['q_state'].get('gq'):
        return states
    hold=calib_qkv_list[-1]
    rows=hold['v'][0].shape[0]
    if not 4<=rows<=4096:
        return states
    try:
        empirical=_empirical_fit(calib_qkv_list,qh,kh,d)
        if empirical is None or not torch.equal(dequantize_nvfp4(*hold['k']),dequantize_nvfp4(*hold['v'])):
            return states
        EMPIRICAL_DEBUG['identity']=True
        raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
        base=hif4_dynamic_quantize_v(*hold['v'],kh,d,states['v_state'])
        plain=_vk_parent_v(*hold['v'],None,kh,d)
        proposal=_vk_refine(raw[2],plain,empirical,kh,d)
        ix=torch.linspace(0,rows-1,min(rows,64)).long()
        target=_joint_output(*raw,qh,kh,d,ix)
        q=_joint_quant_queries(hold,states['q_state'],qh,kh,d,ix)
        k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,states['k_state']))
        costs=[]
        for value in (base,proposal):
            err=(_joint_output(q,k,_deq_params(value),qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
            costs.append(torch.stack((err[::2].mean(0),err[1::2].mean(0))))
        accept=(costs[1]<costs[0]*.99).all(0)
        EMPIRICAL_DEBUG.update(accepted=int(accept.sum()),hold_ratios=(costs[1].mean(0)/costs[0].mean(0).clamp_min(1e-30)).tolist())
        if not bool(accept.any()):
            return states
        empirical['active']=accept
        empirical['fallback']=states['v_state']
        return dict(states,v_state=empirical)
    except Exception as exc:
        EMPIRICAL_DEBUG.update(error=True,error_type=type(exc).__name__)
        return states


_empirical_parent_v=hif4_dynamic_quantize_v


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    if not isinstance(v_state,dict) or 'empirical_queries' not in v_state:
        return _empirical_parent_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)
    parent=_empirical_parent_v(v_quant,v_scale,kv_num_heads,head_dim,v_state.get('fallback'))
    x=dequantize_nvfp4(v_quant,v_scale).float()
    plain=_vk_parent_v(v_quant,v_scale,None,kv_num_heads,head_dim)
    candidate=_vk_refine(x,plain,v_state,kv_num_heads,head_dim)
    mask=v_state['active'].repeat_interleave(head_dim//64)
    result={}
    for key in parent:
        select=mask.reshape(1,-1,*([1]*(parent[key].ndim-2)))
        result[key]=torch.where(select,candidate[key],parent[key])
    return result
