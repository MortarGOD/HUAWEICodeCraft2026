"""Calibration query prior and full V->K map, only as an error metric.

No predicted PV teacher: the residual being quantized is always current V-Vq.
"""
_CQ_QUERIES=32
_CQ_RIDGE=.01
CQ_DEBUG={}
_cq_parent_calibration=hif4_calibration_attention
_cq_parent_dynamic=hif4_dynamic_quantize_v
_cq_parent_probabilities=_vk_probabilities


def _vk_probabilities(x,state,kh):
    if 'conditional_queries' not in state:
        return _cq_parent_probabilities(x,state,kh)
    return query_probabilities(x,state['conditional_queries'])


def _cq_fit(samples,qh,kh,d):
    fit_samples=[];queries=[]
    for sample in samples[:-1]:
        raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
        fit_samples.append(raw)
        ix=torch.linspace(0,raw['q'].shape[0]-1,min(512,raw['q'].shape[0])).long()
        queries.append(raw['q'][ix].reshape(-1,qh,d))
    predictor=fit_key_predictor(fit_samples,_CQ_RIDGE)
    q=torch.cat(queries)
    ix=torch.linspace(0,q.shape[0]-1,min(q.shape[0],_CQ_QUERIES)).long()
    fused=fuse_queries(predictor,q[ix].transpose(0,1).contiguous(),kh,d)
    return dict(conditional_queries=fused,active=torch.ones(kh,dtype=torch.bool)),predictor


def _cq_value_proposal(x,parent,plain,state,kh,d):
    return _vk_refine(x,plain,state,kh,d)


def hif4_calibration_attention(samples,qh,kh,d):
    states=_cq_parent_calibration(samples,qh,kh,d)
    CQ_DEBUG.clear();CQ_DEBUG.update(accepted=0,error=False)
    if len(samples)<2 or d%64 or not states['q_state'].get('gq'):return states
    hold=samples[-1];rows=hold['v'][0].shape[0]
    if not 4<=rows<=4096:return states
    try:
        candidate_state,predictor=_cq_fit(samples,qh,kh,d)
        raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
        predicted=predict_keys(raw[2],predictor)
        key_nmse=float((predicted-raw[1]).square().mean()/raw[1].square().mean().clamp_min(1e-30))
        parent=_cq_parent_dynamic(*hold['v'],kh,d,states['v_state'])
        plain=_vk_parent_v(*hold['v'],None,kh,d)
        proposal=_cq_value_proposal(raw[2],parent,plain,candidate_state,kh,d)
        ix=torch.linspace(0,rows-1,min(64,rows)).long()
        target=_joint_output(*raw,qh,kh,d,ix)
        q=_joint_quant_queries(hold,states['q_state'],qh,kh,d,ix)
        k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,states['k_state']))
        costs=[]
        for p in (parent,proposal):
            err=(_joint_output(q,k,_deq_params(p),qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
            costs.append(torch.stack((err[::2].mean(0),err[1::2].mean(0))))
        active=(costs[1]<costs[0]*.99).all(0)
        CQ_DEBUG.update(accepted=int(active.sum()),identity=predictor['identity'],key_nmse=key_nmse,
                        ratios=(costs[1].mean(0)/costs[0].mean(0).clamp_min(1e-30)).tolist())
        if bool(active.any()):
            candidate_state.update(active=active,fallback=states['v_state'])
            return dict(states,v_state=candidate_state)
    except Exception as exc:
        CQ_DEBUG.update(error=True,error_type=type(exc).__name__)
    return states


def hif4_dynamic_quantize_v(q,s,kh,d,state):
    if not isinstance(state,dict) or 'conditional_queries' not in state:
        return _cq_parent_dynamic(q,s,kh,d,state)
    parent=_cq_parent_dynamic(q,s,kh,d,state.get('fallback'))
    if not bool(state['active'].any()):return parent
    x=dequantize_nvfp4(q,s).float()
    plain=_vk_parent_v(q,s,None,kh,d)
    proposal=_cq_value_proposal(x,parent,plain,state,kh,d)
    mask=state['active'].repeat_interleave(d//64)
    return {k:torch.where(mask.reshape(1,-1,*([1]*(parent[k].ndim-2))),proposal[k],parent[k]) for k in parent}
