"""Deploy-aware Hessian regularization selection, unchanged dynamic budget."""
_DAMP_CANDIDATES = (.01, .3)
_DAMP_MARGIN = .005
DAMP_DEBUG = {}
_damp_parent_calibration = hif4_calibration_attention


def _damp_candidate(state, ratio, d):
    if not isinstance(state,dict) or not state.get('gq') or 'H' not in state:
        return None
    h=state['H'].float()
    h=(h+h.transpose(-1,-2))*.5
    diag=h.diagonal(dim1=-2,dim2=-1).mean(-1)
    if not bool(torch.isfinite(h).all() and (diag>0).all()):
        return None
    try:
        chol,info=torch.linalg.cholesky_ex(h+ratio*diag[:,None,None]*torch.eye(d)[None])
        if bool(info.any()):return None
        inverse=torch.cholesky_solve(torch.eye(d).expand_as(chol),chol)
        u,info=torch.linalg.cholesky_ex(inverse,upper=True)
        if bool(info.any()) or not bool(torch.isfinite(u).all()):return None
        return dict(state,u=u.contiguous())
    except Exception:
        return None


def _damp_guard(samples,states,qh,kh,d):
    hold=samples[-1];rows=hold['q'][0].shape[0]
    if not 4<=rows<=4096:return states
    bank={role:[states[role+'_state']] for role in 'qk'}
    labels={role:['parent'] for role in 'qk'}
    for role in 'qk':
        for ratio in _DAMP_CANDIDATES:
            candidate=_damp_candidate(states[role+'_state'],ratio,d)
            if candidate is not None:
                bank[role].append(candidate);labels[role].append(ratio)
    if len(bank['q'])==len(bank['k'])==1:return states
    ix=torch.linspace(0,rows-1,min(rows,64)).long()
    raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    target=_joint_output(*raw,qh,kh,d,ix).reshape(-1,qh,d)
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,states['v_state']))
    qs=[(_joint_quant_queries(hold,st,qh,kh,d,ix) if st.get('gq') else
         _deq_params(hif4_dynamic_quantize_q(*hold['q'],qh,d,st))[ix]) for st in bank['q']]
    ks=[_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,st)) for st in bank['k']]
    costs=torch.empty(len(qs),len(ks),2,kh)
    for qi,q in enumerate(qs):
        for ki,k in enumerate(ks):
            out=_joint_output(q,k,v,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d)
            error=(out-target).square().reshape(-1,kh,qh//kh*d).mean(-1)
            costs[qi,ki]=torch.stack((error[::2].mean(0),error[1::2].mean(0)))
    choices={r:torch.zeros(kh,dtype=torch.int64) for r in 'qk'}
    best=costs[0,0].clone()
    for qi in range(len(qs)):
        for ki in range(len(ks)):
            if qi==ki==0:continue
            take=torch.isfinite(costs[qi,ki]).all(0)&(costs[qi,ki]<best*(1-_DAMP_MARGIN)).all(0)
            best=torch.where(take[None],costs[qi,ki],best)
            choices['q']=torch.where(take,qi,choices['q'])
            choices['k']=torch.where(take,ki,choices['k'])
    answer=dict(states)
    for role in 'qk':
        if bool(choices[role].any()):
            u=states[role+'_state']['u'].clone()
            for i,st in enumerate(bank[role][1:],1):
                u=torch.where((choices[role]==i)[:,None,None],st['u'],u)
            answer[role+'_state']=dict(states[role+'_state'],u=u.contiguous())
    DAMP_DEBUG.update(labels=labels,choices={r:c.tolist() for r,c in choices.items()},costs=costs.tolist())
    return answer


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    states=_damp_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    DAMP_DEBUG.clear();DAMP_DEBUG.update(error=False)
    if len(calib_qkv_list)<2 or head_dim%64:return states
    try:
        return _damp_guard(calib_qkv_list,states,q_num_heads,kv_num_heads,head_dim)
    except Exception:
        DAMP_DEBUG['error']=True
        return states
