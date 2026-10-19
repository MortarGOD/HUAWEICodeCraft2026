"""Calibration-selected GPTQ elimination order, original HiF4 layout retained."""
_ORDER_MODE='conditional'
ORDER_DEBUG={}


def _order_quant(x,unit,u,state):
    if 'gptq_order' not in state:
        return _gptq_quantize_batched(x,unit,u)
    order=state['gptq_order']
    order=order.repeat_interleave(x.shape[0]//order.shape[0],0)
    index=order[:,None,:].expand_as(x)
    ordered_u=state['gptq_order_u']
    ordered_u=ordered_u.repeat_interleave(x.shape[0]//ordered_u.shape[0],0)
    values=_gptq_quantize_batched(x.gather(2,index),unit.gather(2,index),ordered_u)
    return torch.empty_like(values).scatter_(2,index,values)


def _order_proposal(state):
    u=state['u'].float()
    inverse=u.transpose(-1,-2)@u
    if _ORDER_MODE=='diagonal':
        if 'H' in state:
            score=state['H'].float().diagonal(dim1=-2,dim2=-1)
        else:
            iu=torch.linalg.solve_triangular(u,torch.eye(u.shape[-1]).expand_as(u),upper=True)
            score=(iu@iu.transpose(-1,-2)).diagonal(dim1=-2,dim2=-1)
        order=score.argsort(dim=-1,descending=True,stable=True)
    else:
        # High conditional curvature first: 1 / diagonal(H^-1).
        order=inverse.diagonal(dim1=-2,dim2=-1).argsort(dim=-1,descending=_ORDER_MODE=='reverse',stable=True)
    ordered=inverse.gather(1,order[:,:,None].expand_as(inverse)).gather(2,order[:,None,:].expand_as(inverse))
    ordered_u=torch.linalg.cholesky(ordered,upper=True)
    return dict(state,gptq_order=order.contiguous(),gptq_order_u=ordered_u.contiguous())


def _order_guard(hold,states,proposed,qh,kh,d):
    rows=hold['q'][0].shape[0]
    ix=torch.linspace(0,rows-1,min(rows,64)).long()
    raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    reference=_joint_output(*raw,qh,kh,d,ix).reshape(-1,qh,d)
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,states['v_state']))
    qq=[_joint_quant_queries(hold,s['q_state'],qh,kh,d,ix) for s in (states,proposed)]
    kk=[_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,s['k_state'])) for s in (states,proposed)]
    cost=torch.empty(2,2,2,qh)
    for qi in range(2):
        for ki in range(2):
            out=_joint_output(qq[qi],kk[ki],v,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d)
            err=(out-reference).square().mean(-1)
            cost[qi,ki]=torch.stack((err[::2].mean(0),err[1::2].mean(0)))
    if not bool(torch.isfinite(cost).all()):
        ORDER_DEBUG['nonfinite_guard']=True
        return states
    qm,km=torch.zeros(qh,dtype=torch.bool),torch.zeros(kh,dtype=torch.bool)
    for hv in range(kh):
        sl=slice(hv*(qh//kh),(hv+1)*(qh//kh))
        q0=(cost[1,0,:,sl]<cost[0,0,:,sl]*.995).all(0)
        best=torch.where(q0[None],cost[1,0,:,sl],cost[0,0,:,sl])
        q1=(cost[1,1,:,sl]<cost[0,1,:,sl]*.995).all(0)
        other=torch.where(q1[None],cost[1,1,:,sl],cost[0,1,:,sl])
        take=bool((other.sum(-1)<best.sum(-1)*.995).all() and (other<=best*1.01).all())
        qm[sl],km[hv]=q1 if take else q0,take
    result=dict(states)
    for role,mask in (('q',qm),('k',km)):
        if not bool(mask.any()):
            continue
        old,new=states[role+'_state'],proposed[role+'_state']
        order=new['gptq_order'].repeat_interleave(mask.numel()//new['gptq_order'].shape[0],0)
        nu=new['gptq_order_u'].repeat_interleave(mask.numel()//new['gptq_order_u'].shape[0],0)
        ou=old['u'].repeat_interleave(mask.numel()//old['u'].shape[0],0)
        result[role+'_state']=dict(old,
            gptq_order=torch.where(mask[:,None],order,torch.arange(d)[None]).contiguous(),
            gptq_order_u=torch.where(mask[:,None,None],nu,ou).contiguous())
    ORDER_DEBUG.update(q_heads=int(qm.sum()),k_heads=int(km.sum()),
                       hold_ratios=(cost.mean((2,3))/cost[0,0].mean().clamp_min(1e-30)).tolist())
    return result


_order_parent_calibration=hif4_calibration_attention


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    states=_order_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    ORDER_DEBUG.clear();ORDER_DEBUG.update(q_heads=0,k_heads=0,error=False)
    if (not states['q_state'].get('gq') or not states['k_state'].get('gq')
            or not 4<=calib_qkv_list[-1]['q'][0].shape[0]<=4096):
        return states
    try:
        proposed=dict(states,q_state=_order_proposal(states['q_state']),k_state=_order_proposal(states['k_state']))
        return _order_guard(calib_qkv_list[-1],states,proposed,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        ORDER_DEBUG.update(error=True,error_type=type(exc).__name__)
        return states
