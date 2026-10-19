"""Research-only full-scale-tree Value co-optimization, embedded in a parent module."""
_VM_ROUNDS=3
_VM_VARIANCE=1.0
_VM_CAP=1.01
VM_DEBUG={}
_vm_parent_calibration=hif4_calibration_attention
_vm_parent_dynamic=hif4_dynamic_quantize_v


def _vm_proposal(x,parent,variance):
    tokens,channels=x.shape
    if tokens<=1 or tokens>4096:
        return parent
    def costs(value):
        error=x-value
        mse=error.square().mean(0)
        objective=error.mean(0).square()+variance/tokens*mse
        return objective.reshape(-1,64).sum(1),mse,error.mean(0)
    best={k:v.clone() for k,v in parent.items()}
    best_cost,reference_mse,shift=costs(_deq_params(best))
    reference_mse=reference_mse.clamp_min(1e-30)
    for _ in range(_VM_ROUNDS):
        candidate=_dyn_table(x+shift,None,False)
        cost,mse,mean=costs(_deq_params(candidate))
        accept=(cost<best_cost)&(mse<=reference_mse*_VM_CAP+1e-20).reshape(-1,64).all(1)
        for key in best:
            mask=accept.reshape(1,-1,*([1]*(best[key].ndim-2)))
            best[key]=torch.where(mask,candidate[key],best[key])
        best_cost=torch.where(accept,cost,best_cost)
        shift=shift+mean*.75
    return best


def hif4_calibration_attention(samples,qh,kh,d):
    states=_vm_parent_calibration(samples,qh,kh,d)
    VM_DEBUG.clear();VM_DEBUG.update(accepted=0,error=False)
    if len(samples)<2 or d%64 or not states['q_state'].get('gq'):
        return states
    hold=samples[-1];rows=hold['v'][0].shape[0]
    if not 4<=rows<=4096:return states
    try:
        raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
        parent=_vm_parent_dynamic(*hold['v'],kh,d,states['v_state'])
        proposal=_vm_proposal(raw[2],parent,_VM_VARIANCE)
        ix=torch.linspace(0,rows-1,min(64,rows)).long()
        target=_joint_output(*raw,qh,kh,d,ix)
        q=_joint_quant_queries(hold,states['q_state'],qh,kh,d,ix)
        k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,states['k_state']))
        costs=[]
        for params in (parent,proposal):
            error=(_joint_output(q,k,_deq_params(params),qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
            costs.append(torch.stack((error[::2].mean(0),error[1::2].mean(0))))
        mask=(costs[1]<costs[0]*.995).all(0)
        VM_DEBUG.update(accepted=int(mask.sum()),ratios=(costs[1].mean(0)/costs[0].mean(0).clamp_min(1e-30)).tolist())
        if bool(mask.any()):
            return dict(states,v_state=dict(vm_active=mask,vm_variance=float(_VM_VARIANCE),fallback=states['v_state']))
    except Exception as exc:
        VM_DEBUG.update(error=True,error_type=type(exc).__name__)
    return states


def hif4_dynamic_quantize_v(q,s,kh,d,state):
    if not isinstance(state,dict) or 'vm_active' not in state:
        return _vm_parent_dynamic(q,s,kh,d,state)
    parent=_vm_parent_dynamic(q,s,kh,d,state['fallback'])
    if not bool(state['vm_active'].any()):return parent
    x=dequantize_nvfp4(q,s).float()
    proposal=_vm_proposal(x,parent,state['vm_variance'])
    active=state['vm_active'].repeat_interleave(d//64)
    return {k:torch.where(active.reshape(1,-1,*([1]*(parent[k].ndim-2))),proposal[k],parent[k]) for k in parent}
