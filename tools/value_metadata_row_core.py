"""Row-local scale-tree candidates with exact global surrogate acceptance.

Local delta for E=sum_rows(error):
T^2*dL=(1+variance)*delta^2-2*(E+variance*error_row)*delta.
Simultaneous row changes interact; exact loss is checked after each proposal.
"""
def _vm_proposal(x,parent,variance):
    tokens,channels=x.shape
    if tokens<=1 or tokens>4096:return parent
    blocks=channels//64
    original=_deq_params(parent)
    error=x-original
    reference_mse=error.square().mean(0).clamp_min(1e-30)
    initial_cost=(error.mean(0).square()+variance/tokens*reference_mse).reshape(blocks,64).sum(1)
    best_cost=initial_cost
    best={k:v.clone() for k,v in parent.items()}
    pool={k:v.clone() for k,v in parent.items()}
    local_cost=torch.zeros(tokens,blocks)
    residual_sum=error.sum(0)
    for specification in CAND_GRID:
        candidate=_vm_refine_new_tree(x,_quantize_weighted(x,torch.ones(1,channels),grid=(specification,)),variance)
        delta=_deq_params(candidate)-original
        local=((1+variance)*delta.square()-2*(residual_sum+variance*error)*delta).reshape(tokens,blocks,64).sum(2)
        take=local<local_cost
        for key in pool:
            mask=take.reshape(tokens,blocks,*([1]*(pool[key].ndim-2)))
            pool[key]=torch.where(mask,candidate[key],pool[key])
        local_cost=torch.minimum(local,local_cost)
    for budget in dict.fromkeys((tokens,min(tokens,8),1)):
        chosen=local_cost<0
        if budget<tokens:
            rows=local_cost.topk(budget,dim=0,largest=False).indices
            chosen=chosen&torch.zeros_like(chosen).scatter_(0,rows,True)
        candidate={}
        for key in pool:
            mask=chosen.reshape(tokens,blocks,*([1]*(pool[key].ndim-2)))
            candidate[key]=torch.where(mask,pool[key],parent[key])
        err=x-_deq_params(candidate)
        mse=err.square().mean(0)
        cost=(err.mean(0).square()+variance/tokens*mse).reshape(blocks,64).sum(1)
        take=(cost<best_cost)&(mse<=reference_mse*_VM_CAP+1e-20).reshape(blocks,64).all(1)
        for key in best:
            mask=take.reshape(1,blocks,*([1]*(best[key].ndim-2)))
            best[key]=torch.where(mask,candidate[key],best[key])
        best_cost=torch.where(take,cost,best_cost)
    return best
