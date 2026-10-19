"""Combine calibrated probability metrics with per-token HiF4 scale trees."""
_CQ_METADATA_CAP=1.1


def _cq_value_proposal(x,parent,plain,state,kh,d):
    tokens,channels=x.shape
    if tokens<=1 or tokens>4096:return parent
    probs=_vk_probabilities(x,state,kh)
    n=probs.shape[1];blocks=channels//64;penalty=1./tokens
    to_heads=lambda a:a.reshape(tokens,kh,d).transpose(0,1)
    def metric(error):
        e=to_heads(error)
        return ((probs@e).square().mean(1)+penalty*e.square().mean(1)).reshape(blocks,64).sum(1)
    original=_deq_params(parent);error=x-original
    eh=to_heads(error)
    residual=probs@eh
    gradient=(probs.transpose(1,2)@residual)/n+penalty/tokens*eh
    gradient=gradient.transpose(0,1).reshape(tokens,channels)
    curvature=probs.square().mean(1)+penalty/tokens
    curvature=curvature.transpose(0,1)[:,:,None].expand(tokens,kh,d).reshape(tokens,channels)
    reference_mse=error.square().mean(0).clamp_min(1e-30)
    best_cost=metric(error)
    pool={k:v.clone() for k,v in parent.items()}
    best={k:v.clone() for k,v in parent.items()}
    local_cost=torch.zeros(tokens,blocks)
    for specification in CAND_GRID:
        tree=_quantize_weighted(x,torch.ones(1,channels),grid=(specification,))
        candidate=_vk_refine(x,tree,state,kh,d)
        delta=_deq_params(candidate)-original
        local=(curvature*delta.square()-2*gradient*delta).reshape(tokens,blocks,64).sum(2)
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
        candidate={k:torch.where(chosen.reshape(tokens,blocks,*([1]*(pool[k].ndim-2))),pool[k],parent[k]) for k in pool}
        err=x-_deq_params(candidate)
        cost=metric(err);mse=err.square().mean(0)
        take=(cost<best_cost)&(mse<=reference_mse*_CQ_METADATA_CAP+1e-20).reshape(blocks,64).all(1)
        for key in best:
            mask=take.reshape(1,blocks,*([1]*(best[key].ndim-2)))
            best[key]=torch.where(mask,candidate[key],best[key])
        best_cost=torch.where(take,cost,best_cost)
    return best
