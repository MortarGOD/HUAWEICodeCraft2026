"""Refine integer codes on each new scale tree before comparing to the parent."""
_VM_INNER_ROUNDS=8


def _vm_refine_new_tree(x,p,variance):
    tokens=x.shape[0]
    steps=_params_unit_flat(p)*.25
    best=_deq_params(p)
    error=x-best
    cost=error.mean(0).square()+variance/tokens*error.square().mean(0)
    shift=error.mean(0)
    for _ in range(_VM_INNER_ROUNDS):
        proposal=torch.round((x+shift)/steps).clamp(-7,7)*steps
        error=x-proposal
        mean=error.mean(0)
        next_cost=mean.square()+variance/tokens*error.square().mean(0)
        take=next_cost<cost
        best=torch.where(take[None],proposal,best)
        cost=torch.minimum(cost,next_cost)
        shift=shift+mean*.75
    return _values_to_params(best,p)


def _vm_proposal(x,parent,variance):
    tokens,channels=x.shape
    if tokens<=1 or tokens>4096:return parent
    def costs(value):
        error=x-value
        mse=error.square().mean(0)
        objective=error.mean(0).square()+variance/tokens*mse
        return objective.reshape(-1,64).sum(1),mse,error.mean(0)
    best={k:v.clone() for k,v in parent.items()}
    best_cost,reference_mse,shift=costs(_deq_params(best))
    reference_mse=reference_mse.clamp_min(1e-30)
    for _ in range(_VM_ROUNDS):
        candidate=_vm_refine_new_tree(x,_dyn_table(x+shift,None,False),variance)
        cost,mse,mean=costs(_deq_params(candidate))
        accept=(cost<best_cost)&(mse<=reference_mse*_VM_CAP+1e-20).reshape(-1,64).all(1)
        for key in best:
            mask=accept.reshape(1,-1,*([1]*(best[key].ndim-2)))
            best[key]=torch.where(mask,candidate[key],best[key])
        best_cost=torch.where(accept,cost,best_cost)
        shift=shift+mean*.75
    return best
