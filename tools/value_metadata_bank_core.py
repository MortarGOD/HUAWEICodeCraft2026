"""Enumerate scale bases before output-surrogate code refinement.

Unlike requantizing x+shift, this retains trees rejected by elementwise MSE
long enough to test their mean/variance output objective.
"""
def _vm_proposal(x,parent,variance):
    tokens,channels=x.shape
    if tokens<=1 or tokens>4096:return parent
    def costs(value):
        error=x-value
        mse=error.square().mean(0)
        objective=error.mean(0).square()+variance/tokens*mse
        return objective.reshape(-1,64).sum(1),mse
    best={k:v.clone() for k,v in parent.items()}
    best_cost,reference_mse=costs(_deq_params(best))
    reference_mse=reference_mse.clamp_min(1e-30)
    for specification in CAND_GRID:
        p=_quantize_weighted(x,torch.ones(1,channels),grid=(specification,))
        candidate=_vm_refine_new_tree(x,p,variance)
        cost,mse=costs(_deq_params(candidate))
        take=(cost<best_cost)&(mse<=reference_mse*_VM_CAP+1e-20).reshape(-1,64).all(1)
        for key in best:
            mask=take.reshape(1,-1,*([1]*(best[key].ndim-2)))
            best[key]=torch.where(mask,candidate[key],best[key])
        best_cost=torch.where(take,cost,best_cost)
    return best
