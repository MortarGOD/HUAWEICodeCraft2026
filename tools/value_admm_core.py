"""v399: full-token metadata/code projection with low-rank ADMM splitting."""
_AD_ROUNDS=3
AD_DEBUG={}
_ad_parent_refine=_vk_refine


def _ad_inverse(rhs,p,rho,lam):
    # Solve ((rho+lam)I + P^T P/R) z=rhs without a token-square matrix.
    n=p.shape[1];a=rho+lam
    small=torch.bmm(p,p.transpose(1,2))/n+torch.diag_embed(a[:,None].expand(-1,n))
    chol,info=torch.linalg.cholesky_ex(small)
    if bool(info.any()):
        raise ValueError('AD small system is not positive definite')
    correction=torch.cholesky_solve(torch.bmm(p,rhs),chol)
    return (rhs-torch.bmm(p.transpose(1,2),correction)/n)/a[:,None,None]


def _ad_quantize(x,parent,probs,active,kh,d):
    t=x.shape[0]
    ids=torch.where(active)[0]
    if not ids.numel():return parent
    blocks=active.repeat_interleave(d//64)
    source=x.reshape(t,kh,d)[:,active].transpose(0,1).contiguous()
    current=_deq_params(parent).float().reshape(t,kh,d)[:,active].transpose(0,1).contiguous()
    pp=probs[active].float();n=pp.shape[1]
    original=(source-current).square().mean(1).clamp_min(1e-30)
    def loss(value):
        e=source-value
        return (torch.bmm(pp,e).square().mean(1)+e.square().mean(1)/t).reshape(-1,d//64,64).sum(-1)
    bestcost=loss(current)
    best={name:value[:,blocks].clone() for name,value in parent.items()}
    lam=1./(t*t)
    rho=(pp.square().sum((1,2))/(n*t)+lam).clamp_min(1e-20)
    dual=torch.zeros_like(current)
    for _ in range(_AD_ROUNDS):
        offset=current-dual-source
        continuous=source+_ad_inverse(rho[:,None,None]*offset,pp,rho,lam)
        target=(continuous+dual).transpose(0,1).reshape(t,-1).contiguous()
        trial=_quantize_weighted(target,torch.ones(1,target.shape[1]))
        value=_deq_params(trial).float().reshape(t,ids.numel(),d).transpose(0,1).contiguous()
        mse=(source-value).square().mean(1)
        cost=loss(value)
        take=(cost<bestcost)&(mse<=original*_VK_MSE_CAP+1e-20).reshape(-1,d//64,64).all(-1)&torch.isfinite(cost)
        mask=take.reshape(1,-1,1,1,1)
        best={name:torch.where(mask,trial[name],best[name]) for name in best}
        bestcost=torch.where(take,cost,bestcost)
        dual=dual+continuous-value
        current=value
    result={name:value.clone() for name,value in parent.items()}
    for name in result:result[name][:,blocks]=best[name]
    return result


def _vk_refine(x,p,state,kh,d):
    if x.shape[0]<=1 or state is None or not bool(state['active'].any()):return p
    if x.shape[0]>4096 or d%64:return _ad_parent_refine(x,p,state,kh,d)
    probs=_vk_probabilities(x,state,kh)
    if probs.shape[1]>128:return _ad_parent_refine(x,p,state,kh,d)
    try:
        return _ad_quantize(x,p,probs,state['active'],kh,d)
    except (RuntimeError,ValueError):
        return _ad_parent_refine(x,p,state,kh,d)
