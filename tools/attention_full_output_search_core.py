"""Full output-space nonlinear search, not projected rank-16 loss.

Extends the existing NQ prototype research with exact candidate acceptance
for one-coordinate and four-coordinate simultaneous steps.
"""
_NQ_OUTPUT=True
_NQ_ALIGNED=False
_NQ_ROUNDS=4


def _nq_refine(raw,p,bank,qh,d):
    rows=raw.shape[0]
    if not 1<rows<=4096:return p
    active=bank['active'];ids=torch.arange(qh)[active]
    if not ids.numel():return p
    repeat=qh//bank['key'].shape[0]
    k=bank['key'].float().repeat_interleave(repeat,0)[active]
    kr=bank['raw'].float().repeat_interleave(repeat,0)[active]
    vv=bank['value'].float().repeat_interleave(repeat,0)[active]
    vr=bank['raw_value'].float().repeat_interleave(repeat,0)[active]
    value=_deq_params(p).reshape(rows,qh,d).transpose(0,1).contiguous()
    step=_params_unit_flat(p).reshape(rows,qh,d).transpose(0,1)*.25
    x=raw.reshape(rows,qh,d).transpose(0,1)
    for start in range(0,rows,128):
        stop=min(rows,start+128)
        v=value[active,start:stop].clone();delta=step[active,start:stop]
        goal=torch.softmax(x[active,start:stop]@kr.transpose(1,2),-1)@vr
        def loss(z):
            return (torch.softmax(z@k.transpose(1,2),-1)@vv-goal).square().sum(-1)
        best=loss(v)
        for _ in range(bank['rounds']):
            prob=torch.softmax(v@k.transpose(1,2),-1)
            residual=prob@vv-goal
            influence=residual@vv.transpose(1,2)
            gradient=2*(prob*(influence-(prob*influence).sum(-1,keepdim=True)))@k
            direction=torch.where(gradient<0,1.,-1.)
            level=torch.round(v/delta)
            estimate=(gradient*direction*delta).masked_fill((level+direction).abs()>7,float('inf'))
            indices=estimate.topk(min(4,d),dim=-1,largest=False).indices
            original=v
            any_take=False
            for width in (1,min(4,d)):
                coord=indices[...,:width]
                change=(direction*delta).gather(-1,coord)
                legal=torch.isfinite(estimate.gather(-1,coord))
                proposal=original.clone().scatter_add_(-1,coord,change*legal)
                cost=loss(proposal)
                take=(cost<best)&torch.isfinite(cost)
                any_take=any_take or bool(take.any())
                v=torch.where(take[...,None],proposal,v)
                best=torch.where(take,cost,best)
            if not any_take:break
        value[ids[:,None],torch.arange(start,stop)[None]]=v
    return _values_to_params(value.transpose(0,1).reshape(raw.shape).contiguous(),p)
