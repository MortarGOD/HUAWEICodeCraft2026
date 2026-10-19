"""Exact two-coordinate lattice moves against a fixed symmetric metric."""
import torch


def correlated_pairs(h,limit=256):
    if h.ndim!=3 or h.shape[-1]!=h.shape[-2] or h.shape[-1]<2 or limit<1:
        raise ValueError('Batched square metric with at least two channels and positive pair budget required')
    d=h.shape[-1]
    ij=torch.triu_indices(d,d,1)
    diagonal=h.diagonal(dim1=-2,dim2=-1).clamp_min(1e-30)
    score=h[:,ij[0],ij[1]].abs()/(diagonal[:,ij[0]]*diagonal[:,ij[1]]).sqrt()
    index=score.topk(min(limit,ij.shape[1]),dim=1).indices
    return ij[0][index],ij[1][index]


def single_best_gain(code,step,gradient,h):
    curvature=step.square()*h.diagonal(dim1=-2,dim2=-1)[:,None,:]
    plus=(curvature+2*step*gradient).masked_fill(code>=7,float('inf'))
    minus=(curvature-2*step*gradient).masked_fill(code<=-7,float('inf'))
    return torch.minimum(plus,minus).amin(-1)


def best_pair_move(code,step,gradient,h,pairs,traps_only=True):
    first,second=pairs;b,n,d=code.shape
    shape=(b,n,first.shape[1])
    i=first[:,None,:].expand(shape);j=second[:,None,:].expand(shape)
    ci=code.gather(2,i);cj=code.gather(2,j)
    di=step.gather(2,i);dj=step.gather(2,j)
    gi=gradient.gather(2,i);gj=gradient.gather(2,j)
    heads=torch.arange(b)[:,None]
    hii=h.diagonal(dim1=-2,dim2=-1).gather(1,first)[:,None,:]
    hjj=h.diagonal(dim1=-2,dim2=-1).gather(1,second)[:,None,:]
    hij=h[heads,first,second][:,None,:]
    best=torch.zeros(b,n,dtype=code.dtype)
    choice=torch.zeros(b,n,dtype=torch.long)
    sign_i=torch.zeros_like(best);sign_j=torch.zeros_like(best)
    for si,sj in ((-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)):
        one=2*si*di*gi+di.square()*hii
        two=2*sj*dj*gj+dj.square()*hjj
        delta=one+two+2*si*sj*di*dj*hij
        legal=((ci+si).abs()<=7)&((cj+sj).abs()<=7)
        if traps_only:legal=legal&(one>=0)&(two>=0)
        delta=delta.masked_fill(~legal,float('inf'))
        gain,index=delta.min(-1)
        take=gain<best
        best=torch.where(take,gain,best);choice=torch.where(take,index,choice)
        sign_i=torch.where(take,si,sign_i);sign_j=torch.where(take,sj,sign_j)
    return dict(gain=best,first=first.gather(1,choice),second=second.gather(1,choice),si=sign_i,sj=sign_j)


def best_pair_exact(code,step,gradient,h,pairs,traps_only=False):
    """Exact two-variable bounded minimum when traps_only=False.

    Enumerate 15 codes of i; j is the nearest constrained univariate optimum.
    A nonpositive j curvature is handled by checking its minimizing endpoint.
    """
    first,second=pairs;b,n,d=code.shape
    shape=(b,n,first.shape[1])
    i=first[:,None,:].expand(shape);j=second[:,None,:].expand(shape)
    ci=code.gather(2,i);cj=code.gather(2,j)
    di=step.gather(2,i);dj=step.gather(2,j)
    gi=gradient.gather(2,i);gj=gradient.gather(2,j)
    heads=torch.arange(b)[:,None]
    hii=h.diagonal(dim1=-2,dim2=-1).gather(1,first)[:,None,:]
    hjj=h.diagonal(dim1=-2,dim2=-1).gather(1,second)[:,None,:]
    hij=h[heads,first,second][:,None,:]
    best=torch.zeros(b,n,dtype=code.dtype);choice=torch.zeros(b,n,dtype=torch.long)
    jump_i=torch.zeros_like(best);jump_j=torch.zeros_like(best)
    for new_i in range(-7,8):
        si=new_i-ci;delta_i=si*di
        conditional=gj+hij*delta_i
        new_j=(cj-conditional/(hjj.clamp_min(1e-30)*dj)).round().clamp(-7,7)
        low=-7-cj;high=7-cj
        low_cost=2*conditional*low*dj+hjj*(low*dj).square()
        high_cost=2*conditional*high*dj+hjj*(high*dj).square()
        endpoint=torch.where(low_cost<=high_cost,-7.,7.)
        new_j=torch.where(hjj>0,new_j,endpoint)
        sj=new_j-cj;delta_j=sj*dj
        one=2*delta_i*gi+delta_i.square()*hii
        two=2*delta_j*gj+delta_j.square()*hjj
        cost=one+two+2*delta_i*delta_j*hij
        if traps_only:cost=cost.masked_fill((one<0)|(two<0),float('inf'))
        gain,index=cost.min(-1);take=gain<best
        best=torch.where(take,gain,best);choice=torch.where(take,index,choice)
        jump_i=torch.where(take,si.gather(2,index[:,:,None]).squeeze(-1),jump_i)
        jump_j=torch.where(take,sj.gather(2,index[:,:,None]).squeeze(-1),jump_j)
    return dict(gain=best,first=first.gather(1,choice),second=second.gather(1,choice),si=jump_i,sj=jump_j)


def pair_refine(x,values,step,h,pairs,rounds=1,traps_only=True,chunk=128,exact=False):
    if x.shape!=values.shape or x.shape!=step.shape or not bool(torch.isfinite(step).all()) or bool((step<=0).any()):
        raise ValueError('Equal BxNxD inputs with positive fixed steps required')
    if rounds<0 or chunk<1:
        raise ValueError('Nonnegative rounds and positive row chunk required')
    h=(h+h.transpose(1,2))*.5
    result=values.clone();total_moves=0;initial_stalled=0;escaped_stalled=0
    before_total=0.;after_total=0.;changed_rows=0
    for start in range(0,x.shape[1],chunk):
        target=x[:,start:start+chunk];unit=step[:,start:start+chunk]
        code=(result[:,start:start+chunk]/unit).round().clamp(-7,7)
        error=code*unit-target;gradient=error@h;cost=(error*gradient).sum(-1)
        before_total+=float(cost.sum())
        stalled=single_best_gain(code,unit,gradient,h)>=0
        initial_stalled+=int(stalled.sum());ever=torch.zeros_like(stalled)
        for _ in range(rounds):
            move=(best_pair_exact if exact else best_pair_move)(code,unit,gradient,h,pairs,traps_only)
            candidate=code.clone().scatter_add_(2,move['first'][:,:,None],move['si'][:,:,None])
            candidate.scatter_add_(2,move['second'][:,:,None],move['sj'][:,:,None])
            new_error=candidate*unit-target;new_gradient=new_error@h
            new_cost=(new_error*new_gradient).sum(-1)
            take=(move['gain']<0)&(new_cost<cost-cost.abs().clamp_min(1e-20)*1e-7)
            total_moves+=int(take.sum());ever|=take
            code=torch.where(take[:,:,None],candidate,code)
            gradient=torch.where(take[:,:,None],new_gradient,gradient)
            cost=torch.where(take,new_cost,cost)
            if not bool(take.any()):break
        changed_rows+=int(ever.sum());escaped_stalled+=int((ever&stalled).sum())
        after_total+=float(cost.sum());result[:,start:start+chunk]=code*unit
    return result,dict(rows=x.shape[0]*x.shape[1],initial_stalled=initial_stalled,
                       escaped_stalled=escaped_stalled,changed_rows=changed_rows,moves=total_moves,
                       before_metric=before_total,after_metric=after_total)
