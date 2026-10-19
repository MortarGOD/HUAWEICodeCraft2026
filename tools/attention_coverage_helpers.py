"""Deterministic query-behavior coverage with empirical mixture weights."""
_CQ_MOMENT_WEIGHTS=False


def _cq_weight_fit(prototypes,signature,weights):
    gram=(prototypes@prototypes.transpose(1,2)).square()
    rhs=(prototypes@signature.transpose(1,2)).square().mean(2)
    lipschitz=torch.linalg.eigvalsh(gram)[:,-1].clamp_min(1e-30)[:,None]
    w=weights.clone();rank=w.shape[1]
    order=torch.arange(1,rank+1,dtype=w.dtype)[None]
    for _ in range(32):
        trial=w-((gram@w[:,:,None]).squeeze(2)-rhs)/lipschitz
        sorted_values=trial.sort(1,descending=True).values
        cumulative=sorted_values.cumsum(1)-1.
        rho=(sorted_values-cumulative/order>0).sum(1).clamp_min(1)
        threshold=cumulative.gather(1,(rho-1)[:,None])/rho[:,None]
        w=(trial-threshold).clamp_min(0.)
    return w


def _cq_select(queries,keys,rank):
    ix=torch.linspace(0,keys.shape[1]-1,min(128,keys.shape[1])).long()
    signature=(queries@keys[:,ix].transpose(1,2)/(keys.shape[-1]**.5)).softmax(-1)
    norm=signature.square().sum(-1)
    distances=(norm[:,:,None]+norm[:,None,:]-2*(signature@signature.transpose(1,2))).clamp_min(0.)
    rank=min(rank,queries.shape[1]);h,n=signature.shape[:2]
    current=(signature-signature.mean(1,keepdim=True)).square().sum(-1).argmin(1)
    selected=torch.zeros(h,n,dtype=torch.bool);nearest=torch.full((h,n),float('inf'))
    indices=[]
    for _ in range(rank):
        indices.append(current)
        selected.scatter_(1,current[:,None],True)
        dd=distances.gather(2,current[:,None,None].expand(h,n,1)).squeeze(2)
        nearest=torch.minimum(nearest,dd)
        current=nearest.masked_fill(selected,-1.).argmax(1)
    index=torch.stack(indices,1)
    assignment=distances.gather(2,index[:,None].expand(h,n,rank)).argmin(2)
    weights=torch.zeros(h,rank,dtype=queries.dtype)
    weights.scatter_add_(1,assignment,torch.ones(h,n,dtype=queries.dtype));weights/=n
    if _CQ_MOMENT_WEIGHTS:
        representatives=signature.gather(1,index[:,:,None].expand(h,rank,signature.shape[-1]))
        weights=_cq_weight_fit(representatives,signature,weights)
    return queries.gather(1,index[:,:,None].expand(h,rank,queries.shape[-1])),weights


def _af_probability(v,st,kh,d):
    x=v.reshape(-1,kh,d).transpose(0,1)
    logits=(_af_features(x,st)@st['beta']).transpose(1,2)
    p=logits.clamp(-40.,40.).softmax(-1)
    return p*(st['query_weights']*p.shape[1]).sqrt()[:,:,None]


def _am_moments(v,st,kh,d):
    b=_af_probability(v,st,kh,d)
    mean=(b*(st['query_weights']/b.shape[1]).sqrt()[:,:,None]).sum(1)
    variance=(b.square().mean(1)-mean.square()).clamp_min(0.)
    return mean,variance,torch.zeros(kh,v.shape[0],d)
