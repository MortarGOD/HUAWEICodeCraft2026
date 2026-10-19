"""Deployable low-rank attention-output risk learned from calibration P^T P."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def spectral_risk_source(grid=128, rank=16, cross_guard=False):
    """Return the factor-risk core with an SVD-optimal fixed positional Gram."""
    source = (ROOT / 'tools/attention_factor_risk_core.py').read_text()
    assert source.count('_AF_SEARCH_RIDGE=.1') == 1
    source = source.replace('_AF_SEARCH_RIDGE=.1', '_AF_SEARCH_RIDGE=1.', 1)
    source += f'''

_SR_GRID={int(grid)}
_SR_RANK={int(rank)}


def _sr_bin_probability(prob,qh,kh):
    # Preserve probability mass in normalized-position bins.  Grouped-query
    # heads and sampled queries are independent observations of the same KV
    # head risk matrix.
    t=prob.shape[-1];g=qh//kh
    rows=prob.reshape(kh,g,-1,t).reshape(kh,-1,t)
    index=(torch.arange(t)*_SR_GRID//t).clamp_max(_SR_GRID-1)
    out=torch.zeros(kh,rows.shape[1],_SR_GRID,dtype=rows.dtype)
    out.scatter_add_(2,index[None,None].expand_as(rows),rows)
    return out


def _sr_expand_factor(basis,t):
    kh,rank,grid=basis.shape
    if t>=grid:
        index=(torch.arange(t)*grid//t).clamp_max(grid-1)
        count=torch.bincount(index,minlength=grid).clamp_min(1).to(basis.dtype)
        return basis.gather(2,index[None,None].expand(kh,rank,t))/count[index][None,None]
    index=(torch.arange(grid)*t//grid).clamp_max(t-1)
    out=torch.zeros(kh,rank,t,dtype=basis.dtype)
    out.scatter_add_(2,index[None,None].expand(kh,rank,grid),basis)
    return out


def _am_fit(samples,parent,qh,kh,d):
    pool=[s for s in samples[:-1] if 4<=s['v'][0].shape[0]<=2048
          and s['v'][0].numel()<=4194304][-2:]
    if len(pool)<2:return None
    rows=[]
    for sample in pool:
        t=sample['v'][0].shape[0]
        index=torch.linspace(0,t-1,min(16,t)).long()
        q=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,index)
        k=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
        rows.append(_sr_bin_probability(_vr_probs(q,k,qh,kh,d),qh,kh))
    rows=torch.cat(rows,1)
    _,singular,vh=torch.linalg.svd(rows,full_matrices=False)
    use=min(_SR_RANK,vh.shape[1])
    # mean_r ||factor_r E||^2 equals the best rank-use approximation of
    # mean_query ||prob_query E||^2.
    basis=vh[:,:use]*singular[:,:use,None]*(use/rows.shape[1])**.5
    return dict(basis=basis.contiguous(),search_ridge=_AF_SEARCH_RIDGE)


def _af_probability(v,st,kh,d):
    return _sr_expand_factor(st['basis'],v.shape[0])


def _am_moments(v,st,kh,d):
    p=_af_probability(v,st,kh,d)
    # Compatibility only: the spectral factor is signed and _am_quant uses it
    # directly.  These moments are not interpreted as probabilities.
    return p.mean(1),p.var(1,unbiased=False),torch.zeros(kh,v.shape[0],d)
'''
    if cross_guard:
        source += '''

def _sr_fold_loss(sample,parent,base,trial,qh,kh,d):
    t=sample['v'][0].shape[0]
    index=torch.linspace(0,t-1,min(32,t)).long()
    rawq=dequantize_nvfp4(sample['q'][0][index],sample['q'][1][index]).float()
    rawk=dequantize_nvfp4(*sample['k']).float()
    v=dequantize_nvfp4(*sample['v']).float()
    q=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,index)
    k=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
    probability=_vr_probs(q,k,qh,kh,d)
    truth=_vr_probs(rawq,rawk,qh,kh,d)
    target=truth@v.reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    def loss(params):
        values=_deq_params(params).reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        error=(probability@values-target).square().mean(-1).reshape(kh,qh//kh,-1).mean(1)
        return torch.stack((error[:,::2].mean(1),error[:,1::2].mean(1)))
    return loss(base),loss(trial)


def _am_guard(samples,parent,qh,kh,d,model=None):
    AM_DEBUG.clear();AM_DEBUG.update(error=False,adopted=0)
    if len(samples)<3 or d%64 or not parent['q_state'].get('gq'):return parent
    state=_am_fit(samples,parent,qh,kh,d) if model is None else dict(model)
    if state is None:return parent
    state=dict(state,uncertainty=_AM_UNCERTAINTY,gradient_gain=_AM_GRADIENT_GAIN)
    ratios=[]
    for sample in samples[:2]+samples[-1:]:
        t=sample['v'][0].shape[0]
        if not 4<=t<=4096 or sample['v'][0].numel()>4194304:continue
        base=_am_parent_v(*sample['v'],kh,d,parent['v_state'])
        trial=_am_quant(dequantize_nvfp4(*sample['v']).float(),base,state,kh,d)
        old,new=_sr_fold_loss(sample,parent,base,trial,qh,kh,d)
        ratios.append(new/old.clamp_min(1e-30))
    if len(ratios)<2:return parent
    ratio=torch.stack(ratios)
    take=(ratio.amax((0,1))<1.002)&(ratio.mean((0,1))<.995)&torch.isfinite(ratio).all((0,1))
    AM_DEBUG.update(adopted=int(take.sum()),hold_ratio=ratio.mean((0,1)).tolist())
    if not bool(take.any()):return parent
    state.update(active=take,parent=parent['v_state'])
    return dict(parent,v_state=dict(moment=state))
'''
    return source
