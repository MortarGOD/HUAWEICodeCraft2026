"""Compose the learned adaptive-factor metric with a small positional spectrum."""
from pathlib import Path

from attention_factor_metric_source import factor_metric_source

ROOT = Path(__file__).resolve().parents[1]


def metric_spectral_source(spectral_alpha=.05, grid=64, spectral_rank=4):
    assert 0. < spectral_alpha < 1.
    source = factor_metric_source(.25, .01)
    source += f'''

_MS_ALPHA={float(spectral_alpha)!r}
_MS_GRID={int(grid)}
_MS_RANK={int(spectral_rank)}
_ms_parent_fit=_am_fit
_ms_parent_probability=_af_probability


def _ms_bin_probability(prob,qh,kh):
    t=prob.shape[-1];g=qh//kh
    rows=prob.reshape(kh,g,-1,t).reshape(kh,-1,t)
    index=(torch.arange(t)*_MS_GRID//t).clamp_max(_MS_GRID-1)
    out=torch.zeros(kh,rows.shape[1],_MS_GRID,dtype=rows.dtype)
    out.scatter_add_(2,index[None,None].expand_as(rows),rows)
    return out


def _ms_expand_factor(basis,t):
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
    state=_ms_parent_fit(samples,parent,qh,kh,d)
    if state is None:return None
    pool=[s for s in samples[:-1] if 4<=s['v'][0].shape[0]<=2048
          and s['v'][0].numel()<=4194304][-2:]
    rows=[]
    for sample in pool:
        t=sample['v'][0].shape[0]
        index=torch.linspace(0,t-1,min(16,t)).long()
        q=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,index)
        k=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
        rows.append(_ms_bin_probability(_vr_probs(q,k,qh,kh,d),qh,kh))
    rows=torch.cat(rows,1)
    _,singular,vh=torch.linalg.svd(rows,full_matrices=False)
    use=min(_MS_RANK,vh.shape[1])
    basis=vh[:,:use]*singular[:,:use,None]*(use/rows.shape[1])**.5
    state.update(spectral_basis=basis.contiguous())
    return state


def _af_probability(v,state,kh,d):
    adaptive=_ms_parent_probability(v,state,kh,d)
    spectral=_ms_expand_factor(state['spectral_basis'],v.shape[0])
    total=adaptive.shape[1]+spectral.shape[1]
    adaptive=adaptive*(total*(1.-_MS_ALPHA)/adaptive.shape[1])**.5
    spectral=spectral*(total*_MS_ALPHA/spectral.shape[1])**.5
    return torch.cat((adaptive,spectral),1)
'''
    return source
