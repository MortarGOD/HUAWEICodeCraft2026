"""Blend adaptive factor risk with a small calibration positional spectrum."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hybrid_risk_source(alpha=.05, grid=64, spectral_rank=4, adaptive_rank=16):
    assert 0. < alpha < 1.
    source = (ROOT / 'tools/attention_factor_risk_core.py').read_text()
    assert source.count('_AF_SEARCH_RIDGE=.1') == 1
    source = source.replace('_AF_SEARCH_RIDGE=.1', '_AF_SEARCH_RIDGE=1.', 1)
    source += f'''

_HY_ALPHA={float(alpha)!r}
_HY_GRID={int(grid)}
_HY_RANK={int(spectral_rank)}
_HY_ADAPTIVE={int(adaptive_rank)}
_hy_parent_fit=_am_fit
_hy_parent_probability=_af_probability


def _hy_bin_probability(prob,qh,kh):
    t=prob.shape[-1];g=qh//kh
    rows=prob.reshape(kh,g,-1,t).reshape(kh,-1,t)
    index=(torch.arange(t)*_HY_GRID//t).clamp_max(_HY_GRID-1)
    out=torch.zeros(kh,rows.shape[1],_HY_GRID,dtype=rows.dtype)
    out.scatter_add_(2,index[None,None].expand_as(rows),rows)
    return out


def _hy_expand_factor(basis,t):
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
    state=_hy_parent_fit(samples,parent,qh,kh,d)
    if state is None:return None
    pool=[s for s in samples[:-1] if 4<=s['v'][0].shape[0]<=2048
          and s['v'][0].numel()<=4194304][-2:]
    rows=[]
    for sample in pool:
        t=sample['v'][0].shape[0]
        index=torch.linspace(0,t-1,min(16,t)).long()
        q=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,index)
        k=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
        rows.append(_hy_bin_probability(_vr_probs(q,k,qh,kh,d),qh,kh))
    rows=torch.cat(rows,1)
    _,singular,vh=torch.linalg.svd(rows,full_matrices=False)
    use=min(_HY_RANK,vh.shape[1])
    basis=vh[:,:use]*singular[:,:use,None]*(use/rows.shape[1])**.5
    state.update(spectral_basis=basis.contiguous())
    return state


def _af_probability(v,state,kh,d):
    adaptive=_hy_parent_probability(v,state,kh,d)
    if _HY_ADAPTIVE<adaptive.shape[1]:
        index=torch.linspace(0,adaptive.shape[1]-1,_HY_ADAPTIVE).long()
        adaptive=adaptive[:,index]
    spectral=_hy_expand_factor(state['spectral_basis'],v.shape[0])
    total=adaptive.shape[1]+spectral.shape[1]
    adaptive=adaptive*(total*(1.-_HY_ALPHA)/adaptive.shape[1])**.5
    spectral=spectral*(total*_HY_ALPHA/spectral.shape[1])**.5
    return torch.cat((adaptive,spectral),1)
'''
    return source
