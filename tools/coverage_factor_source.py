"""Offline source builder and mathematical checks for query coverage."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def coverage_source():
    source=(ROOT/'tools/attention_factor_risk_core.py').read_text()
    # Formal factor versions use ridge=1.0. Re-emitting the whole function
    # core must not silently restore the research default of 0.1.
    assert source.count('_AF_SEARCH_RIDGE=.1')==1
    source=source.replace('_AF_SEARCH_RIDGE=.1','_AF_SEARCH_RIDGE=1.',1)
    old="""    ix=torch.linspace(0,proto.shape[1]-1,min(_AF_RANK,proto.shape[1])).long()
    proto=proto[:,ix];x=torch.cat(xs,1);k=torch.cat(ks,1)"""
    new="""    x=torch.cat(xs,1);k=torch.cat(ks,1)
    proto,query_weights=_cq_select(proto,k,_AF_RANK)"""
    assert source.count(old)==1
    source=source.replace(old,new,1)
    source=source.replace('st.update(beta=beta.contiguous(),search_ridge=_AF_SEARCH_RIDGE)',
        'st.update(beta=beta.contiguous(),search_ridge=_AF_SEARCH_RIDGE,query_weights=query_weights.contiguous())',1)
    return source+'\n'+(ROOT/'tools/attention_coverage_helpers.py').read_text()


def coverage_checks(m):
    import torch
    gen=torch.Generator().manual_seed(352);checks=0
    for n in (1,7,32):
        q=torch.randn(2,n,8,generator=gen,dtype=torch.float64)
        k=torch.randn(2,19,8,generator=gen,dtype=torch.float64)
        for zero in (False,True):
            qq=q*0 if zero else q
            selected,w=m._cq_select(qq,k,16)
            assert selected.shape==(2,min(n,16),8)
            assert torch.isfinite(w).all() and (w>=0).all()
            assert torch.allclose(w.sum(1),torch.ones(2,dtype=w.dtype));checks+=3
            p=(selected@k.transpose(1,2)).softmax(-1)
            b=p*(w*p.shape[1]).sqrt()[:,:,None]
            h=b.transpose(1,2)@b/p.shape[1]
            direct=(p*w[:,:,None]).transpose(1,2)@p
            assert torch.allclose(h,direct,atol=1e-13,rtol=0);checks+=1
            allp=(qq@k.transpose(1,2)).softmax(-1)
            fitted=m._cq_weight_fit(p,allp,w)
            target=allp.transpose(1,2)@allp/allp.shape[1]
            new=(p*fitted[:,:,None]).transpose(1,2)@p
            assert torch.allclose(fitted.sum(1),torch.ones(2,dtype=w.dtype),atol=1e-12,rtol=0)
            assert (fitted>=0).all() and torch.isfinite(fitted).all()
            assert bool(((new-target).square().sum((1,2))<=(direct-target).square().sum((1,2))+1e-12).all());checks+=3
    return checks
