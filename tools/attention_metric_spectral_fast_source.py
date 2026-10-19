"""One-pass calibration fit for the v358 metric-plus-spectrum risk model."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fast_metric_spectral_source():
    source = (ROOT / 'tools/attention_factor_risk_core.py').read_text()
    assert source.count('_AF_SEARCH_RIDGE=.1') == 1
    source = source.replace('_AF_SEARCH_RIDGE=.1', '_AF_SEARCH_RIDGE=1.', 1)
    source += '''

_FS_METRIC_ALPHA=.25
_FS_METRIC_RIDGE=.01
_FS_SPECTRAL_ALPHA=.05
_FS_GRID=64
_FS_RANK=4
_fs_base_probability=_af_probability


def _fs_observed_metric(adaptive,target):
    gram=adaptive@adaptive.transpose(1,2)
    diagonal=gram.diagonal(dim1=1,dim2=2).mean(1).clamp_min(1e-12)
    eye=torch.eye(gram.shape[-1])
    inverse=torch.linalg.solve(gram+(_FS_METRIC_RIDGE*diagonal)[:,None,None]*eye,
                               torch.eye(gram.shape[-1]).expand_as(gram))
    mapping=(target@adaptive.transpose(1,2))@inverse
    return mapping.transpose(1,2)@mapping/target.shape[1]


def _fs_bin_probability(prob):
    kh,rows,t=prob.shape
    index=(torch.arange(t)*_FS_GRID//t).clamp_max(_FS_GRID-1)
    out=torch.zeros(kh,rows,_FS_GRID,dtype=prob.dtype)
    out.scatter_add_(2,index[None,None].expand_as(prob),prob)
    return out


def _fs_expand_factor(basis,t):
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
    proto=[];values=[];xs=[];ks=[];observed=[]
    for sample in pool:
        t=sample['v'][0].shape[0]
        index=torch.linspace(0,t-1,min(16,t)).long()
        q=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,index)
        key=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
        target=_vr_probs(q,key,qh,kh,d).reshape(kh,qh//kh,-1,t).reshape(kh,-1,t)
        value=dequantize_nvfp4(*sample['v']).float()
        proto.append(q.reshape(-1,kh,qh//kh,d).permute(1,0,2,3).reshape(kh,-1,d))
        values.append(value);xs.append(value.reshape(t,kh,d).transpose(0,1))
        ks.append(key.reshape(t,kh,d).transpose(0,1));observed.append(target)
    proto=torch.cat(proto,1)
    index=torch.linspace(0,proto.shape[1]-1,min(_AF_RANK,proto.shape[1])).long()
    proto=proto[:,index];x=torch.cat(xs,1);key=torch.cat(ks,1)
    center=x.mean(1);scale=(x-center[:,None]).square().mean(1).sqrt().clamp_min(1e-6)
    state=dict(center=center.contiguous(),scale=scale.contiguous())
    features=_af_features(x,state);target=(key@proto.transpose(1,2))/(d**.5)
    h=features.transpose(1,2)@features/x.shape[1]
    h=h+torch.eye(h.shape[-1])*_AF_RIDGE
    beta=torch.linalg.solve(h,features.transpose(1,2)@target/x.shape[1])
    state.update(beta=beta.contiguous(),search_ridge=_AF_SEARCH_RIDGE)
    metrics=[];rows=[]
    for value,target in zip(values,observed):
        adaptive=_fs_base_probability(value,state,kh,d)
        metrics.append(_fs_observed_metric(adaptive,target))
        rows.append(_fs_bin_probability(target))
    metric=torch.stack(metrics).mean(0);rank=metric.shape[-1];eye=torch.eye(rank)
    metric=(1.-_FS_METRIC_ALPHA)*eye/rank+_FS_METRIC_ALPHA*metric
    metric=(metric+metric.transpose(1,2))*.5
    transform=torch.linalg.cholesky(metric).transpose(1,2)*(rank**.5)
    rows=torch.cat(rows,1)
    _,singular,vh=torch.linalg.svd(rows,full_matrices=False)
    use=min(_FS_RANK,vh.shape[1])
    basis=vh[:,:use]*singular[:,:use,None]*(use/rows.shape[1])**.5
    state.update(metric_transform=transform.contiguous(),spectral_basis=basis.contiguous())
    return state


def _af_probability(v,state,kh,d):
    adaptive=state['metric_transform']@_fs_base_probability(v,state,kh,d)
    spectral=_fs_expand_factor(state['spectral_basis'],v.shape[0])
    total=adaptive.shape[1]+spectral.shape[1]
    adaptive=adaptive*(total*(1.-_FS_SPECTRAL_ALPHA)/adaptive.shape[1])**.5
    spectral=spectral*(total*_FS_SPECTRAL_ALPHA/spectral.shape[1])**.5
    return torch.cat((adaptive,spectral),1)
'''
    return source
