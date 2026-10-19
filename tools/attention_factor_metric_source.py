"""Learn a PSD metric over adaptive attention-risk factors."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def factor_metric_source(alpha=.25, ridge=.01, calibrations=2):
    assert 0. < alpha <= 1.
    assert ridge > 0.
    source = (ROOT / 'tools/attention_factor_risk_core.py').read_text()
    assert source.count('_AF_SEARCH_RIDGE=.1') == 1
    source = source.replace('_AF_SEARCH_RIDGE=.1', '_AF_SEARCH_RIDGE=1.', 1)
    source += f'''

_FM_ALPHA={float(alpha)!r}
_FM_RIDGE={float(ridge)!r}
_fm_parent_fit=_am_fit
_fm_parent_probability=_af_probability


def _fm_observed_metric(adaptive,target):
    # Project all observed calibration query probabilities onto the row space
    # of the 16 adaptive factors.  C^T C/m is the PSD risk metric induced by
    # the least-squares projection target ~= C @ adaptive.
    gram=adaptive@adaptive.transpose(1,2)
    diagonal=gram.diagonal(dim1=1,dim2=2).mean(1).clamp_min(1e-12)
    eye=torch.eye(gram.shape[-1])
    inverse=torch.linalg.solve(gram+(_FM_RIDGE*diagonal)[:,None,None]*eye,
                               torch.eye(gram.shape[-1]).expand_as(gram))
    mapping=(target@adaptive.transpose(1,2))@inverse
    return mapping.transpose(1,2)@mapping/target.shape[1]


def _am_fit(samples,parent,qh,kh,d):
    state=_fm_parent_fit(samples,parent,qh,kh,d)
    if state is None:return None
    pool=[s for s in samples[:-1] if 4<=s['v'][0].shape[0]<=2048
          and s['v'][0].numel()<=4194304][-2:]
    metrics=[]
    for sample in pool:
        t=sample['v'][0].shape[0]
        index=torch.linspace(0,t-1,min(16,t)).long()
        q=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,index)
        k=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state']))
        target=_vr_probs(q,k,qh,kh,d).reshape(kh,qh//kh,-1,t).reshape(kh,-1,t)
        value=dequantize_nvfp4(*sample['v']).float()
        adaptive=_fm_parent_probability(value,state,kh,d)
        metrics.append(_fm_observed_metric(adaptive,target))
    metric=torch.stack(metrics).mean(0)
    rank=metric.shape[-1];eye=torch.eye(rank)
    metric=(1.-_FM_ALPHA)*eye/rank+_FM_ALPHA*metric
    metric=(metric+metric.transpose(1,2))*.5
    transform=torch.linalg.cholesky(metric).transpose(1,2)*(rank**.5)
    state.update(metric_transform=transform.contiguous())
    return state


def _af_probability(v,state,kh,d):
    adaptive=_fm_parent_probability(v,state,kh,d)
    return state['metric_transform']@adaptive
'''
    if calibrations != 2:
        assert 2 < calibrations <= 4
        old = "and s['v'][0].numel()<=4194304][-2:]"
        new = f"and s['v'][0].numel()<=4194304][-{int(calibrations)}:]"
        assert source.count(old) == 2
        source = source.replace(old, new)
    return source
