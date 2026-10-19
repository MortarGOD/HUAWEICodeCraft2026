"""Calibration-only bounded-feature regression; owns no test labels or I/O."""
import torch


def fit_predictor(samples, source='q', features=256, ridge=.01, max_rows=512):
    if not samples or source not in ('q', 'k', 'v') or features<=0 or ridge<=0 or max_rows<=0:
        raise ValueError('Nonempty calibration, valid role, and positive budgets are required')
    targets=tuple(r for r in 'qkv' if r!=source)
    c=samples[0][source].shape[1]
    rng=torch.Generator().manual_seed(31673+c)
    indices=torch.randperm(c,generator=rng)[:min(features,c)].sort().values
    x,y=[],[]
    for sample in samples:
        t=sample[source].shape[0]
        rows=torch.linspace(0,t-1,min(t,max_rows)).long()
        x.append(sample[source][rows][:,indices].float())
        y.append(torch.cat([sample[r][rows].float() for r in targets],-1))
    x,y=torch.cat(x),torch.cat(y)
    center=x.mean(0)
    scale=(x-center).square().mean(0).sqrt()
    scale=scale.clamp_min(scale.mean().clamp_min(1e-12)*1e-4)
    target_center=y.mean(0)
    f=(x-center)/scale
    gram=f.T@f + torch.eye(f.shape[1])*(ridge*f.shape[0])
    coef=torch.cholesky_solve(f.T@(y-target_center),torch.linalg.cholesky(gram))
    return dict(source=source,targets=list(targets),indices=indices.contiguous(),center=center.contiguous(),
                scale=scale.contiguous(),coef=coef.contiguous(),target_center=target_center.contiguous(),
                widths=[samples[0][r].shape[1] for r in targets])


def predict_roles(x,state):
    f=(x[:,state['indices']].float()-state['center'])/state['scale']
    y=f@state['coef']+state['target_center']
    return dict(zip(state['targets'],y.split(state['widths'],dim=-1)))
