"""Offline source builder for attention-mass-weighted logit regression ablations."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def weighted_source(power):
    assert power in (.5,1.)
    source=(ROOT/'tools/attention_factor_risk_core.py').read_text()
    old="""    h=features.transpose(1,2)@features/x.shape[1]
    h=h+torch.eye(h.shape[-1])*_AF_RIDGE
    beta=torch.linalg.solve(h,features.transpose(1,2)@target/x.shape[1])"""
    new="""    weights=[];offset=0
    for part in xs:
        length=part.shape[1]
        mass=target[:,offset:offset+length].softmax(1).mean(-1)*length
        weights.append(mass.clamp(.1,256.).pow(_AF_WEIGHT_POWER));offset+=length
    weight=torch.cat(weights,1)
    weight=weight/weight.mean(1,keepdim=True).clamp_min(1e-12)
    h=features.transpose(1,2)@(features*weight[:,:,None])/x.shape[1]
    h=h+torch.eye(h.shape[-1])*_AF_RIDGE
    beta=torch.linalg.solve(h,features.transpose(1,2)@(target*weight[:,:,None])/x.shape[1])"""
    assert source.count(old)==1
    return '_AF_WEIGHT_POWER='+str(power)+'\n'+source.replace(old,new,1)
