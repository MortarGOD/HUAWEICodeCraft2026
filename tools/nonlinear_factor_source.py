"""Offline builder: bounded nonlinear and sequence-relative Value features."""
from weighted_factor_source import ROOT,weighted_source


FEATURES='''
def _af_features(x,st):
    z=((x-st['center'][:,None])/st['scale'][:,None]).clamp(-8.,8.)
    ix=torch.linspace(0,z.shape[-1]-1,min(16,z.shape[-1])).long()
    small=z[:,:,ix]
    energy=z.square().mean(-1,keepdim=True)
    relative=energy/energy.mean(1,keepdim=True).clamp_min(1e-8)
    alignment=(z*z.mean(1,keepdim=True)).mean(-1,keepdim=True)
    extra=torch.cat((small.square()*.25,small*small.roll(1,-1)*.25,
        energy.log1p(),relative.clamp_max(64.).log1p(),
        z.abs().amax(-1,keepdim=True)*.25,alignment*.25),-1)
    return torch.cat((torch.ones_like(z[:,:,:1]),z,extra),-1)
'''


def nonlinear_source(power=None):
    source=weighted_source(power) if power is not None else (ROOT/'tools/attention_factor_risk_core.py').read_text()
    old='features=_af_features(x,st)'
    assert source.count(old)==1
    # Sequence-relative features must be computed separately for each training sequence.
    return source.replace(old,'features=torch.cat([_af_features(part,st) for part in xs],1)',1)+'\n'+FEATURES


def feature_checks(m):
    import torch
    generator=torch.Generator().manual_seed(351);checks=0
    for d in (8,64,512):
        st=dict(center=torch.zeros(2,d),scale=torch.ones(2,d))
        for t in (1,17,128):
            x=torch.randn(2,t,d,generator=generator);order=torch.randperm(t,generator=generator)
            before=x.clone();features=m._af_features(x,st)
            assert torch.isfinite(features).all() and torch.equal(x,before)
            assert torch.allclose(m._af_features(x[:,order],st),features[:,order],atol=2e-6,rtol=1e-5)
            assert torch.isfinite(m._af_features(torch.zeros_like(x),st)).all()
            assert features.shape==(2,t,d+1+2*min(16,d)+4)
            checks+=4
    return checks
