"""Small invariant check for a retained, non-production STE scale experiment."""
import copy
import json
from pathlib import Path
import torch
from probe_attention_gates import configured
from build_fitted_dataset import quantize_nvfp4
from reciprocal_scale_core import inverse_factor, prepare, train
from self_check import validate_frozen_state


def main():
    torch.set_num_threads(4)
    m=configured('frames',0)
    rng=torch.Generator().manual_seed(5316)
    qh,kh,d=2,1,64
    samples=[{r:quantize_nvfp4(torch.randn(t,(qh if r=='q' else kh)*d,generator=rng)) for r in 'qkv'} for t in (9,16,32)]
    parent=m._gate_parent_calibration(samples,qh,kh,d)
    ctx=prepare(m,samples,parent,qh,kh,d,queries=8)
    changed=copy.deepcopy(samples)
    changed[-1]=changed[0]
    other=prepare(m,changed,parent,qh,kh,d,queries=8)
    assert torch.equal(ctx['hq'],other['hq']) and torch.equal(ctx['hk'],other['hk'])
    h=ctx['hq']
    u=inverse_factor(h,m.GPTQ_DAMP).double()
    hd=h.double()+torch.eye(d)[None]*h.double().diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*m.GPTQ_DAMP
    assert torch.allclose(u.transpose(1,2)@u,torch.linalg.inv(hd),atol=1e-6,rtol=1e-5)
    initial,final,debug=train(m,samples,parent,qh,kh,d,steps=1,queries=8)
    assert not validate_frozen_state(initial,'initial') and not validate_frozen_state(final,'final')
    assert debug['reciprocal_error']<2e-7
    history=torch.tensor(debug['history'])
    assert bool((history[1:]<=history[:-1]+1e-7).all())
    report=dict(passed=True,hold_excluded=True,upper_factor_matches_inverse=True,training_monotone=True,
                finite_detached_states=True,debug=debug,scope='Small v314 research case, not production or broad efficacy validation.')
    (Path(__file__).resolve().parents[1]/'docs/reciprocal-scale-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
