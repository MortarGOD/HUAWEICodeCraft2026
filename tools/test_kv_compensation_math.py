"""Validate the output gradient and discrete/nonlinear acceptance independently."""
import json
from pathlib import Path
import torch
from kv_compensation_math import output_gradient,refine_keys


def main():
    torch.set_num_threads(4)
    rng=torch.Generator().manual_seed(31709)
    q=torch.randn(1,7,8,dtype=torch.float64,generator=rng)
    k=torch.randn(1,12,8,dtype=torch.float64,generator=rng)*.5
    v=torch.randn(1,12,8,dtype=torch.float64,generator=rng)*.5
    target=torch.randn(1,7,8,dtype=torch.float64,generator=rng)*.3
    g,h,_=output_gradient(q,k,v,target)
    f=lambda x:torch.softmax(q@x.transpose(-1,-2)/8**.5,-1)@v
    actual=torch.autograd.functional.jacobian(lambda x:((f(x)-target).square().sum()*.5),k)
    jac=torch.autograd.functional.jacobian(f,k).reshape(-1,k.numel())
    expected_h=jac.square().sum(0).reshape_as(k)
    assert torch.allclose(g,actual,rtol=1e-10,atol=1e-12)
    assert torch.allclose(h,expected_h,rtol=1e-10,atol=1e-12)
    step=torch.full_like(k,.125)
    kq=(k/step).round().clamp(-7,7)*step
    vq=(v/step).round().clamp(-7,7)*step
    result,stats=refine_keys(q,k,kq,v,vq,step,8,4.)
    assert bool(((result/step).round()==result/step).all())
    assert bool((result.abs()<=7*step).all())
    assert bool((stats['history'][1:]<=stats['history'][:-1]+1e-15).all())
    assert bool((stats['final']<stats['initial']).all())
    reference=q@k.transpose(-1,-2)/8**.5
    distortion=(q@result.transpose(-1,-2)/8**.5-reference).square().mean((-1,-2))
    assert bool((distortion<=stats['logit_limit']+1e-14).all())
    report=dict(passed=True,gradient_max_error=float((g-actual).abs().max()),
                gn_diagonal_max_error=float((h-expected_h).abs().max()),
                initial_loss=stats['initial'].tolist(),final_loss=stats['final'].tolist(),
                monotone=True,legal_fixed_lattice=True,logit_budget=True,
                warning='Toy mathematical test; not real-data or online gain.')
    path=Path(__file__).resolve().parents[1]/'docs/kv-compensation-math-invariants.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
