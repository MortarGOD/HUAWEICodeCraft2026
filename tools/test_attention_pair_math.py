"""Exhaustive pair gains, a genuine 1-opt trap, boundaries and monotonicity."""
import itertools
import json
from pathlib import Path
import torch
from attention_pair_math import correlated_pairs,best_pair_move,best_pair_exact,pair_refine,single_best_gain


def main():
    torch.set_num_threads(4);rng=torch.Generator().manual_seed(32273)
    a=torch.randn(2,6,6,generator=rng,dtype=torch.float64)
    h=a.transpose(1,2)@a+torch.eye(6,dtype=torch.float64)[None]
    x=torch.randn(2,7,6,generator=rng,dtype=torch.float64)
    step=torch.rand(2,7,6,generator=rng,dtype=torch.float64)*.3+.1
    code=(x/step).round().clamp(-7,7);e=code*step-x;g=e@h
    pairs=correlated_pairs(h,100)
    measured=best_pair_move(code,step,g,h,pairs,False)['gain']
    expected=torch.zeros_like(measured)
    max_error=0.
    for b in range(2):
        for row in range(7):
            old=float(e[b,row]@h[b]@e[b,row])
            for i,j in itertools.combinations(range(6),2):
                for si,sj in itertools.product((-1,1),repeat=2):
                    if abs(float(code[b,row,i])+si)>7 or abs(float(code[b,row,j])+sj)>7:continue
                    ne=e[b,row].clone();ne[i]+=si*step[b,row,i];ne[j]+=sj*step[b,row,j]
                    delta=float(ne@h[b]@ne)-old
                    expected[b,row]=min(float(expected[b,row]),delta)
    max_error=float((measured-expected).abs().max())
    assert torch.allclose(measured,expected,atol=1e-12,rtol=1e-11)
    trap_h=torch.tensor([[[1.,.9],[.9,1.]]],dtype=torch.float64)
    target=torch.tensor([[[-.8,.6]]],dtype=torch.float64)
    zero=torch.zeros_like(target);unit=torch.ones_like(target)
    grad=(zero-target)@trap_h
    assert float(single_best_gain(zero,unit,grad,trap_h))>0
    escaped,stats=pair_refine(target,zero,unit,trap_h,correlated_pairs(trap_h,1),2,True)
    assert stats['escaped_stalled']==1 and stats['after_metric']<stats['before_metric']
    assert torch.equal(escaped,torch.tensor([[[-1.,1.]]],dtype=torch.float64))
    changed,random_stats=pair_refine(x,code*step,step,h,pairs,4,False)
    before=((code*step-x)@h*(code*step-x)).sum(-1)
    after=((changed-x)@h*(changed-x)).sum(-1)
    assert bool((after<=before+1e-12).all())
    assert torch.allclose(changed/step,(changed/step).round(),atol=1e-12,rtol=0)
    assert bool(((changed/step).abs()<=7+1e-12).all())
    off,_=pair_refine(x,code*step,step,h,pairs,0)
    assert torch.equal(off,code*step)
    # Exhaust all 225 code pairs to verify the conditional 15-code search.
    measured_exact=best_pair_exact(code[:1,:2],step[:1,:2],g[:1,:2],h[:1],tuple(p[:1] for p in pairs))['gain']
    expected_exact=torch.zeros_like(measured_exact)
    for row in range(2):
        old=float(e[0,row]@h[0]@e[0,row])
        for i,j in itertools.combinations(range(6),2):
            for ni,nj in itertools.product(range(-7,8),repeat=2):
                ne=e[0,row].clone()
                ne[i]+=(ni-code[0,row,i])*step[0,row,i]
                ne[j]+=(nj-code[0,row,j])*step[0,row,j]
                expected_exact[0,row]=min(float(expected_exact[0,row]),float(ne@h[0]@ne)-old)
    exact_error=float((expected_exact-measured_exact).abs().max())
    assert torch.allclose(expected_exact,measured_exact,atol=1e-11,rtol=1e-11)
    unequal_h=torch.tensor([[[1.,.95],[.95,1.]]],dtype=torch.float64)
    unequal_x=torch.tensor([[[-2.,2.]]],dtype=torch.float64)
    unequal_step=torch.tensor([[[.25,1.]]],dtype=torch.float64)
    zero=torch.zeros_like(unequal_x);pairs2=correlated_pairs(unequal_h,1)
    one,_=pair_refine(unequal_x,zero,unequal_step,unequal_h,pairs2,1,True)
    assert torch.equal(one,zero)
    jump,jump_stats=pair_refine(unequal_x,zero,unequal_step,unequal_h,pairs2,1,False,exact=True)
    assert jump_stats['after_metric']<jump_stats['before_metric']
    assert torch.equal(jump/unequal_step,torch.tensor([[[-7.,2.]]],dtype=torch.float64))
    # Zero/negative conditional curvature: the minimum is at an endpoint.
    endpoint_error=0.
    for curvature in (0.,-1.):
        edge_h=torch.tensor([[[1.,.3],[.3,curvature]]],dtype=torch.float64)
        edge_x=torch.tensor([[[.4,-.2]]],dtype=torch.float64)
        edge_code=torch.zeros_like(edge_x);edge_step=torch.ones_like(edge_x)
        edge_e=-edge_x;edge_g=edge_e@edge_h
        edge_pairs=(torch.tensor([[0]]),torch.tensor([[1]]))
        observed=float(best_pair_exact(edge_code,edge_step,edge_g,edge_h,edge_pairs)['gain'])
        old=float((edge_e@edge_h*edge_e).sum())
        expected=0.
        for ni,nj in itertools.product(range(-7,8),repeat=2):
            ne=torch.tensor([ni,nj],dtype=torch.float64)-edge_x[0,0]
            expected=min(expected,float(ne@edge_h[0]@ne)-old)
        endpoint_error=max(endpoint_error,abs(observed-expected))
    assert endpoint_error<1e-11
    try:
        correlated_pairs(torch.ones(1,1,1),1)
    except ValueError:
        pass
    else:
        raise AssertionError('One-channel input must not generate an empty pair pool')
    report=dict(passed=True,exhaustive_gain_error=max_error,exact_block_gain_error=exact_error,multi_code_trap=jump_stats,trap=stats,random=random_stats,
                row_monotone=True,legal_lattice=True,zero_rounds_exact=True,endpoint_gain_error=endpoint_error)
    (Path(__file__).resolve().parents[1]/'docs/attention-pair-math-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
