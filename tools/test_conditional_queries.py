"""Calibration isolation, fused-probability layout and dynamic boundaries."""
import copy
import argparse
import json
from pathlib import Path
import torch
from probe_conditional_queries import configured
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--metadata',action='store_true')
    parser.add_argument('--output',default='docs/conditional-query-invariants.json')
    args=parser.parse_args()
    torch.set_num_threads(4);m=configured(8,metadata=args.metadata)
    rng=torch.Generator().manual_seed(31991)
    qh,kh,d=4,2,64;samples=[]
    for rows in (64,96,129):
        v=torch.randn(rows,kh*d,generator=rng)
        samples.append(dict(v=quantize_nvfp4(v),k=quantize_nvfp4(v*.8+.03),
                            q=quantize_nvfp4(torch.randn(rows,qh*d,generator=rng))))
    saved=copy.deepcopy(samples)
    state,predictor=m._cq_fit(samples,qh,kh,d)
    changed=copy.deepcopy(samples)
    for role in 'qkv':changed[-1][role]=quantize_nvfp4(torch.randn(129,(qh if role=='q' else kh)*d,generator=rng)*2)
    again,_=m._cq_fit(changed,qh,kh,d)
    for field in ('anchors','center','scale','offset'):
        assert torch.equal(state['conditional_queries'][field],again['conditional_queries'][field])
    state.update(active=torch.tensor([True,False]),fallback=None)
    assert not validate_frozen_state(state,'conditional')
    records=[]
    for rows in (1,129,1024,4097):
        pair=quantize_nvfp4(torch.randn(rows,kh*d,generator=rng))
        parent=m._cq_parent_dynamic(*pair,kh,d,None)
        result=m.hif4_dynamic_quantize_v(*pair,kh,d,state)
        assert not validate_hif4_params(result,pair[0].shape,'conditional')
        for key in parent:
            assert torch.equal(result[key][:,1:],parent[key][:,1:])
            if rows==1 or rows>4096:assert torch.equal(result[key],parent[key])
        disabled=m.hif4_dynamic_quantize_v(*pair,kh,d,dict(state,active=torch.zeros(kh,dtype=torch.bool)))
        assert all(torch.equal(disabled[key],parent[key]) for key in parent)
        x=m.dequantize_nvfp4(*pair).float()
        mse0=(x-m._deq_params(parent)).square().mean(0)
        mse1=(x-m._deq_params(result)).square().mean(0)
        cap=1.10001 if args.metadata else 1.05001
        assert bool((mse1<=mse0*cap+1e-20).all())
        if rows<=4096:
            probs=m._vk_probabilities(x,state,kh)
            def cost(p):
                error=(x-m._deq_params(p)).reshape(rows,kh,d).transpose(0,1)
                return (probs@error).square().mean(1)+error.square().mean(1)/rows
            assert bool((cost(result).sum(1)<=cost(parent).sum(1)+1e-8).all())
        records.append(dict(rows=rows,valid=True,changed_elements=int((m._deq_params(parent)!=m._deq_params(result)).sum())))
    for current,original in zip(samples,saved):
        for r in 'qkv':assert all(torch.equal(a,b) for a,b in zip(current[r],original[r]))
    e=torch.randn(2,9,64,generator=rng,dtype=torch.float64)
    delta=torch.randn(2,64,generator=rng,dtype=torch.float64)*.03
    p=torch.softmax(torch.randn(2,7,9,generator=rng,dtype=torch.float64),-1)
    penalty=1/9
    loss=lambda error:((p@error).square().mean(1)+penalty*error.square().mean(1)).sum(1)
    changed=e.clone();changed[:,3]-=delta
    gradient=(p.transpose(1,2)@(p@e))/7+penalty/9*e
    curvature=p.square().mean(1)+penalty/9
    local=(curvature[:,3,None]*delta.square()-2*gradient[:,3]*delta).sum(1)
    exact=loss(changed)-loss(e)
    delta_error=float((local-exact).abs().max())
    assert torch.allclose(local,exact,atol=1e-12,rtol=1e-11)
    report=dict(passed=True,metadata=args.metadata,checks=records,row_delta_error=delta_error,proxy_monotone=True,holdout_all_roles_excluded=True,calibration_immutable=True,
                state_valid=True,inactive_exact=True,disabled_exact=True,mse_cap=True)
    (Path(__file__).resolve().parents[1]/args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
