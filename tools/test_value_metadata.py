"""Format, preservation and objective checks for changing complete scale trees."""
import argparse
import json
from pathlib import Path
import torch
from probe_value_metadata import configured
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params,validate_frozen_state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bank',action='store_true')
    parser.add_argument('--row',action='store_true')
    parser.add_argument('--cap',type=float,default=1.01)
    parser.add_argument('--output',default='docs/value-metadata-invariants.json')
    args=parser.parse_args()
    m=configured(1,True,args.cap,args.bank,args.row);torch.set_num_threads(4)
    generator=torch.Generator().manual_seed(31873)
    kh,d=2,64;checks=[]
    for rows in (1,17,129,1024,4097):
        pair=quantize_nvfp4(torch.randn(rows,kh*d,generator=generator)*.3+.02)
        x=m.dequantize_nvfp4(*pair).float()
        parent=m._vm_parent_dynamic(*pair,kh,d,None)
        saved={k:v.clone() for k,v in parent.items()}
        candidate=m._vm_proposal(x,parent,1.)
        assert not validate_hif4_params(candidate,x.shape,'metadata')
        assert all(torch.equal(parent[k],saved[k]) for k in parent)
        e0=x-m._deq_params(parent);e1=x-m._deq_params(candidate)
        mse0=e0.square().mean(0);mse1=e1.square().mean(0)
        c0=(e0.mean(0).square()+mse0/rows).reshape(-1,64).sum(1)
        c1=(e1.mean(0).square()+mse1/rows).reshape(-1,64).sum(1)
        assert bool((c1<=c0+1e-12).all())
        assert bool((mse1<=mse0*(args.cap+1e-5)+1e-20).all())
        state=dict(vm_active=torch.tensor([True,False]),vm_variance=1.,fallback=None)
        assert not validate_frozen_state(state,'state')
        mixed=m.hif4_dynamic_quantize_v(*pair,kh,d,state)
        for key in parent:
            assert torch.equal(mixed[key][:,1:],parent[key][:,1:])
            if rows==1 or rows>4096:assert torch.equal(mixed[key],parent[key])
        off=dict(state,vm_active=torch.zeros(kh,dtype=torch.bool))
        disabled=m.hif4_dynamic_quantize_v(*pair,kh,d,off)
        assert all(torch.equal(disabled[k],parent[k]) for k in parent)
        checks.append(dict(rows=rows,valid=True,changed_elements=int((e1!=e0).sum())))
    # Verify the row-local quadratic difference independently in float64.
    err=torch.randn(9,128,generator=generator,dtype=torch.float64)
    delta=torch.randn(128,generator=generator,dtype=torch.float64)*.03
    factor=8.
    shifted=err.clone();shifted[3]-=delta
    objective=lambda e:(e.mean(0).square()+factor/e.shape[0]*e.square().mean(0)).reshape(-1,64).sum(1)
    exact=(objective(shifted)-objective(err))*81
    analytic=((1+factor)*delta.square()-2*(err.sum(0)+factor*err[3])*delta).reshape(-1,64).sum(1)
    delta_error=float((exact-analytic).abs().max())
    assert torch.allclose(exact,analytic,atol=1e-11,rtol=1e-11)
    # Exercise adoption as well as rejection: the poor parent is still a legal
    # HiF4 result but leaves ample room for a new tree to improve the objective.
    poor={k:v.clone() for k,v in parent.items()}
    poor['mant'].zero_()
    probe_x=x[:129].contiguous()
    poor={k:v[:129].clone() for k,v in poor.items()}
    adopted=m._vm_proposal(probe_x,poor,1.)
    assert not validate_hif4_params(adopted,probe_x.shape,'adopted')
    assert bool((m._deq_params(adopted)!=m._deq_params(poor)).any())
    assert float((probe_x-m._deq_params(adopted)).square().mean())<float(probe_x.square().mean())
    result=dict(passed=True,bank=args.bank,row=args.row,cap=args.cap,checks=checks,row_delta_max_error=delta_error,adoption_exercised=True,objective_monotone=True,per_channel_mse_cap=True,parent_immutable=True,
                inactive_head_exact=True,disabled_exact=True,state_valid=True)
    (Path(__file__).resolve().parents[1]/args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
