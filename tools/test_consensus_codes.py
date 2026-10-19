"""Verify exact fixed-code descent and metadata/parent preservation."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params


def main():
    torch.set_num_threads(4)
    m=load_solution(ROOT/'solution/solution_v373.py')
    rng=torch.Generator().manual_seed(373)
    changes=0
    for width in (64,128,256):
        for noise in (0.,.02,.5):
            weight=torch.randn(64,width,generator=rng)*.1
            params=m._quantize_linear_values(weight)
            saved={k:v.clone() for k,v in params.items()}
            b=m._dequantize_linear_params(params).float()
            zs=[torch.randn(n,width,generator=rng) for n in (1,10,32,64)]
            ys=[z@weight.T+noise*torch.randn(len(z),64,generator=rng) for z in zs]
            new=m._dc_refine(zs,ys,params)
            qb=m._dequantize_linear_params(new).float()
            for fold in range(2):
                old=sum((ys[i]-zs[i]@b.T).double().square().sum() for i in range(fold,4,2))
                now=sum((ys[i]-zs[i]@qb.T).double().square().sum() for i in range(fold,4,2))
                assert now<=old+1.e-5*old
            assert all(torch.equal(v,saved[k]) for k,v in params.items())
            for k in ('scale_factor','scale_lv2','scale_lv3'):assert torch.equal(new[k],params[k])
            before=params['sign'].float()*params['mant'].float()*4
            after=new['sign'].float()*new['mant'].float()*4
            assert (before-after).abs().max()<=1
            assert not validate_hif4_params(new,(64,width),'consensus')
            changes+=m.DG_DEBUG['changed_weight_codes']
    assert changes>0
    print('PASS: nine cases, both-fold descent, immutable parent/scales, one-code bounds; changes',changes)


if __name__=='__main__':main()
