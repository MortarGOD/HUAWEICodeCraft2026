"""One-pass merged metric equals duplicated equal-metric v374 refinement."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params


def main():
    torch.set_num_threads(4)
    old=load_solution(ROOT/'solution/solution_v374.py')
    new=load_solution(ROOT/'solution/solution_v375.py')
    rng=torch.Generator().manual_seed(375)
    for d in (64,128,256):
        for n in (16,129):
            kh,qh=2,4
            raw=torch.randn(n,qh*d,generator=rng)
            mat=torch.randn(kh,d,d,generator=rng)
            H=mat@mat.transpose(-1,-2)+torch.eye(d)[None]
            H/=H.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]
            centers=torch.randn(kh,2,d,generator=rng)
            st={'kvh':kh,'cm_centers':centers,'cm_metrics':H[:,None].expand(kh,2,d,d).contiguous(),
                'cm_active':torch.tensor([True,False])}
            params=old._dyn_table(raw,None,has_scale=False)
            a=old._cm_trial(raw,st,qh,d,params);b=new._cm_trial(raw,st,qh,d,params)
            assert all(torch.equal(a[k],b[k]) for k in a)
            assert not validate_hif4_params(b,raw.shape,'merged')
    print('PASS: six shapes, bit-exact one-pass vs duplicated merged metrics, legal parameters')


if __name__=='__main__':main()
