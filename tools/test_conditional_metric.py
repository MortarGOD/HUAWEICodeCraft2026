"""Per-query selected-metric descent, head masking and parameter legality."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params


def main():
    torch.set_num_threads(4)
    m=load_solution(ROOT/'solution/solution_v374.py')
    rng=torch.Generator().manual_seed(374)
    changed=0
    for d in (64,128,256):
        kh,qh,n=2,4,16
        x=torch.randn(n,qh*d,generator=rng)
        centers=torch.randn(kh,2,d,generator=rng)
        centers/=centers.norm(dim=-1,keepdim=True)
        basis=torch.randn(kh,2,d,d,generator=rng)
        bank=basis@basis.transpose(-1,-2)+torch.eye(d)[None,None]
        bank/=bank.diagonal(dim1=-2,dim2=-1).mean(-1)[:,:,None,None]
        st={'kvh':kh,'cm_centers':centers,'cm_metrics':bank,'cm_active':torch.tensor([True,False])}
        parent=m._dyn_table(x,None,has_scale=False)
        out=m._cm_trial(x,st,qh,d,parent)
        assert not validate_hif4_params(out,x.shape,'conditional')
        old=m._deq_params(parent).reshape(n,qh,d)
        new=m._deq_params(out).reshape(n,qh,d)
        label=torch.einsum('thd,hcd->thc',x.reshape(n,qh,d),centers.repeat_interleave(2,0)).argmax(-1)
        for t in range(n):
            for h in range(qh):
                H=bank[h//2,label[t,h]]
                e0=old[t,h]-x.reshape(n,qh,d)[t,h]
                e1=new[t,h]-x.reshape(n,qh,d)[t,h]
                assert e1@H@e1<=e0@H@e0+1.e-5
        assert torch.equal(old[:,2:],new[:,2:])
        for key in ('scale_factor','scale_lv2','scale_lv3'):assert torch.equal(parent[key],out[key])
        changed+=int((old!=new).sum())
    assert changed>0
    print('PASS: 192 row-head metrics, inactive heads unchanged, metadata fixed, valid HiF4; changes',changed)


if __name__=='__main__':main()
