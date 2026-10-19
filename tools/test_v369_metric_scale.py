"""Full-head Gram selection and legal metadata, independent of model scores."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params


def main():
    torch.manual_seed(369);torch.set_num_threads(4)
    m=load_solution(ROOT/'solution/solution_v369.py')
    for d in (64,128,256):
        kh,qh,n=2,4,17
        a=torch.randn(kh,d,d)
        h=a@a.transpose(-1,-2)/d+torch.eye(d)*.1
        st={'kvh':kh,'gq':1,'rf':0,'rot':0,'H':h,'u':m._upper_cholesky_inv(h)}
        x=torch.randn(n,qh*d)*torch.linspace(.2,4,qh*d)
        p=m._dyn_table(x,None,False)
        out=m._hs_trial(x,st,qh,d,p,n)
        assert not validate_hif4_params(out,x.shape,'query')
        hf=h.repeat_interleave(qh//kh,0)
        def cost(params):
            e=(m._deq_params(params)-x).reshape(n,qh,d).transpose(0,1)
            return ((e@hf)*e).sum(-1)
        assert bool((cost(out)<=cost(p)+1e-5).all())
        inactive=m._hs_trial(x,dict(st,hs_active=torch.zeros(kh,dtype=torch.bool)),qh,d,p,n)
        assert all(torch.equal(p[k],inactive[k]) for k in p)
        print(d,'PASS Gram non-increase, legal full-block selection, inactive exact fallback')


if __name__=='__main__':main()
