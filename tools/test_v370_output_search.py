"""Verify exact prototype objective and inactive full-output search behavior."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params


def main():
    torch.manual_seed(370);torch.set_num_threads(4)
    m=load_solution(ROOT/'solution/solution_v370.py')
    for d in (64,128,256):
        qh,kh,n,t=4,2,19,32
        x=torch.randn(n,qh*d)
        p=m._dyn_table(x,None,False)
        k=torch.randn(kh,t,d)/d**.5;v=torch.randn(kh,t,d)
        bank=dict(raw=k,key=k+.03*torch.randn_like(k),raw_value=v,
                  value=v+.05*torch.randn_like(v),active=torch.ones(qh,dtype=torch.bool),rounds=4)
        out=m._nq_refine(x,p,bank,qh,d)
        assert not validate_hif4_params(out,x.shape,'query')
        goal=torch.softmax(x.reshape(n,qh,d).transpose(0,1)@k.repeat_interleave(2,0).transpose(1,2),-1)@v.repeat_interleave(2,0)
        def loss(params):
            z=m._deq_params(params).reshape(n,qh,d).transpose(0,1)
            pr=torch.softmax(z@bank['key'].repeat_interleave(2,0).transpose(1,2),-1)
            return (pr@bank['value'].repeat_interleave(2,0)-goal).square().sum(-1)
        assert bool((loss(out)<=loss(p)+1e-6).all())
        fallback=m._nq_refine(x,p,dict(bank,active=torch.zeros(qh,dtype=torch.bool)),qh,d)
        assert all(torch.equal(p[key],fallback[key]) for key in p)
        print(d,'PASS exact full-output objective non-increase and inactive equality')


if __name__=='__main__':main()
