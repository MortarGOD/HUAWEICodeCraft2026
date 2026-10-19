"""Verify ridge normal equation, shift invariance, identity and fused maps."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution

torch.set_num_threads(4)
m=load_solution(ROOT/'solution/solution_v378.py')
torch.manual_seed(731)
for d in (64,128,256):
    k=torch.randn(2,512,d,dtype=torch.float64)
    decoded=k+.1*torch.randn_like(k)
    a=m._kr_map(k,decoded)
    x=decoded-decoded.mean(1,keepdim=True);y=k-k.mean(1,keepdim=True)
    h=x.transpose(1,2)@x/x.shape[1]
    ridge=.1*h.diagonal(dim1=-2,dim2=-1).mean(-1)
    eye=torch.eye(d,dtype=k.dtype)
    lhs=(h+ridge[:,None,None]*eye)@a.transpose(1,2)
    rhs=x.transpose(1,2)@y/x.shape[1]+ridge[:,None,None]*eye
    torch.testing.assert_close(lhs,rhs,atol=1.e-10,rtol=1.e-10)
    torch.testing.assert_close(m._kr_map(k+3,decoded-5),a,atol=1.e-10,rtol=1.e-10)
    torch.testing.assert_close(m._kr_map(k,k),eye.expand_as(a))
    assert ((x@a.transpose(1,2)-y).square().mean()<(x-y).square().mean())
    raw=torch.randn(19,4*d);matrix=a.float();st={'kvh':2,'rot':1,'rseed':0,'qs':torch.rand(2,d)+.5}
    original=m._kt_input(raw,st,4,d,'q').reshape(19,4,d).transpose(0,1)
    fused=m._kt_input(raw,m._kr_transform(st,matrix,d,2),4,d,'q').reshape(19,4,d).transpose(0,1)
    torch.testing.assert_close(fused,original@matrix.repeat_interleave(2,0),atol=2.e-5,rtol=2.e-5)
print('PASS: three dimensions; normal equations, shifts, identity, training loss and fused transform')
