"""Check ridge fit and block feedback against constrained quadratic solves."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params


def main():
    torch.set_num_threads(4)
    m=load_solution(ROOT/'solution/solution_v372.py')
    gen=torch.Generator().manual_seed(372)
    for width in (64,128,256):
        z=torch.randn(40,width,generator=gen,dtype=torch.float64)
        y=torch.randn(40,64,generator=gen,dtype=torch.float64)
        b=torch.randn(64,width,generator=gen,dtype=torch.float64)
        p,u=m._dg_fit(z,y,b)
        ridge=.1*(z.T@z).diagonal().mean()
        h=z.T@z+ridge*torch.eye(width,dtype=z.dtype)
        assert torch.allclose(h@p.T,z.T@y+ridge*b.T,atol=1e-10,rtol=1e-10)
        def loss(w):return (z@w.T-y).square().sum()+ridge*(w-b).square().sum()
        assert loss(p)<=loss(b)+1e-9
        if width>64:
            frozen=p[:,:64].round()
            err=p[:,:64]-frozen
            step=torch.linalg.solve_triangular(u[:64,:64].T,err.T,upper=False).T@u[:64,64:]
            candidate=p[:,64:]-step
            gradient=(torch.cat([frozen,candidate],1)-p)@h
            assert gradient[:,64:].abs().max()<1e-9
        q=m._dg_quantize(p.float(),u.float())
        assert not validate_hif4_params(q,(64,width),'block_feedback')
    print('PASS: widths64/128/256 ridge normal equations, constrained feedback, legal HiF4')


if __name__=='__main__':main()
