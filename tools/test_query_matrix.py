import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4)
m=load_solution(ROOT/'solution/solution_v379.py')
torch.manual_seed(53)
for d in (64,128):
    q=torch.randn(4,7,d,dtype=torch.float64,requires_grad=True)
    k=torch.randn(4,23,d,dtype=torch.float64)
    v=torch.randn_like(k);target=torch.randn_like(q)
    loss,g,_=m._qm_loss_gradient(q,k,v,target)
    truth=torch.autograd.grad(loss.sum(),q)[0]
    torch.testing.assert_close(g,truth,atol=1.e-12,rtol=1.e-10)
    x=torch.randn_like(q);matrix=torch.eye(d,dtype=q.dtype).expand(2,-1,-1).clone().requires_grad_()
    value=x@matrix.repeat_interleave(2,0)
    loss,g,_=m._qm_loss_gradient(value,k,v,target)
    truth=torch.autograd.grad(loss.sum()/2,matrix)[0]
    analytic=(x.transpose(1,2)@g).reshape(2,2,d,d).mean(1)
    torch.testing.assert_close(analytic,truth,atol=1.e-12,rtol=1.e-10)
    dq=torch.randn_like(q);eps=1.e-5
    a,b=m._qm_tangent(q.detach(),k,v,dq,target)
    def out(z):return torch.softmax(z@k.transpose(1,2)/d**.5,-1)@v
    tangent=(out(q.detach()+eps*dq)-out(q.detach()-eps*dq))/(2*eps)
    torch.testing.assert_close(b,tangent.square().mean((1,2)),atol=1.e-10,rtol=1.e-7)
    torch.testing.assert_close(a,-(tangent*(out(q.detach())-target)).mean((1,2)),atol=1.e-10,rtol=1.e-7)
print('PASS: output derivative and shared GQA matrix chain rule')
