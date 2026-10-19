"""Low-rank solve and actual full-scale V candidate invariants."""
import ast
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params
torch.set_num_threads(4);torch.manual_seed(399)
m=load_solution(ROOT/'solution/solution_v399.py')
source=(ROOT/'solution/solution_v399.py').read_text()
assert source.startswith((ROOT/'solution/solution_v387.py').read_text())
for node in ast.walk(ast.parse(source)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
for t in (7,31,129):
    p=torch.softmax(torch.randn(2,5,t,dtype=torch.float64),-1)
    rho=torch.rand(2,dtype=torch.float64)*.1+.01;lam=1./t**2
    rhs=torch.randn(2,t,8,dtype=torch.float64)
    expected=torch.linalg.solve((rho+lam)[:,None,None]*torch.eye(t)+p.transpose(1,2)@p/5,rhs)
    torch.testing.assert_close(m._ad_inverse(rhs,p,rho,lam),expected,rtol=1e-10,atol=1e-10)
changed=0;tree_changes=0
for t,d in ((17,64),(129,128),(513,256)):
    kh=2;data=quantize_nvfp4(torch.randn(t,kh*d));x=m.dequantize_nvfp4(*data).float()
    parent=m._dyn_table(x,None,has_scale=False)
    prob=torch.softmax(torch.randn(kh,8,t)*2,-1)
    active=torch.ones(kh,dtype=torch.bool)
    answer=m._ad_quantize(x,parent,prob,active,kh,d)
    assert not validate_hif4_params(answer,data[0].shape,'admm')
    def stats(p):
        e=(x-m._deq_params(p)).reshape(t,kh,d).transpose(0,1)
        mse=e.square().mean(1)
        cost=((prob@e).square().mean(1)+mse/t).reshape(kh,d//64,64).sum(-1)
        return cost,mse
    before,em=stats(parent);after,nm=stats(answer)
    assert bool((after<=before+1e-6).all())
    assert bool((nm<=em*m._VK_MSE_CAP+1e-7).all())
    partial=m._ad_quantize(x,parent,prob,torch.tensor([True,False]),kh,d)
    for key in partial:
        assert torch.equal(partial[key][:,:d//64],answer[key][:,:d//64])
        assert torch.equal(partial[key][:,d//64:],parent[key][:,d//64:])
    changed+=int((m._deq_params(answer)!=m._deq_params(parent)).sum())
    for key in ('scale_factor','scale_lv2','scale_lv3'):
        if key in parent:tree_changes+=int((answer[key]!=parent[key]).sum())
    snapshot={k:v.clone() for k,v in parent.items()};rng=torch.random.get_rng_state().clone()
    m._ad_quantize(x,parent,prob,active,kh,d)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(v,parent[k]) for k,v in snapshot.items())
assert changed>0 and tree_changes>0,(changed,tree_changes)
assert m._vk_refine(torch.empty(1,128),parent,{'active':torch.ones(2,dtype=torch.bool)},2,64) is parent
print('PASS: Woodbury/direct solve, global block objective, column MSE cap, legal full trees, inactive exactness, state/RNG, parent prefix; changed',changed,'metadata',tree_changes)
