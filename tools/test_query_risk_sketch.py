"""v403 signed Gram sketch and V solver invariants."""
import ast,copy,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params
torch.set_num_threads(4);torch.manual_seed(403)
m=load_solution(ROOT/'solution/solution_v403.py')
s=(ROOT/'solution/solution_v403.py').read_text();assert s.startswith((ROOT/'solution/solution_v402.py').read_text())
for node in ast.walk(ast.parse(s)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
for rows,keep,t in ((16,16,33),(64,16,129),(16,4,65)):
    p=torch.softmax(torch.randn(2,rows,t,dtype=torch.float64),-1)
    f=m._qr_compress(p,keep)
    assert f.shape==(2,keep,t)
    h=p.transpose(-1,-2)@p/rows;sketch=f.transpose(-1,-2)@f/keep
    tail=torch.linalg.eigvalsh(h-sketch)
    assert float(tail.min())>-1e-12
    if rows==keep:torch.testing.assert_close(h,sketch)
    # The maximal residual eigenvalue is the first discarded singular energy / rows.
    bound=torch.linalg.eigvalsh(p@p.transpose(-1,-2))[:,-keep-1]/rows if keep<rows else torch.zeros(2,dtype=p.dtype)
    torch.testing.assert_close(tail[:,-1],bound,atol=1e-12,rtol=1e-8)
    rotation=torch.linalg.qr(torch.randn(2,keep,keep,dtype=torch.float64)).Q
    g=rotation@f
    torch.testing.assert_close(f.square().sum(1),g.square().sum(1),atol=1e-12,rtol=1e-8)
x=torch.randn(129,128)
state=dict(center=torch.zeros(128),projection=torch.randn(128,4)*.1,kernel=torch.randn(4,5,5)*.1,
           query_sigma_modes=torch.randn(4,2,5)*.2,active=torch.ones(2,dtype=torch.bool))
snapshot=copy.deepcopy(state);rng=torch.random.get_rng_state().clone()
factors=m._vk_probabilities(x,state,2)
assert factors.shape==(2,32,129) and bool(torch.isfinite(factors).all())
assert torch.equal(rng,torch.random.get_rng_state()) and all(torch.equal(v,state[k]) for k,v in snapshot.items())
parent=m._dyn_table(x,None,has_scale=False)
answer=m._vk_refine(x,parent,state,2,64)
assert not validate_hif4_params(answer,x.shape,'sketch')
def cost(p):
    e=(x-m._deq_params(p)).reshape(129,2,64).transpose(0,1)
    return (factors@e).square().mean(1)+e.square().mean(1)/129,e.square().mean(1)
before,bm=cost(parent);after,am=cost(answer)
assert bool((after<=before+1e-7).all()) and bool((am<=bm*m._VK_MSE_CAP+1e-7).all())
partial=m._vk_refine(x,parent,dict(state,active=torch.tensor([True,False])),2,64)
assert torch.equal(m._deq_params(partial)[:,:64],m._deq_params(answer)[:,:64])
assert torch.equal(m._deq_params(partial)[:,64:],m._deq_params(parent)[:,64:])
assert torch.equal(m._vk_probabilities(x[:4],state,2),m._qr_parent_probabilities(x[:4],state,2))
changed=int((m._deq_params(answer)!=m._deq_params(parent)).sum());assert changed>0
print('PASS: parent/imports, Gram scaling, PSD tail bound, full-rank identity, rotation-invariant denominator, fixed solver width, packed cost/MSE, mixed heads, state/RNG, small-T probability fallback; changed',changed)
