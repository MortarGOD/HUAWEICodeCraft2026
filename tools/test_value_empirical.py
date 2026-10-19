"""v401 empirical risk, representative masses and submission invariants."""
import ast
import copy
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params
torch.set_num_threads(4);torch.manual_seed(401)
m=load_solution(ROOT/'solution/solution_v401.py')
s=(ROOT/'solution/solution_v401.py').read_text()
assert s.startswith((ROOT/'solution/solution_v400.py').read_text())
for node in ast.walk(ast.parse(s)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
for n in (1,3,41,129):
    coeff=torch.randn(2,n,9)+.7;coeff[:,:,-1]=0
    centers,mass=m._ve_centroids(coeff,torch.rand(9))
    torch.testing.assert_close(mass.sum(1),torch.ones(2))
    torch.testing.assert_close((centers*mass[:,:,None]).sum(1),coeff.mean(1),rtol=1e-5,atol=1e-6)
    assert bool((mass>=0).all())
identical=torch.ones(2,23,5)
centers,mass=m._ve_centroids(identical,torch.ones(5))
torch.testing.assert_close((centers*mass[:,:,None]).sum(1),torch.ones(2,5))
x=torch.randn(129,128)
state=dict(center=torch.zeros(128),projection=torch.randn(128,3)*.1,kernel=torch.randn(4,4,4)*.1,
           query_residual_centers=torch.randn(4,8,4)*.3,query_residual_mass=torch.softmax(torch.randn(4,8),1),active=torch.ones(2,dtype=torch.bool))
snapshot=copy.deepcopy(state);rng=torch.random.get_rng_state().clone()
factors=m._vk_probabilities(x,state,2)
assert torch.equal(rng,torch.random.get_rng_state())
assert all(torch.equal(v,state[k]) for k,v in snapshot.items())
f=m._vk_features(x,state);ix=torch.linspace(0,128,4).long()
rawp=torch.softmax(((f[ix][None]@state['kernel'])@f.T)[:,:,None,:]+(state['query_residual_centers']@f.T)[:,None,:,:],-1)
e=torch.randn(2,129,64)
lhs=(factors@e).square().mean(1)
pe=rawp@e.repeat_interleave(2,0)[:,None,:,:]
rhs=(pe.square()*state['query_residual_mass'][:,None,:,None]).sum(2).mean(1).reshape(2,2,64).mean(1)
torch.testing.assert_close(lhs,rhs)
parent=m._dyn_table(x,None,has_scale=False)
answer=m._vk_refine(x,parent,state,2,64)
assert not validate_hif4_params(answer,x.shape,'empirical')
partial=m._vk_refine(x,parent,dict(state,active=torch.tensor([True,False])),2,64)
for key in parent:
    assert torch.equal(partial[key][:,0],answer[key][:,0])
    if key not in ('sign','mant'):assert torch.equal(partial[key][:,1],parent[key][:,1])
assert torch.equal(m._deq_params(partial)[:,64:],m._deq_params(parent)[:,64:])
changed=int((m._deq_params(answer)!=m._deq_params(parent)).sum());assert changed>0
samples=[{r:quantize_nvfp4(torch.randn(65,128 if r=='q' else 64)) for r in 'qkv'} for _ in range(4)]
fitted=m._vk_fit(samples,2,1,64)
assert 'query_residual_centers' in fitted and bool((fitted['query_residual_centers'][:,:,-1]==0).all())
other=samples[:-1]+[{r:quantize_nvfp4(torch.randn(65,128 if r=='q' else 64)*10) for r in 'qkv'}]
again=m._vk_fit(other,2,1,64)
assert all(torch.equal(v,again[k]) if torch.is_tensor(v) else v==again[k] for k,v in fitted.items())
print('PASS: parent/imports, empirical mean and mass, empty clusters, weighted Gram identity, state/RNG, legal mixed heads, no last-sequence fit; changed',changed)
