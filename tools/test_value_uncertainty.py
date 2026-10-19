"""Mechanism checks for v400, not a local score predictor."""
import ast
import copy
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params
torch.set_num_threads(4);torch.manual_seed(400)
m=load_solution(ROOT/'solution/solution_v400.py')
s=(ROOT/'solution/solution_v400.py').read_text()
assert s.startswith((ROOT/'solution/solution_v387.py').read_text())
for node in ast.walk(ast.parse(s)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
x=torch.randn(129,64)
state=dict(center=torch.zeros(64),projection=torch.randn(64,3)*.1,kernel=torch.randn(2,4,4)*.1,
           query_sigma_modes=torch.randn(2,2,4)*.3,active=torch.ones(1,dtype=torch.bool))
snapshot=copy.deepcopy(state);rng=torch.random.get_rng_state().clone()
f=m._vk_features(x,state)
points=torch.cat((state['query_sigma_modes'],-state['query_sigma_modes']),1)*2**.5
torch.testing.assert_close(points.mean(1),torch.zeros(2,4))
torch.testing.assert_close(points.transpose(1,2)@points/4,state['query_sigma_modes'].transpose(1,2)@state['query_sigma_modes'])
probs=m._vk_probabilities(x,state,1)
assert torch.equal(rng,torch.random.get_rng_state())
assert probs.shape==(1,32,129) and bool(torch.isfinite(probs).all())
torch.testing.assert_close(probs.sum(-1),torch.ones(1,32))
e=torch.randn(129,64)
pe=(probs@e).reshape(2,4,4,64)
lhs=pe.square().mean(2)
rhs=pe.mean(2).square()+(pe-pe.mean(2,keepdim=True)).square().mean(2)
torch.testing.assert_close(lhs,rhs)
zero=dict(state,query_sigma_modes=torch.zeros_like(state['query_sigma_modes']))
assert torch.equal(m._vk_probabilities(x,zero,1),m._vu_parent_probabilities(x,zero,1))
assert all(torch.equal(v,state[k]) for k,v in snapshot.items())
parent=m._dyn_table(x,None,has_scale=False)
rng=torch.random.get_rng_state().clone()
answer=m._vk_refine(x,parent,state,1,64)
assert not validate_hif4_params(answer,x.shape,'sigma')
assert torch.equal(rng,torch.random.get_rng_state())
changed=int((m._deq_params(answer)!=m._deq_params(parent)).sum())
samples=[{r:quantize_nvfp4(torch.randn(65,128 if r=='q' else 64)) for r in 'qkv'} for _ in range(4)]
fitted=m._vk_fit(samples,2,1,64)
assert 'query_sigma_modes' in fitted and fitted['query_sigma_modes'].shape==(2,2,fitted['projection'].shape[1]+1)
assert bool((fitted['query_sigma_modes'][:,:,-1]==0).all())
other=samples[:-1]+[{r:quantize_nvfp4(torch.randn(65,128 if r=='q' else 64)*10) for r in 'qkv'}]
again=m._vk_fit(other,2,1,64)
assert all(torch.equal(v,again[k]) if torch.is_tensor(v) else v==again[k] for k,v in fitted.items())
assert changed>0
print('PASS: parent prefix/imports, sigma covariance, probability normalization, risk variance decomposition, zero fallback, state/RNG, legal packed refinement, last-sequence exclusion; changed',changed)
