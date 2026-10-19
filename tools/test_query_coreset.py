"""Rejected centroid prototype: coreset identities and quantizer invariants."""
import ast,copy,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params
torch.set_num_threads(4);torch.manual_seed(403)
m=load_solution(ROOT/'solution/solution_v402.py')
exec((ROOT/'tools/query_coreset_core.py').read_text(),m.__dict__)
s=(ROOT/'solution/solution_v402.py').read_text()
assert s.startswith((ROOT/'solution/solution_v400.py').read_text())
for node in ast.walk(ast.parse(s)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
f=torch.randn(65,7);kernel=torch.randn(4,7,7)*.1
c=f-f.mean(0);cov=c.T@c/len(f);metric=(kernel@cov@kernel.transpose(-1,-2)).mean(0)
delta=torch.randn(7)
torch.testing.assert_close(delta@metric@delta,((delta[None,None]@kernel)@c.T).square().mean())
centers,mass=m._cq_representatives(f,kernel)
torch.testing.assert_close(mass.sum(),torch.tensor(1.))
torch.testing.assert_close((centers*mass[:,None]).sum(0),f.mean(0),atol=1e-6,rtol=1e-5)
# Four repeated feature classes: four mass-weighted prototypes recover all-query risk.
xy=torch.tensor([[-2.,-1.],[1.,3.],[3.,-2.],[-3.,2.]])
x=xy.repeat_interleave(torch.tensor([40,12,8,4]),0)
state=dict(center=torch.zeros(2),projection=torch.eye(2),kernel=torch.eye(3)[None].repeat(2,1,1)*.35,active=torch.ones(1,dtype=torch.bool))
features=m._vk_features(x,state);e=torch.randn(64,5)
full=torch.softmax((features[None]@state['kernel'])@features.T,-1)
truth=(full@e).square().mean((0,1))
new=(m._vk_probabilities(x,state,1)@e).square().mean((0,1))
old=(m._cq_parent_probabilities(x,state,1)@e).square().mean((0,1))
torch.testing.assert_close(new,truth,atol=1e-6,rtol=1e-5)
assert float((old-truth).abs().max())>1e-4
perm=torch.randperm(64)
torch.testing.assert_close((m._vk_probabilities(x[perm],state,1)@e[perm]).square().mean((0,1)),truth,atol=1e-6,rtol=1e-5)
x=torch.randn(129,128)
state=dict(center=torch.zeros(128),projection=torch.randn(128,4)*.1,kernel=torch.randn(4,5,5)*.1,
           query_sigma_modes=torch.randn(4,2,5)*.2,active=torch.ones(2,dtype=torch.bool))
snapshot=copy.deepcopy(state);rng=torch.random.get_rng_state().clone()
factors=m._vk_probabilities(x,state,2)
assert factors.shape==(2,32,129)
assert torch.equal(rng,torch.random.get_rng_state()) and all(torch.equal(v,state[k]) for k,v in snapshot.items())
f=m._vk_features(x,state);centers,mass=m._cq_representatives(f,state['kernel'])
logits=(centers[None]@state['kernel'])@f.T;offset=state['query_sigma_modes']@f.T*2**.5
p=torch.softmax(logits[:,:,None]+torch.cat((offset,-offset),1)[:,None],-1)
error=torch.randn(2,129,64)
rhs=((p@error.repeat_interleave(2,0)[:,None]).square()*mass[None,:,None,None]).sum(1).mean(1).reshape(2,2,64).mean(1)
torch.testing.assert_close((factors@error).square().mean(1),rhs)
parent=m._dyn_table(x,None,has_scale=False)
answer=m._vk_refine(x,parent,state,2,64)
assert not validate_hif4_params(answer,x.shape,'coreset')
partial=m._vk_refine(x,parent,dict(state,active=torch.tensor([True,False])),2,64)
assert torch.equal(m._deq_params(partial)[:,:64],m._deq_params(answer)[:,:64])
assert torch.equal(m._deq_params(partial)[:,64:],m._deq_params(parent)[:,64:])
assert torch.equal(m._vk_probabilities(x[:4],state,2),m._cq_parent_probabilities(x[:4],state,2))
changed=int((m._deq_params(answer)!=m._deq_params(parent)).sum());assert changed>0
print('PASS: logit-variance metric, weighted mean/mass, four-class full-risk and permutation identity, sigma weighted risk, state/RNG, legal mixed heads, small-T fallback, parent/imports; changed',changed)
