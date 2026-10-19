"""Dual operator normal equations, packed guard and state contracts."""
import ast, copy, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_hif4_params
torch.set_num_threads(4);torch.manual_seed(405)
m=load_solution(ROOT/'solution/solution_v402.py')
exec(compile((ROOT/'tools/dual_operator_core.py').read_text(),'dual_operator_core.py','exec'),m.__dict__)
x=torch.randn(2,23,8,dtype=torch.float64)
p=torch.softmax(torch.randn(2,7,23,dtype=torch.float64),-1)
a=torch.softmax(torch.randn(2,7,23,dtype=torch.float64),-1)
ridge=torch.tensor([.02,.03],dtype=torch.float64)
answer=m._do_continuous(x,p,a,ridge)
h=a.transpose(1,2)@a+ridge[:,None,None]*torch.eye(23,dtype=torch.float64)
b=a.transpose(1,2)@(p@x)+ridge[:,None,None]*x
torch.testing.assert_close(answer,torch.linalg.solve(h,b),atol=1e-11,rtol=1e-10)
torch.testing.assert_close(m._do_continuous(x,p,p,ridge),x)
n=129;kh=2;d=64;qh=4
x=torch.randn(n,kh*d)
state=dict(center=torch.zeros(kh*d),projection=torch.randn(kh*d,4)*.1,
           raw_kernel=torch.randn(qh,5,5)*.1,modes=torch.randn(qh,2,5)*.1,active=torch.ones(kh,dtype=torch.bool))
state['quant_kernel']=state['raw_kernel']+torch.randn(qh,5,5)*.03
snapshot=copy.deepcopy(state);rng=torch.random.get_rng_state().clone()
base=m._dyn_table(x,None,False);ans=m._do_refine(x,base,state,kh,d)
assert not validate_hif4_params(ans,x.shape,'dual')
raw,quant=m._do_probabilities(x,state,kh)
assert raw.shape==quant.shape==(kh,32,n)
torch.testing.assert_close(raw.sum(-1),torch.ones(kh,32))
torch.testing.assert_close(quant.sum(-1),torch.ones(kh,32))
xh=x.reshape(n,kh,d).transpose(0,1);r=quant.shape[1]
lam=r/(n*n)+.05*quant.square().sum(2).mean(1)
def cost(pp):
    y=m._deq_params(pp).reshape(n,kh,d).transpose(0,1)
    mse=(xh-y).square().mean(1)
    return (raw@xh-quant@y).square().mean(1)+(lam*n/r)[:,None]*mse,mse
c0,m0=cost(base);c1,m1=cost(ans)
assert bool((c1<=c0+1e-7).all()) and bool((m1<=m0*m._DO_MSE_CAP+1e-7).all())
mixed=m._do_refine(x,base,dict(state,active=torch.tensor([True,False])),kh,d)
for k in base:
    assert torch.equal(mixed[k][:,1:],base[k][:,1:])
assert all(torch.equal(state[k],v) for k,v in snapshot.items())
assert torch.equal(torch.random.get_rng_state(),rng)
assert m._do_refine(x[:1],base,state,kh,d) is base
assert m._do_refine(torch.zeros(4097,128),base,state,kh,d) is base
print('PASS: explicit normal equations, equal-operator identity, GQA probability normalization, legal packing, objective/MSE guard, inactive exact parameters, state/RNG and boundaries; changed',int((m._deq_params(ans)!=m._deq_params(base)).sum()))

# Primal/dual PCA subspaces agree without constructing a channels-square Gram.
vc=torch.randn(31,96,dtype=torch.float64);vc-=vc.mean(0)
eig,left=torch.linalg.eigh(vc@vc.T/31)
dual=vc.T@left[:,-12:]/(31*eig[-12:]).sqrt()[None]
_,primal=torch.linalg.eigh(vc.T@vc/31)
torch.testing.assert_close(dual@dual.T,primal[:,-12:]@primal[:,-12:].T,atol=1e-10,rtol=1e-9)

# Last calibration sample is never used to fit either operator.
def quantize_to_nvfp4(x):
    levels=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
    code=(x.abs()[...,None]-levels).abs().argmin(-1)
    return levels[code]*x.sign(),torch.ones(x.shape[0],x.shape[1]//16)
samples=[]
for n in (32,40,48):
    smp={}
    latent=torch.randn(n,128)
    for role in 'qkv':
        smp[role]=quantize_to_nvfp4(latent+torch.randn_like(latent)*.05)
    samples.append(smp)
savedq,savedk=m._joint_quant_queries,m.hif4_dynamic_quantize_k
m._joint_quant_queries=lambda s,st,qh,kh,d,ix: m._deq_params(m._dyn_table(m.dequantize_nvfp4(s['q'][0][ix],s['q'][1][ix]).float(),None,False))
m.hif4_dynamic_quantize_k=lambda q,s,kh,d,st:m._dyn_table(m.dequantize_nvfp4(q,s).float(),None,False)
fit1=m._do_fit(samples,dict(q_state={},k_state={}),2,2,64)
changed_samples=copy.deepcopy(samples);changed_samples[-1]={r:quantize_to_nvfp4(torch.randn(48,128)*10) for r in 'qkv'}
fit2=m._do_fit(changed_samples,dict(q_state={},k_state={}),2,2,64)
assert all(torch.equal(v,fit2[k]) for k,v in fit1.items())
assert all(v.device.type=='cpu' and not v.requires_grad and bool(torch.isfinite(v).all()) for v in fit1.values())
m._joint_quant_queries,m.hif4_dynamic_quantize_k=savedq,savedk
print('PASS: dual PCA equivalence, high-width fit, holdout excluded from both operator fits, finite CPU no-gradient state')

source=(ROOT/'solution/solution_v405.py').read_text()
assert source.startswith((ROOT/'solution/solution_v402.py').read_text())
for node in ast.walk(ast.parse(source)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
actual=load_solution(ROOT/'solution/solution_v405.py')
actual_answer=actual._do_refine(x,base,state,kh,d)
assert all(torch.equal(v,actual_answer[k]) for k,v in ans.items())
from self_check import validate_frozen_state
assert not validate_frozen_state(dict(dual_operator=fit1,fallback=None),'fit')
print('PASS: standalone exact core output, parent prefix/imports, frozen state validator')
