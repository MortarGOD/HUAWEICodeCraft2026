"""Independent integer-search checks and deployed parameter validation."""
import ast
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params
torch.set_num_threads(4)
torch.manual_seed(392)
m=load_solution(ROOT/'solution/solution_v392.py')
source=(ROOT/'solution/solution_v392.py').read_text()
assert source.startswith((ROOT/'solution/solution_v387.py').read_text())
for node in ast.walk(ast.parse(source)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
for t in (4,17,64):
    x=torch.randn(t,9,dtype=torch.float64)
    step=torch.rand(t,9,dtype=torch.float64)*.3+.2
    y=(x/step).round().clamp(-7,7)*step
    p=torch.softmax(torch.randn(8,t,dtype=torch.float64),-1)
    def cost(z):return (p@(x-z)).square().mean(0)+(x-z).square().sum(0)/(t*t)
    m._KC_ROUNDS=1
    actual=m._kc_descent(x,y,step,p)
    # Exhaust all 15 codes and every token independently for each channel.
    for c in range(9):
        best=cost(y)[c]; base_sse=(x[:,c]-y[:,c]).square().sum()
        for row in range(t):
            for code in range(-7,8):
                trial=y.clone();trial[row,c]=step[row,c]*code
                if (x[:,c]-trial[:,c]).square().sum() <= base_sse*1.05+1e-12:
                    best=min(best,cost(trial)[c])
        torch.testing.assert_close(cost(actual)[c],best,rtol=1e-10,atol=1e-12)
    m._KC_ROUNDS=4
    result=m._kc_descent(x,y,step,p)
    assert bool((cost(result)<=cost(y)+1e-12).all())
    assert bool(((x-result).square().sum(0)<=(x-y).square().sum(0)*1.05+1e-12).all())

for t in (2,4,32,129):
    data=quantize_nvfp4(torch.randn(t,128))
    x=m.dequantize_nvfp4(*data).float()
    base=m._vk_parent_v(*data,None,2,64)
    st=dict(center=torch.zeros(128),projection=torch.randn(128,8)*.03,
            kernel=torch.randn(4,9,9)*.02,active=torch.tensor([True,False]))
    snapshot={k:v.clone() for k,v in st.items()}
    rng=torch.random.get_rng_state().clone()
    result=m._vk_refine(x,base,st,2,64)
    assert not validate_hif4_params(result,data[0].shape,'active')
    assert torch.equal(rng,torch.random.get_rng_state())
    for key in snapshot:assert torch.equal(snapshot[key],st[key])
    for key in base:assert torch.equal(base[key][:,1:],result[key][:,1:]),key
    e0=x-m._deq_params(base);e1=x-m._deq_params(result)
    assert bool((e1.square().sum(0)<=e0.square().sum(0)*1.05+1e-10).all())
    prob=m._vk_probabilities(x,st,2)[0]
    c0=(prob@e0[:,:64]).square().mean(0)+e0[:,:64].square().mean(0)/t
    c1=(prob@e1[:,:64]).square().mean(0)+e1[:,:64].square().mean(0)/t
    assert bool((c1<=c0+1e-10).all())
sentinel={}
for t in (1,4097):
    assert m._vk_refine(torch.empty(t,128),sentinel,None,2,64) is sentinel
assert m._vk_refine(torch.empty(4,128),sentinel,dict(active=torch.zeros(2,dtype=torch.bool)),2,64) is sentinel
print('PASS: exhaustive optimum, deployed MSE and kernel objective, inactive head exactness, state/RNG immutability, boundary bypass, prefix/import whitelist')
