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
torch.manual_seed(391)
m=load_solution(ROOT/'solution/solution_v391.py')
source=(ROOT/'solution/solution_v391.py').read_text()
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
    m._KI_ROUNDS=1
    actual=m._ki_descent(x,y,step,p)
    # Exhaust all 15 codes and every token independently for each channel.
    for c in range(9):
        best=cost(y)[c]; base_sse=(x[:,c]-y[:,c]).square().sum()
        for row in range(t):
            for code in range(-7,8):
                trial=y.clone();trial[row,c]=step[row,c]*code
                if (x[:,c]-trial[:,c]).square().sum() <= base_sse*1.05+1e-12:
                    best=min(best,cost(trial)[c])
        torch.testing.assert_close(cost(actual)[c],best,rtol=1e-10,atol=1e-12)
    m._KI_ROUNDS=4
    result=m._ki_descent(x,y,step,p)
    assert bool((cost(result)<=cost(y)+1e-12).all())
    assert bool(((x-result).square().sum(0)<=(x-y).square().sum(0)*1.05+1e-12).all())
for t in (4,32,129):
    data=quantize_nvfp4(torch.randn(t,128))
    x=m.dequantize_nvfp4(*data).float()
    base=m.hif4_dynamic_quantize_v(*data,2,64,None)
    st=dict(ki_probes=torch.randn(2,8,64),ki_active=torch.tensor([True,False]))
    result=m._ki_quant(x,base,st,2,64)
    assert not validate_hif4_params(result,data[0].shape,'active')
    for key in base:assert torch.equal(base[key][:,1:],result[key][:,1:]),key
    assert bool(((x-m._deq_params(result)).square().sum(0)<=(x-m._deq_params(base)).square().sum(0)*1.05001+1e-12).all())
samples=[]
for _ in range(5):
    v=quantize_nvfp4(torch.randn(8,128));k=tuple(z.clone() for z in v)
    k[0][:,64:]=0
    samples.append(dict(q=quantize_nvfp4(torch.randn(8,256)),k=k,v=v))
st=m._ki_fit(samples,4,2,64)
assert st['ki_active'].tolist()==[True,False]
saved=m._ki_parent_v
calls=[]
sentinel={'sentinel':True}
m._ki_parent_v=lambda *a:(calls.append(a[-1]) or sentinel)
for t in (1,4097):
    assert m.hif4_dynamic_quantize_v(torch.empty(t,128),None,2,64,dict(st,ki_fallback=None)) is sentinel
assert calls==[None,None]
assert m.hif4_dynamic_quantize_v(torch.empty(4,128),None,2,64,None) is sentinel
m._ki_parent_v=saved
data=quantize_nvfp4(torch.randn(32,128))
active=dict(st,ki_fallback=None)
snapshot={k:v.clone() for k,v in st.items()}
rng=torch.random.get_rng_state().clone()
answer=m.hif4_dynamic_quantize_v(*data,2,64,active)
assert not validate_hif4_params(answer,data[0].shape,'dynamic')
assert torch.equal(rng,torch.random.get_rng_state())
for k,v in snapshot.items():assert torch.equal(v,active[k])
print('PASS: exhaustive 15-code optimum, four-round descent/MSE cap, legal deployed codes, inactive head exactness, per-head identity, parent prefix/imports')
