"""v395 joint-search invariants; independent of online-score prediction."""
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
torch.manual_seed(395)
m=load_solution(ROOT/'solution/solution_v395.py')
source=(ROOT/'solution/solution_v395.py').read_text()
assert source.startswith((ROOT/'solution/solution_v387.py').read_text())
for node in ast.walk(ast.parse(source)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
h=torch.tensor([[[1.,-.99],[-.99,1.]]],dtype=torch.float64)
x=torch.tensor([[[.6,.6]]],dtype=torch.float64)
def loss(y):
    e=y-x
    return (e@h*e).sum()
zero=torch.zeros_like(x)
for i in range(2):
    for sign in (-1,1):
        y=zero.clone();y[0,0,i]=sign
        assert loss(y)>loss(zero)
y=m._jb_beam(x,torch.ones_like(x),torch.linalg.cholesky(torch.linalg.inv(h),upper=True),4)
assert loss(y)<loss(zero)
for d,t in ((64,17),(128,129),(256,257)):
    qh,kh=4,2
    a=torch.randn(kh,d,d)
    st=m._jb_prepare({'H':a@a.transpose(-1,-2)/d+.1*torch.eye(d),'kvh':kh},kh,d)
    data=quantize_nvfp4(torch.randn(t,qh*d))
    raw=m.dequantize_nvfp4(*data).float()
    base=m._jb_parent_q(*data,qh,d,None)
    def metric(p):
        e=(m._deq_params(p).float()-raw).reshape(t,qh,d).transpose(0,1)
        return (torch.bmm(e,st['jb_h'].repeat_interleave(2,0))*e).sum(-1)
    old=metric(base); results={}
    for mode in ('scale','code','joint'):
        p=m._jb_search(raw,base,st,qh,d,mode)
        assert not validate_hif4_params(p,data[0].shape,mode)
        assert bool((metric(p)<=old+1e-4).all())
        results[mode]=p
    for mode in ('scale','code'):
        assert bool((metric(results['joint'])<=metric(results[mode])+1e-4).all())
    mixed=dict(st,jb_active=torch.tensor([True,False]))
    p=m._jb_search(raw,base,mixed,qh,d)
    for key in p:
        assert torch.equal(p[key][:,:2*d//64],results['joint'][key][:,:2*d//64])
        assert torch.equal(p[key][:,2*d//64:],base[key][:,2*d//64:])
    changed=(m._deq_params(results['joint'])!=m._deq_params(base)).reshape(t,qh,d//64,64).any(-1)
    assert int(changed.sum())<=128
    assert int(changed.sum(-1).max())<=1
    rng=torch.random.get_rng_state().clone()
    snapshot={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in st.items()}
    m._jb_search(raw,base,st,qh,d)
    assert torch.equal(rng,torch.random.get_rng_state())
    for k,v in snapshot.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,st[k])
    full=m.hif4_dynamic_quantize_q(*data,qh,d,st)
    ix=torch.tensor([0,t//2,t-1])
    torch.testing.assert_close(m._joint_quant_queries({'q':data},st,qh,kh,d,ix),m._deq_params(full)[ix],rtol=0,atol=0)
    for length in (1,4097):
        assert m._jb_search(torch.empty(length,qh*d),base,st,qh,d) is base
print('PASS: correlated local-minimum escape, legal codes, full-H monotonicity, joint dominates both ablations, mask independence, 128-block budget, state/RNG immutable, full-query helper, boundaries, parent prefix/imports')
