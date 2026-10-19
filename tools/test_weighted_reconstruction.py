"""v396: mechanism, deployed-code and state invariants."""
import ast
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params,validate_attention_calibration_result
torch.set_num_threads(4);torch.manual_seed(396)
m=load_solution(ROOT/'solution/solution_v396.py')
source=(ROOT/'solution/solution_v396.py').read_text()
assert source.startswith((ROOT/'solution/solution_v387.py').read_text())
for node in ast.walk(ast.parse(source)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
# Repeated paired K/V atoms: mass weighting reproduces the full softmax.
k=torch.randn(1,3,8);v=torch.randn(1,3,8)
ix=torch.tensor([0,1,1,1,2,2,2,2,2]);kf=k[:,ix];vf=v[:,ix]
b=m._wr_cluster(kf,kf,vf,vf,3)
q=torch.randn(1,8)
out,_=m._wr_output(q,b['key'][:,None],b['value'][:,None],b['logmass'][:,None])
full=torch.softmax(q[:,None]@kf.transpose(1,2),-1)@vf
torch.testing.assert_close(out,full,rtol=1e-5,atol=1e-6)
# Analytic full-dimensional nonlinear gradient, including per-sequence means.
bank={k:torch.randn(3,2,7,8,dtype=torch.float64) for k in ('key','value')}
bank['logmass']=torch.randn(3,2,7,dtype=torch.float64)
q=torch.randn(3,8,dtype=torch.float64,requires_grad=True);target=torch.randn(3,2,8,dtype=torch.float64)
cost,grad=m._wr_cost_gradient(q,target,bank,True)
torch.testing.assert_close(torch.autograd.grad(cost.sum(),q)[0],grad,rtol=1e-10,atol=1e-10)
# Separate normalizers: each one-point sequence has its own constant output.
q=torch.ones(1,8);key=torch.stack((torch.zeros(1,1,8),torch.ones(1,1,8)),1)
val=torch.stack((torch.zeros(1,1,8),torch.ones(1,1,8)),1)
out,_=m._wr_output(q,key,val,torch.zeros(1,2,1))
torch.testing.assert_close(out,val.squeeze(2))
for d,t in ((64,17),(128,129),(256,257)):
    qh,kh=4,2
    data=quantize_nvfp4(torch.randn(t,qh*d));raw=m.dequantize_nvfp4(*data).float()
    base=m._wr_parent_q(*data,qh,d,None)
    key=torch.randn(kh,2,16,d)/(d**.5);value=torch.randn(kh,2,16,d)
    bank=dict(key=key,raw=key+.02*torch.randn_like(key),value=value,raw_value=value+.02*torch.randn_like(value),logmass=torch.rand(kh,2,16),active=torch.ones(qh,dtype=torch.bool))
    def losses(p):
        q=m._deq_params(p).reshape(t,qh,d).reshape(-1,d)
        r=raw.reshape(t,qh,d).reshape(-1,d)
        groups=torch.arange(qh).repeat(t)//2
        selected={k:bank[k][groups] for k in ('key','raw','value','raw_value','logmass')}
        target,_=m._wr_output(r,selected['raw'],selected['raw_value'],selected['logmass'])
        return m._wr_cost_gradient(q,target,selected)
    result=m._wr_refine(raw,raw,base,bank,qh,d)
    assert not validate_hif4_params(result,data[0].shape,'wr')
    assert bool((losses(result)<=losses(base)+1e-6).all())
    mixed=dict(bank,active=torch.tensor([True,False,True,False]))
    partial=m._wr_refine(raw,raw,base,mixed,qh,d)
    for name in base:
        for h in range(qh):
            sl=slice(h*d//64,(h+1)*d//64)
            assert torch.equal(partial[name][:,sl],(result if h%2==0 else base)[name][:,sl])
    change=(m._deq_params(result)!=m._deq_params(base)).reshape(t,qh,d).any(-1)
    assert int(change.sum())<=256
    saved={k:v.clone() for k,v in bank.items()};rng=torch.random.get_rng_state().clone()
    state={'kvh':kh,'wr':bank}
    assert not validate_attention_calibration_result({'q_state':state,'k_state':None,'v_state':None},'forced_active')
    full=m.hif4_dynamic_quantize_q(*data,qh,d,state)
    ix=torch.tensor([0,t//2,t-1])
    torch.testing.assert_close(m._joint_quant_queries({'q':data},state,qh,kh,d,ix),m._deq_params(full)[ix],rtol=0,atol=0)
    assert torch.equal(rng,torch.random.get_rng_state())
    for k,v in saved.items():assert torch.equal(v,bank[k])
    for length in (1,4097):assert m._wr_refine(torch.empty(length,qh*d),None,base,bank,qh,d) is base
print('PASS: exact repeated-atom mass, sequence normalizers, autograd gradient, packed loss monotonicity, legal outputs, masks, budget, helper, boundaries, state/RNG, parent prefix/imports')
