"""Single-pass metadata replacement and deployed/helper consistency."""
import ast
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params
torch.set_num_threads(4);torch.manual_seed(394)
m=load_solution(ROOT/'solution/solution_v394.py');old=load_solution(ROOT/'solution/solution_v387.py')
src=(ROOT/'solution/solution_v394.py').read_text()
assert src.startswith((ROOT/'solution/solution_v387.py').read_text())
for node in ast.walk(ast.parse(src)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
for d in (64,128,256):
    qh,kh=4,2
    for t in (1,7,129,1025):
        data=quantize_nvfp4(torch.randn(t,qh*d))
        h=torch.eye(d).repeat(kh,1,1)
        st=dict(gq=1,rf=1,kvh=kh,H=h,u=h,qs=torch.rand(kh,d)+.5,
                joint_transform=torch.eye(d).repeat(kh,1,1),temperature_gain=torch.rand(qh)+.5)
        baseline=old.hif4_dynamic_quantize_q(*data,qh,d,st)
        for active in (torch.zeros(kh,dtype=torch.bool),torch.ones(kh,dtype=torch.bool)):
            unit=dict(st,sp_active=active,sp_weight=torch.ones(kh,d))
            result=m.hif4_dynamic_quantize_q(*data,qh,d,unit)
            for key in baseline:assert torch.equal(baseline[key],result[key]),('unit',t,d,key)
        weight=torch.rand(kh,d)*3+.1
        weight=weight.reshape(kh,-1,64);weight=weight/weight.mean(-1,keepdim=True);weight=weight.reshape(kh,d)
        mixed=dict(st,sp_active=torch.tensor([True,False]),sp_weight=weight)
        transformed=m._kt_input(m.dequantize_nvfp4(*data).float(),st,qh,d,'q')
        pm=m._sp_params(transformed,mixed,qh,d,t)
        pa=m._sp_params(transformed,dict(mixed,sp_active=torch.ones(kh,dtype=torch.bool)),qh,d,t)
        for key in pm:assert torch.equal(pm[key][:,:2*d//64],pa[key][:,:2*d//64]),('mask-independent',t,d,key)
        snapshot={k:v.clone() for k,v in mixed.items() if isinstance(v,torch.Tensor)}
        rng=torch.random.get_rng_state().clone()
        result=m.hif4_dynamic_quantize_q(*data,qh,d,mixed)
        assert torch.equal(rng,torch.random.get_rng_state())
        for key,value in snapshot.items():assert torch.equal(value,mixed[key])
        assert not validate_hif4_params(result,data[0].shape,'mixed')
        for key in baseline:assert torch.equal(baseline[key][:,2*d//64:],result[key][:,2*d//64:]),('inactive',t,d,key)
        if t>1:
            ix=torch.linspace(0,t-1,min(t,8)).long()
            subset=m._joint_quant_queries({'q':data},mixed,qh,kh,d,ix)
            torch.testing.assert_close(subset,m._deq_params(result)[ix],rtol=0,atol=0)
        calls=[];original=m._gptq_quantize_batched
        def counted(*a):calls.append(1);return original(*a)
        m._gptq_quantize_batched=counted
        m.hif4_dynamic_quantize_q(*data,qh,d,mixed)
        m._gptq_quantize_batched=original
        assert len(calls)==int(t>1)
x=torch.randn(3,128);st=dict(sp_active=torch.ones(2,dtype=torch.bool),sp_weight=torch.rand(2,64))
for rows in (1,4097):
    result=m._sp_params(x,st,2,64,rows);baseline=m._dyn_table(x,None,False)
    for key in result:assert torch.equal(result[key],baseline[key])
x=torch.randn(1025,12*256)
w=torch.rand(3,4,64)+.01;w=(w/w.mean(-1,keepdim=True)).reshape(3,256)
st=dict(sp_active=torch.tensor([True,False,False]),sp_weight=w)
mixed=m._sp_params(x,st,12,256,1025)
all_heads=m._sp_params(x,dict(st,sp_active=torch.ones(3,dtype=torch.bool)),12,256,1025)
base=m._dyn_table(x,None,False)
for key in mixed:
    assert torch.equal(mixed[key][:,:16],all_heads[key][:,:16]),('large mask independence',key)
    assert torch.equal(mixed[key][:,16:],base[key][:,16:]),('large inactive',key)
print('PASS: unit/no-op, inactive exactness, legal output, single GPTQ, full/subset budgets, state/RNG, boundaries, large vectorized/loop consistency, prefix/imports')
