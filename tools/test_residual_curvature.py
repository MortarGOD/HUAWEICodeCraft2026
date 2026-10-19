"""Check the derived Hessian against autograd, before PSD approximation."""
import ast
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4);torch.manual_seed(393)
m=load_solution(ROOT/'solution/solution_v393.py')
src=(ROOT/'solution/solution_v393.py').read_text()
assert src.startswith((ROOT/'solution/solution_v387.py').read_text())
for node in ast.walk(ast.parse(src)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
for d in (4,16,64):
    q=torch.randn(3,d,dtype=torch.float64)
    k=torch.randn(11,d,dtype=torch.float64)
    v=torch.randn(11,d,dtype=torch.float64)
    target=torch.randn(3,d,dtype=torch.float64)
    gn,c=m._eh_query_terms(q,k,v,target,d)
    def loss(shift):
        out=torch.softmax((q+shift)@k.T/d**.5,-1)@v
        return .5*(out-target).square().sum()
    expected=torch.autograd.functional.hessian(loss,torch.zeros(d,dtype=torch.float64))
    torch.testing.assert_close(gn+c,expected,rtol=1e-9,atol=1e-10)
    out=torch.softmax(q@k.T/d**.5,-1)@v
    gn0,c0=m._eh_query_terms(q,k,v,out,d)
    torch.testing.assert_close(c0,torch.zeros_like(c0),rtol=0,atol=1e-12)
    torch.testing.assert_close(gn0,gn,rtol=0,atol=0)
    psd,negative=m._eh_project(gn+c)
    assert torch.linalg.eigvalsh(psd).min()>-1e-10
    torch.testing.assert_close(psd,psd.T,rtol=1e-10,atol=1e-10)
    shift=torch.randn(1,d,dtype=torch.float64)
    g2,c2=m._eh_query_terms(q,k+shift,v,target,d)
    torch.testing.assert_close(g2+c2,gn+c,rtol=1e-9,atol=1e-10)
    # Positive Jacobian term agrees with the actual per-query Jacobians.
    check=torch.zeros_like(gn)
    for row in q:
        jac=torch.autograd.functional.jacobian(lambda z:torch.softmax(z@k.T/d**.5,-1)@v,row)
        check+=jac.T@jac
    torch.testing.assert_close(check,gn,rtol=1e-9,atol=1e-10)
# Exercise fitted target/query layout for GQA with more than 16 tokens.
n,qh,kh,d=19,4,2,4
sample={r:(torch.randn(n,heads*d),torch.ones(n,1)) for r,heads in [('q',qh),('k',kh),('v',kh)]}
m.dequantize_nvfp4=lambda x,s:x
m._joint_quant_queries=lambda sample,st,qh,kh,d,ix:sample['q'][0][ix]*.8
m.hif4_dynamic_quantize_k=lambda x,s,kh,d,st:x*.9
m.hif4_dynamic_quantize_v=lambda x,s,kh,d,st:x*.95
m._deq_params=lambda p:p
ix=torch.linspace(0,n-1,16).long()
metrics=m._oj_fit([sample,sample,sample],dict(q_state={},k_state={},v_state={}),qh,kh,d)[0]
expected=[]
for head in range(kh):
    q=sample['q'][0][ix].reshape(-1,qh,d)[:,head*2:head*2+2].reshape(-1,d)
    k=sample['k'][0].reshape(n,kh,d)[:,head]
    v=sample['v'][0].reshape(n,kh,d)[:,head]
    target=torch.softmax(q@k.T/d**.5,-1)@v
    gn,c=m._eh_query_terms(q*.8,k*.9,v*.95,target,d)
    expected.append(m._eh_project(gn+c)[0])
torch.testing.assert_close(metrics,m._oj_normalize(torch.stack(expected)),rtol=2e-5,atol=2e-6)
assert m.EH_DEBUG['folds']==2 and m.EH_DEBUG['finite']
print('PASS: exact Hessian/autograd, zero residual GN, PSD, key-translation invariance, Jacobian orientation, GQA fit/target order, parent prefix/imports')
