"""Local-only Jacobian and deployment-point regression tests."""
from pathlib import Path
import sys
import ast
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4)
torch.manual_seed(387)
path = ROOT/'solution/solution_v387.py'
source = path.read_text()
assert source.startswith((ROOT/'solution/solution_v382.py').read_text())
allowed = {'__future__', 'typing', 'torch', 'numpy', 'threading'}
for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Import):
        assert all(a.name.split('.')[0] in allowed for a in node.names)
    elif isinstance(node, ast.ImportFrom):
        assert node.module.split('.')[0] in allowed
m = load_solution(path)
for d in (64, 128, 256):
    q = torch.randn(2,d)*.3
    k = torch.randn(11,d)
    v = torch.randn(11,d)
    actual = m._dm_query_gram(q,k,v,d)
    expected = torch.zeros_like(actual)
    for row in q:
        # Autograd differentiates the output, not the HiF4 rounding operation.
        j = torch.autograd.functional.jacobian(lambda x: torch.softmax(x@k.T/d**.5,-1)@v, row)
        expected += j.T@j
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    direction = torch.randn(d); direction /= direction.norm()
    eps = 1e-3
    f = lambda x: torch.softmax(x@k.T/d**.5,-1)@v
    fd = (f(q[0]+eps*direction)-f(q[0]-eps*direction))/(2*eps)
    j = torch.autograd.functional.jacobian(f,q[0])
    torch.testing.assert_close(fd,j@direction,atol=2e-4,rtol=2e-3)

# Force distinctive deployed values and prohibit raw NVFP4 decoding in fit.
d=64; qh=4; kh=2; n=17
sample={r:(torch.zeros(n, qh*d if r=='q' else kh*d),torch.ones(1)) for r in 'qkv'}
states={r+'_state':{} for r in 'qkv'}
q=torch.randn(16,qh*d); k=torch.randn(n,kh*d); v=torch.randn(n,kh*d)
calls={'q':0,'k':0,'v':0}
def query(s,st,a,b,c,ix):
    assert (a,b,c)==(qh,kh,d) and len(ix)==16 and s['k'][0].shape[0]==n
    calls['q']+=1
    return q
def key(*args): calls['k']+=1; return k
def value(*args): calls['v']+=1; return v
def forbidden(*args): raise AssertionError('raw decoder used by deployed fit')
m._joint_quant_queries=query
m.hif4_dynamic_quantize_k=key
m.hif4_dynamic_quantize_v=value
m._deq_params=lambda x:x
m.dequantize_nvfp4=forbidden
fit=m._oj_fit([sample]*5,states,qh,kh,d)[0]
expect=torch.stack([m._dm_query_gram(q.reshape(16,kh,2,d)[:,h].reshape(-1,d),k.reshape(n,kh,d)[:,h],v.reshape(n,kh,d)[:,h],d) for h in range(kh)])
torch.testing.assert_close(fit,m._oj_normalize(expect))
assert calls=={'q':2,'k':2,'v':2} and m.DM_DEBUG['folds']==2
assert m._oj_fit([sample],states,qh,kh,d) is None
v.fill_(float('nan'))
assert m._oj_fit([sample]*5,states,qh,kh,d) is None and not m.DM_DEBUG['finite']
assert states=={r+'_state':{} for r in 'qkv'}
print('PASS: source prefix/imports, Jacobian/autograd/finite differences, GQA deployed operands, budgets, nonfinite/empty fallback, parent immutability')
