"""Math and deployed-path checks for v389; no online score assertions."""
import ast
import copy
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import load_solution
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params, validate_attention_calibration_result

torch.set_num_threads(4)
torch.manual_seed(389)
old = load_solution(ROOT / 'solution/solution_v387.py')
m = load_solution(ROOT / 'solution/solution_v389.py')
source = (ROOT / 'solution/solution_v389.py').read_text()
assert source.startswith((ROOT / 'solution/solution_v387.py').read_text())
for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Import):
        assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node, ast.ImportFrom):
        assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}

for d in (64,128,256):
    q = torch.randn(3,d) * .2; k = torch.randn(11,d); v = torch.randn(11,d)
    target = torch.randn(3,d) * .1
    h, g = m._rb_moments(q,k,v,target,d)
    delta = torch.zeros(d,requires_grad=True)
    loss = .5 * ((torch.softmax((q+delta)@k.T/d**.5,-1)@v-target).square().sum()) / len(q)
    expected = torch.autograd.grad(loss,delta)[0]
    torch.testing.assert_close(g,expected,atol=2e-6,rtol=2e-5)
    expected_h = torch.zeros_like(h)
    for row in q:
        jac = torch.autograd.functional.jacobian(lambda x: torch.softmax(x@k.T/d**.5,-1)@v,row)
        expected_h += jac.T@jac/len(q)
    torch.testing.assert_close(h,expected_h,atol=2e-6,rtol=2e-5)
    radius = q.square().sum(-1).mean().sqrt().reshape(1)
    step = m._rb_solve(h[None],g[None],radius)[0]
    assert float(g@step) <= 1e-8
    assert step.norm() <= .05001*radius[0]
    assert m._rb_solve(h[None]*float('nan'),g[None],radius) is None

qh,kh,d = 4,2,64
for rows in (1,7,32,129,1025):
    data = quantize_nvfp4(torch.randn(rows,qh*d))
    rotation = torch.linalg.qr(torch.randn(kh,d,d)).Q
    st = dict(gq=1,rf=1,kvh=kh,u=torch.eye(d).repeat(kh,1,1),H=torch.eye(d).repeat(kh,1,1),
              qs=torch.rand(kh,d)+.5,joint_transform=rotation,temperature_gain=torch.rand(qh)+.5)
    before = old.hif4_dynamic_quantize_q(*data,qh,d,st)
    zero = dict(st,rb_shift=torch.zeros(kh,d))
    actual = m.hif4_dynamic_quantize_q(*data,qh,d,zero)
    for key in before: assert torch.equal(before[key],actual[key]), (rows,key)
    assert not validate_hif4_params(actual,data[0].shape,'zero')
    shift = torch.randn(kh,d)*.03; shift[1] = 0
    active = dict(st,rb_shift=shift)
    snapshot = {k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in active.items()}
    altered = m.hif4_dynamic_quantize_q(*data,qh,d,active)
    assert not validate_hif4_params(altered,data[0].shape,'active')
    for key in before:
        assert torch.equal(before[key][:,2:],altered[key][:,2:]), ('inactive group',rows,key)
        if rows == 1: assert torch.equal(before[key],altered[key])
    for key,value in snapshot.items():
        if isinstance(value,torch.Tensor):assert torch.equal(value,active[key])
    if rows > 1:
        ix = torch.linspace(0,rows-1,min(rows,8)).long()
        observed = m._joint_quant_queries({'q':data},active,qh,kh,d,ix)
        torch.testing.assert_close(observed,m._deq_params(altered)[ix],rtol=0,atol=0)

sentinel = {'q_state':{},'k_state':{},'v_state':{}}
m._rb_parent_calibration=lambda *args:sentinel
assert m.hif4_calibration_attention([],qh,kh,d) is sentinel
assert not validate_attention_calibration_result(sentinel,'sentinel')
print('PASS: residual gradient/Hessian, descent/trust radius/nonfinite, original prefix/imports, zero shift equivalence, inactive groups, full/subset query budgets, single-token bypass, state immutability, empty fallback')
