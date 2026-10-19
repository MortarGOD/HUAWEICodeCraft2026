"""v397 risk selection, full-sequence verification and capture isolation."""
import ast
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params,validate_attention_calibration_result
torch.set_num_threads(4);torch.manual_seed(397)
m=load_solution(ROOT/'solution/solution_v397.py')
source=(ROOT/'solution/solution_v397.py').read_text()
assert source.startswith((ROOT/'solution/solution_v387.py').read_text())
for node in ast.walk(ast.parse(source)):
    if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
    if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
cost=torch.tensor([[1.,1.,1.],[.8,.8,1.2],[.995,.995,.995],[.97,.98,1.001]])
assert m._cs_pick(cost)==3
assert m._cs_pick(cost*torch.tensor([.1,10.,100.]))==3
assert m._cs_pick(torch.tensor([[1.,1.,1.],[.8,1.001,1.001]]))==0
assert m._cs_pick(torch.tensor([[1.,1.,1.],[.999,.999,.999]]))==0
assert m._cs_pick(torch.tensor([[1.,1.,1.],[float('nan'),.8,.8]]))==0
assert m._cs_pick(torch.zeros(4,3))==0
# A joint improvement can survive even when either single change loses.
assert m._cs_pick(torch.tensor([[1.,1.,1.],[1.1,1.1,1.1],[1.2,1.2,1.2],[.9,.9,.9]]))==3
for t in (7,129,257):
    qh,kh,d=4,2,64
    raw=[torch.randn(t,h*d) for h in (qh,kh,kh)]
    deployed=[x+.05*torch.randn_like(x) for x in raw]
    ix=torch.arange(t)
    direct=(m._joint_output(*deployed,qh,kh,d,ix)-m._joint_output(*raw,qh,kh,d,ix)).square().double().mean()
    torch.testing.assert_close(m._cs_full_cost(*raw,*deployed,qh,kh,d),direct,rtol=1e-6,atol=1e-9)
samples=[{role:quantize_nvfp4(torch.randn(t,h*64)) for role,h in zip('qkv',(4,2,2))} for t in (8,12,16,20,24)]
captured=m._cs_parent_calibration(samples,4,2,64)
assert '_cs_logit_q' in captured and '_cs_plain_k' in captured
assert not validate_attention_calibration_result(m._cs_clean(captured),'clean')
for candidate in m._cs_candidates(captured):
    assert not validate_attention_calibration_result(candidate,'candidate')
    assert candidate['v_state'] is captured['v_state']
    for role,heads in zip('qkv',(4,2,2)):
        data=samples[-1][role]
        p=getattr(m,'hif4_dynamic_quantize_'+role)(*data,heads,64,candidate[role+'_state'])
        assert not validate_hif4_params(p,data[0].shape,role)
# Captures return only temporary outer fields and do not mutate their inputs.
oldbase=m._cs_parent_base;oldkt=m._cs_parent_kt
sentinel=m._cs_clean(captured);m._cs_parent_base=lambda *args:sentinel;m._cs_parent_kt=lambda *args:sentinel
snap=set(sentinel)
assert m._oj_parent_calibration(samples,4,2,64)['_cs_logit_q'] is sentinel['q_state']
assert m._kt_guard(samples,sentinel,4,2,64)['_cs_plain_k'] is sentinel['k_state']
assert set(sentinel)==snap
m._cs_parent_base=oldbase;m._cs_parent_kt=oldkt
# Synthetic small calibration does not enable GPTQ naturally; construct a
# valid active metric explicitly to exercise routing, not to claim real gain.
captured=dict(captured)
for role in 'qk':
    captured[role+'_state']=dict(captured[role+'_state'],gq=True,H=torch.eye(64).repeat(2,1,1),u=torch.eye(64).repeat(2,1,1),rf=0)
captured['_cs_logit_q']=dict(captured['q_state'])
captured['_cs_plain_k']=dict(captured['k_state'])
savedlimit=m._CS_WORK_LIMIT;m._CS_WORK_LIMIT=0
answer=m._cs_select(samples,captured,4,2,64)
assert not validate_attention_calibration_result(answer,'budget')
assert set(answer)=={'q_state','k_state','v_state'}
assert all(answer[k] is captured[k] for k in answer)
assert m.CS_DEBUG['reason']=='work_budget'
m._CS_WORK_LIMIT=savedlimit
# Force the full verification control-flow only in this test; no fake score report.
picker=m._cs_pick;m._cs_pick=lambda costs:1
answer=m._cs_select(samples,captured,4,2,64)
assert m.CS_DEBUG['reason']=='accepted' and len(m.CS_DEBUG['full_ratios'])==3
assert not validate_attention_calibration_result(answer,'forced_branch')
expected=m._cs_candidates(captured)[1]
for role,heads in zip('qkv',(4,2,2)):
    fn=getattr(m,'hif4_dynamic_quantize_'+role);data=samples[-1][role]
    a=fn(*data,heads,64,answer[role+'_state']);b=fn(*data,heads,64,expected[role+'_state'])
    assert all(torch.equal(a[k],b[k]) for k in a)
m._cs_pick=picker
print('PASS: normalized cross-sequence/tail risk, interaction case, full streamed MSE, all four legal combinations, temporary capture isolation, budget fallback, forced full branch and exact deployment, parent prefix/imports')
