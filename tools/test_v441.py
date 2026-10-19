"""Mathematics, integration and bounded synthetic cost for v441."""
import ast
import copy
import json
from pathlib import Path
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
from self_check import validate_attention_calibration_result, validate_hif4_params
from hif4_quant import dequantize_hif4

torch.set_num_threads(4)
m, base = load(441, 'mechanism'), load(435, 'mechanism')
gen = torch.Generator().manual_seed(441)
report = {'checks': [], 'cases': []}
parent = (ROOT / 'solution/solution_v435.py').read_bytes()
source = (ROOT / 'solution/solution_v441.py').read_bytes()
assert source.startswith(parent)
assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(ast.parse(source[len(parent):])))
for name in ('hif4_calibration_and_quantize_weight', 'hif4_dynamic_quantize_activation',
             'hif4_dynamic_quantize_q', 'hif4_dynamic_quantize_k'):
    assert getattr(m, name).__code__.co_code == getattr(base, name).__code__.co_code
report['checks'].append('prefix_no_imports_four_apis_unchanged')

def model(kh=2, qh=4, d=64):
    f = 5
    return {'center': torch.zeros(kh*d), 'projection': torch.randn(kh*d, f-1, generator=gen)*.03,
            'kernel': torch.randn(qh, f, f, generator=gen)*.05,
            'query_sigma_modes': torch.randn(qh, 2, f, generator=gen)*.1,
            'key_error_metric': torch.eye(f)[None].repeat(qh,1,1)*.2,
            'key_error_floor': torch.ones(qh)*.1, 'active': torch.arange(kh)%2==0,
            'a430_temperature': torch.ones(kh)*1.3, 'a430_active': torch.ones(kh,dtype=torch.bool)}

for t in (2, 7, 32, 128):
    q, s = packed_pair(t, 128, gen); x = m.dequantize_nvfp4(q,s).float()
    state = model(); frozen = copy.deepcopy(state)
    p, variance = m._v441_fields(x,state,2)
    f = m._vk_features(x,state);ix=torch.linspace(0,t-1,min(m._VK_QUERIES,t)).long()
    v = ((f[ix][None]@state['key_error_metric'])*f[ix][None]).sum(-1)+state['key_error_floor'][:,None]
    expected = v[:,:,None].expand(-1,-1,4).reshape(2,-1)
    assert torch.allclose(variance,expected)
    assert torch.equal(p,m._vk_probabilities(x,state,2))
    pd=p.double();vd=variance.double()
    j=torch.diag_embed(pd)-pd.unsqueeze(-1)*pd.unsqueeze(-2)
    h=pd.transpose(-1,-2)@pd/p.shape[1]+(vd[:,:,None,None]*(j.transpose(-1,-2)@j)).mean(1)
    h+=torch.eye(t,dtype=torch.float64)/t**2
    e=torch.randn(2,t,64,generator=gen,dtype=torch.float64)
    grad,diag=m._v441_terms(e,pd,vd)
    assert torch.allclose(grad,h@e,atol=1e-10,rtol=1e-10)
    assert torch.allclose(diag,h.diagonal(dim1=-1,dim2=-2),atol=1e-10,rtol=1e-10)
    params=m._dyn_table(x,None,False)
    before=dequantize_hif4(params,x.shape).reshape(t,2,64).transpose(0,1)
    start=time.perf_counter();out=m._v441_refine(x,params,state,2,64);elapsed=time.perf_counter()-start
    assert not validate_hif4_params(out,x.shape,'risk')
    after=dequantize_hif4(out,x.shape).reshape(t,2,64).transpose(0,1)
    target=x.reshape(t,2,64).transpose(0,1)
    e0=target-before;e1=target-after
    c0=(e0.double()*(h@e0.double())).sum(1);c1=(e1.double()*(h@e1.double())).sum(1)
    assert torch.all(c1<=c0+1e-8)
    assert torch.all(e1.double().square().sum(1)<=e0.double().square().sum(1)*1.05+1e-9)
    assert torch.equal(before[1],after[1]) and equal(state,frozen)
    report['cases'].append({'t':t,'changed':int((before!=after).sum()),'seconds':elapsed})
assert sum(r['changed'] for r in report['cases'])>0
report['checks'].append('gqa_sigma_temperature_gradient_curvature_packed_risk_sse_inactive')

samples=[{r:packed_pair(t,256 if r=='q' else 128,gen) for r in 'qkv'} for t in (64,80,96)]
captured=[];fn=m._v441_parent_calibration
def capture(*a):
    result=fn(*a);captured.append(copy.deepcopy(result));return result
m._v441_parent_calibration=capture
snapshot=copy.deepcopy(samples);rng=torch.get_rng_state().clone()
result=m.hif4_calibration_attention(samples,4,2,64);ps=captured[0]
assert not validate_attention_calibration_result(result,'v441')
assert equal(ps['q_state'],result['q_state']) and equal(ps['k_state'],result['k_state'])
assert equal(snapshot,samples) and torch.equal(rng,torch.get_rng_state())
fit=m._v441_fit(samples,ps['v_state'],4,2,64)
assert fit is not None
native=ps['v_state']['v435_content_parent']
if isinstance(native,dict) and 'kernel' in native:
    assert equal(fit['kernel'],native['kernel'])
    for name in ('a430_temperature','a430_active','query_sigma_modes'):
        if name in native:assert equal(fit[name],native[name])
forced=dict(ps['v_state'],v441_key_risk=fit)
fresh=load(441,'fresh')
for sample in samples:
    saved=copy.deepcopy(forced)
    actual=m.hif4_dynamic_quantize_v(*sample['v'],2,64,forced)
    assert not validate_hif4_params(actual,sample['v'][0].shape,'forced')
    assert equal(actual,fresh.hif4_dynamic_quantize_v(*sample['v'],2,64,copy.deepcopy(forced)))
    assert equal(saved,forced)
report['checks'].append('actual_parent_fit_qk_immutable_fresh_forced_model')

# Ensure presence of an inverse factor never forces GPTQ when gq is disabled.
seen=[]
fit_fn,refine_fn,query_fn=m._v441_fit,m._v441_refine,m.hif4_dynamic_quantize_q
m._v441_fit=lambda *a:fit
def forced_change(value,params,*a):
    out={k:v.clone() for k,v in params.items()};out['mant']=torch.zeros_like(out['mant']);out['sign']=torch.zeros_like(out['sign']);return out
m._v441_refine=forced_change
def query(*a):seen.append(True);return query_fn(*a)
m.hif4_dynamic_quantize_q=query
helper=m._joint_quant_queries
def forbidden(*a):raise AssertionError('RTN used GPTQ subset helper')
m._joint_quant_queries=forbidden
plain=dict(ps,q_state=dict(ps['q_state'],gq=0))
m._v441_calibrate(samples,plain,4,2,64)
assert seen
m._v441_fit,m._v441_refine,m.hif4_dynamic_quantize_q=fit_fn,refine_fn,query_fn
m._joint_quant_queries=helper
report['checks'].append('rtn_calibration_uses_public_q')
report['completed']=True
(ROOT/'docs/v441-mechanisms.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
