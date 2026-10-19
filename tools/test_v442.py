"""v442 mechanism checks against v441 with zero K variance."""
import ast, copy, json, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
from self_check import validate_hif4_params,validate_attention_calibration_result

torch.set_num_threads(4)
m,ref,base=load(442,'check'),load(441,'check'),load(435,'check')
gen=torch.Generator().manual_seed(442)
report={'cases':[],'checks':[]}
parent=(ROOT/'solution/solution_v435.py').read_bytes();source=(ROOT/'solution/solution_v442.py').read_bytes()
assert source.startswith(parent)
assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(ast.parse(source[len(parent):])))
for name in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation','hif4_dynamic_quantize_q','hif4_dynamic_quantize_k'):
    assert getattr(m,name).__code__.co_code==getattr(base,name).__code__.co_code
report['checks'].append('prefix_four_apis_no_new_import')
f,kh,qh,d=9,2,4,64
model={'center':torch.zeros(kh*d),'projection':torch.randn(kh*d,f-1,generator=gen)*.02,
       'kernel':torch.randn(qh,f,f,generator=gen)*.04,'query_sigma_modes':torch.randn(qh,2,f,generator=gen)*.01,
       'active':torch.tensor([True,False]),'a430_temperature':torch.tensor([1.2,.7]),'a430_active':torch.tensor([True,False])}
zero=dict(model,key_error_metric=torch.zeros(qh,f,f),key_error_floor=torch.zeros(qh))
for t in (1,2,7,32,128,4097):
    q,s=packed_pair(t,kh*d,gen);x=m.dequantize_nvfp4(q,s).float();params=m._dyn_table(x,None,False)
    snapshot=copy.deepcopy((x,params,model));rng=torch.get_rng_state().clone()
    start=time.perf_counter();actual=m._v442_refine(x,params,model,kh,d);elapsed=time.perf_counter()-start
    expected=ref._v441_refine(x,params,zero,kh,d)
    assert not validate_hif4_params(actual,x.shape,'442')
    assert equal(actual,expected)
    assert equal(snapshot,(x,params,model)) and torch.equal(rng,torch.get_rng_state())
    if 2<=t<=128:
        p,v=m._v442_fields(x,model,kh);p0,v0=ref._v441_fields(x,zero,kh)
        assert torch.equal(p,p0) and torch.equal(v,v0)
        e=torch.randn(kh,t,d,generator=gen)
        a,b=m._v442_terms(e,p,v);c,dg=ref._v441_terms(e,p0,v0)
        assert torch.equal(a,c) and torch.equal(b,dg)
    report['cases'].append({'t':t,'seconds':elapsed,'changed':int((actual['mant']!=params['mant']).sum())})
report['checks'].append('zero_variance_exact_gradient_curvature_codes_temperature_gqa_boundaries_rng')
samples=[{r:packed_pair(t,256 if r=='q' else 128,gen) for r in 'qkv'} for t in (64,80,96)]
reuse={'v435_content_parent':model}
newfit=m._v442_fit(samples,reuse,qh,kh,64)
oldfit=ref._v441_fit(samples,reuse,qh,kh,64)
for name in ('center','projection','kernel','query_sigma_modes','a430_temperature','a430_active'):
    assert equal(newfit[name],oldfit[name])
assert 'key_error_metric' not in newfit and 'key_error_floor' not in newfit
captured=[];original=m._v442_parent_calibration
def capture(*a):
    r=original(*a);captured.append(copy.deepcopy(r));return r
m._v442_parent_calibration=capture
frozen=copy.deepcopy(samples)
result=m.hif4_calibration_attention(samples,qh,kh,64)
assert not validate_attention_calibration_result(result,'442')
assert equal(captured[0]['q_state'],result['q_state']) and equal(captured[0]['k_state'],result['k_state'])
assert equal(samples,frozen)
forced=dict(captured[0]['v_state'],v442_key_risk=newfit)
before=copy.deepcopy(forced);fresh=load(442,'fresh')
for sample in samples:
    actual=m.hif4_dynamic_quantize_v(*sample['v'],kh,64,forced)
    assert not validate_hif4_params(actual,sample['v'][0].shape,'fresh')
    assert equal(actual,fresh.hif4_dynamic_quantize_v(*sample['v'],kh,64,copy.deepcopy(forced)))
assert equal(forced,before)
report['checks'].append('fit_model_fields_parent_calibration_qk_preserved_fresh_state')
report['complete']=True;(ROOT/'docs/v442-mechanisms.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
