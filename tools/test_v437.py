"""Synthetic dual-metric transport and public interface checks."""
import copy
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
from self_check import validate_attention_calibration_result,validate_hif4_params
from hif4_quant import dequantize_hif4

torch.set_num_threads(4)
base,m=load(435,'test437'),load(437,'test437')
gen=torch.Generator().manual_seed(437)
record={'checks':{},'transport':[]}

def mapping(st,nh,d,role):
    if st is None:return torch.eye(d)[None].repeat(nh,1,1)
    return m._kt_input(torch.eye(d).repeat(1,nh),st,nh,d,role).reshape(d,nh,d).transpose(0,1)

for d in (64,96,128,256,512):
    r=m._v437_rotation(d)
    assert torch.equal(r@r.T,torch.eye(d))
    z=torch.randn(2,d,d,generator=gen)*.1
    h=z@z.transpose(1,2)+torch.eye(d)
    # H is deliberately unrelated to the GPTQ metric, as in the live Q path.
    u=torch.triu(torch.randn(2,d,d,generator=gen)*.003)+torch.eye(d)
    a=torch.eye(d)[None].repeat(2,1,1)+torch.randn(2,d,d,generator=gen)*.01
    st={'kvh':2,'joint_transform':a,'qs':torch.rand(2,d,generator=gen)+.5,
        'u':u,'H':h,'gq':1,'rf':1,'temperature_gain':torch.tensor([.9,1.1,1.2,1.0])}
    frozen=copy.deepcopy(st);rng=torch.get_rng_state().clone()
    start=time.perf_counter();new=m._v437_state(st,2,d);elapsed=time.perf_counter()-start
    assert equal(st,frozen) and torch.equal(rng,torch.get_rng_state())
    assert torch.allclose(mapping(new,4,d,'q'),mapping(st,4,d,'q')@r,atol=2e-6,rtol=2e-6)
    assert torch.allclose(new['H'],r.T@h@r,atol=3e-5,rtol=3e-5)
    inv=r.T@(u.transpose(-1,-2)@u)@r
    observed=new['u'].transpose(-1,-2)@new['u']
    assert torch.allclose(observed,inv,atol=3e-6,rtol=3e-6)
    assert not new['post_rotation_metric_fallback']
    assert equal(m._v437_state(new,2,d),new)
    kst={'kvh':2,'rot':1,'ks':torch.rand(2,d,generator=gen)+.5,'u':u,'H':h}
    nested=dict(kst,ts_fallback=copy.deepcopy(kst),kt_probe=torch.randn(2,3,d,generator=gen))
    knew=m._v437_state(nested,2,d)
    for old,converted in ((nested,knew),(nested['ts_fallback'],knew['ts_fallback'])):
        assert torch.allclose(mapping(converted,2,d,'k'),mapping(old,2,d,'k')@r,atol=2e-6,rtol=2e-6)
    assert equal(knew['kt_probe'],nested['kt_probe'])
    aq=mapping(st,4,d,'q');bq=mapping(new,4,d,'q')
    ak=mapping(nested,2,d,'k').repeat_interleave(2,0)
    bk=mapping(knew,2,d,'k').repeat_interleave(2,0)
    err=float((aq@ak.transpose(-1,-2)-bq@bk.transpose(-1,-2)).abs().max())
    assert err<5e-6
    record['transport'].append({'d':d,'seconds':elapsed,'parent_product_max_error':err})
assert m._v437_state(None,1,24) is None
none=m._v437_state(None,1,64)
assert torch.equal(none['joint_transform'][0],m._v437_rotation(64))
bad=m._v437_state({'kvh':1,'u':torch.zeros(1,64,64)},1,64)
assert bad['post_rotation_metric_fallback'] and torch.isfinite(bad['u']).all()
record['checks']['independent_H_U_transport_parent_product_nested_boundary_rng']=True

samples=[{r:packed_pair(n,128 if r=='q' else 64,gen) for r in 'qkv'} for n in (4,8,16)]
frozen=copy.deepcopy(samples);rng=torch.get_rng_state().clone()
a=base.hif4_calibration_attention(samples,2,1,64)
b=m.hif4_calibration_attention(samples,2,1,64)
assert not validate_attention_calibration_result(b,'v437')
assert equal(a['v_state'],b['v_state'])
assert equal(m._v437_state(a['q_state'],1,64),b['q_state'])
assert equal(m._v437_state(a['k_state'],1,64),b['k_state'])
saved=copy.deepcopy(b);changed={}
for role in 'qkv':
    fn='hif4_dynamic_quantize_'+role;pair=samples[-1][role]
    args=(*pair,2 if role=='q' else 1,64)
    old=getattr(base,fn)(*args,a[role+'_state'])
    result=getattr(m,fn)(*args,b[role+'_state'])
    assert not validate_hif4_params(result,pair[0].shape,role)
    fresh=load(437,'fresh_'+role)
    assert equal(result,getattr(fresh,fn)(*args,copy.deepcopy(b[role+'_state'])))
    changed[role]=int((dequantize_hif4(old,pair[0].shape)!=dequantize_hif4(result,pair[0].shape)).sum())
    if role=='v':assert equal(old,result)
assert changed['q']>0 and changed['k']>0 and changed['v']==0
assert equal(b,saved) and equal(samples,frozen) and torch.equal(rng,torch.get_rng_state())
for name in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation',
             'hif4_dynamic_quantize_q','hif4_dynamic_quantize_k','hif4_dynamic_quantize_v'):
    assert getattr(m,name).__code__.co_code==getattr(base,name).__code__.co_code
record['checks']['parent_calibration_preserved_V_equal_public_fresh_replay']=True
record['synthetic_changed_values']=changed
(ROOT/'docs/v437-mechanisms.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
