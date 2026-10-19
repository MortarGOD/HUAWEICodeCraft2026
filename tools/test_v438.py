"""Exact independent quadratic transport, common coordinates, API isolation."""
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
from hif4_quant import dequantize_hif4
from self_check import validate_hif4_params,validate_attention_calibration_result
torch.set_num_threads(4)
m=load(438,'mechanism');parent=load(435,'mechanism')
gen=torch.Generator().manual_seed(438)
records=[]
with torch.inference_mode():
    for d in (64,96,128,512):
        a=torch.eye(d)[None].repeat(2,1,1)+torch.randn(2,d,d,generator=gen)*.003
        u=torch.triu(torch.randn(4,d,d,generator=gen)*.002)+torch.eye(d)
        z=torch.randn(2,d,d,generator=gen)*.01
        h=z@z.transpose(-1,-2)+torch.eye(d)*2
        st={'kvh':2,'joint_transform':a,'u':u,'H':h,'gq':1,'rf':1,
            'qs':torch.rand(2,d,generator=gen)+.8,'temperature_gain':torch.tensor([.9,1.,1.1,1.2])}
        frozen=copy.deepcopy(st);rng=torch.get_rng_state().clone()
        new=m._v438_state(st,'q',2,d)
        c=new['joint_transform'].double()
        am=(torch.diag_embed(st['qs'])@a).double()
        s=torch.linalg.solve(am,c)
        expected=s.repeat_interleave(2,0).transpose(-1,-2)@(u.double().transpose(-1,-2)@u.double())@s.repeat_interleave(2,0)
        got=new['u'].double().transpose(-1,-2)@new['u'].double()
        assert torch.allclose(got,expected,rtol=2e-5,atol=2e-6)
        expected_h=c.transpose(-1,-2)@am@h.double()@am.transpose(-1,-2)@c
        assert torch.allclose(new['H'].double(),expected_h,rtol=2e-5,atol=2e-6)
        x=torch.randn(8,4*d,generator=gen)
        mapped=m._kt_input(x,new,4,d,'q')
        expected_x=((x.reshape(8,4,d).transpose(0,1)@c.repeat_interleave(2,0).float()).transpose(0,1)*st['temperature_gain'][None,:,None]).reshape(x.shape)
        assert torch.allclose(mapped,expected_x,atol=2e-6,rtol=2e-6)
        ks={'kvh':2,'joint_transform':a*.9,'ks':st['qs'],'u':u[:2],'H':h}
        kn=m._v438_state(dict(ks,ts_fallback=ks),'k',2,d)
        assert torch.equal(new['joint_transform'],kn['joint_transform'])
        assert torch.equal(kn['joint_transform'],kn['ts_fallback']['joint_transform'])
        assert torch.allclose(c@c.transpose(-1,-2),torch.eye(d).double(),atol=2e-6,rtol=2e-6)
        assert equal(st,frozen) and torch.equal(rng,torch.get_rng_state())
        assert equal(new,m._v438_state(new,'q',2,d))
        records.append({'d':d,'u_max_error':float((got-expected).abs().max()),'fallback':new['canonical_dual_metric_fallback']})
    singular={'kvh':1,'joint_transform':torch.zeros(1,64,64),'u':torch.eye(64)[None]}
    assert m._v438_state(singular,'q',1,64)['canonical_dual_metric_fallback']
    assert m._v438_state(None,'k',1,64)['canonical_dual_metric']
    samples=[{r:packed_pair(n,128 if r=='q' else 64,gen) for r in 'qkv'} for n in (4,8,16)]
    old=parent.hif4_calibration_attention(samples,2,1,64)
    new=m.hif4_calibration_attention(samples,2,1,64)
    assert not validate_attention_calibration_result(new,'v438')
    assert equal(old['v_state'],new['v_state'])
    for r in 'qk':assert equal(new[r+'_state'],m._v438_state(old[r+'_state'],r,1,64))
    frozen=copy.deepcopy(new);frozen_input=copy.deepcopy(samples);rng=torch.get_rng_state().clone()
    changed={}
    for r in 'qkv':
        fn='hif4_dynamic_quantize_'+r;pair=samples[-1][r];nh=2 if r=='q' else 1
        ans=getattr(m,fn)(*pair,nh,64,new[r+'_state'])
        before=getattr(parent,fn)(*pair,nh,64,old[r+'_state'])
        assert not validate_hif4_params(ans,pair[0].shape,r)
        assert equal(ans,getattr(load(438,'fresh_'+r),fn)(*pair,nh,64,copy.deepcopy(new[r+'_state'])))
        changed[r]=int((dequantize_hif4(ans,pair[0].shape)!=dequantize_hif4(before,pair[0].shape)).sum())
    assert changed['q'] and changed['k'] and not changed['v']
    assert equal(new,frozen) and equal(samples,frozen_input) and torch.equal(rng,torch.get_rng_state())
    for fn in ('hif4_calibration_and_quantize_weight','hif4_dynamic_quantize_activation','hif4_dynamic_quantize_q','hif4_dynamic_quantize_k','hif4_dynamic_quantize_v'):
        assert getattr(m,fn).__code__.co_code==getattr(parent,fn).__code__.co_code
report={'dual_metric_checks':records,'changed_values':changed,'fresh_role_state_input_rng_checks':True}
(ROOT/'docs/v438-mechanisms.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
