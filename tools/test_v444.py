import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
from self_check import validate_hif4_params,validate_attention_calibration_result
m=load(444,'test');torch.set_num_threads(4);gen=torch.Generator().manual_seed(444)
records=[]
with torch.inference_mode():
 for t in (1,2,7,32,128,1024,4097):
  q,s=packed_pair(t,128,gen);x=m.dequantize_nvfp4(q,s).float();p=m._dyn_table(x,None,False)
  model={'active':torch.tensor([True,False]),'query_heads':4};frozen=copy.deepcopy((q,s,p,model))
  out=m._v444_refine(x,p,model,2,64)
  assert not validate_hif4_params(out,x.shape,'444')
  a=(x-m._deq_params(p)).reshape(t,2,64).transpose(0,1).double()
  b=(x-m._deq_params(out)).reshape(t,2,64).transpose(0,1).double()
  objective=lambda e:e.sum(1).square()+e.square().sum(1)
  assert bool((objective(b)<=objective(a)+1e-12).all())
  assert bool((b.square().sum(1)<=a.square().sum(1).clamp_min(1e-30)*1.05+1e-12).all())
  assert torch.equal(a[1],b[1]) and equal(frozen,(q,s,p,model))
  if t in (1,4097):assert equal(out,p)
  if t<=128:
   g,c=m._v444_terms(a)
   P=torch.full((2,3,t),1/t,dtype=torch.float64)
   expect=P.transpose(1,2)@(P@a)/3+a/t**2
   assert torch.allclose(g,expect,atol=1e-12,rtol=1e-12)
   assert torch.allclose(c.double(),torch.full_like(c.double(),2/t**2),rtol=1e-6)
  records.append({'t':t,'changed':int((out['mant']!=p['mant']).sum())})
 samples=[{r:packed_pair(t,256 if r=='q' else 128,gen) for r in 'qkv'} for t in (8,16,32)]
 frozen=copy.deepcopy(samples);state=m.hif4_calibration_attention(samples,4,2,64)
 assert not validate_attention_calibration_result(state,'444')
 assert equal(frozen,samples)
 forced=dict(state['v_state'] or {},v444_key_risk={'active':torch.ones(2,dtype=torch.bool),'query_heads':4})
 original=copy.deepcopy(forced)
 p=m.hif4_dynamic_quantize_v(*samples[-1]['v'],2,64,forced)
 fresh=load(444,'fresh');assert equal(p,fresh.hif4_dynamic_quantize_v(*samples[-1]['v'],2,64,copy.deepcopy(forced)))
 assert equal(original,forced)
(ROOT/'docs/v444-mechanisms.json').write_text(json.dumps({'complete':True,'cases':records,'checks':['uniform_gradient','final_risk_sse','inactive_boundaries','public_calibration','state_fresh_immutable']},indent=2)+'\n')
print('passed')
