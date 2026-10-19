"""Check proposed calibration query helper against actual dynamic dispatch."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from tools.verify_attention_campaign import equal
m=load(443,'nongptq');torch.set_num_threads(4)

gen=torch.Generator().manual_seed(443)
records=[]
for t in (4,32,129):
 for qh,kh in ((1,1),(4,2)):
  d=64;hold={'q':packed_pair(t,qh*d,gen)}
  for rf in (0,1):
   for rot in (0,1):
    state={'gq':0,'rf':rf,'rot':rot,'rseed':0,'kvh':kh,'temperature_gain':torch.linspace(.8,1.2,qh)}
    if rf:state['H']=torch.eye(d).repeat(kh,1,1)
    frozen=copy.deepcopy((hold,state));ix=torch.linspace(0,t-1,min(t,16)).long()
    actual=m._joint_quant_queries(hold,state,qh,kh,d,ix)
    whole=m._deq_params(m.hif4_dynamic_quantize_q(*hold['q'],qh,d,state))[ix]
    assert equal(actual,whole),(t,qh,kh,rf,rot,float((actual-whole).abs().max()))
    assert equal(frozen,(hold,state))
    records.append([t,qh,kh,rf,rot])
Path('docs/v443-mechanisms.json').write_text(json.dumps({'complete':True,'cases':records,'limits':'Synthetic no-u states, current dynamic Q; not dataset or online gain evidence.'},indent=2)+'\n')
print('passed',len(records))
