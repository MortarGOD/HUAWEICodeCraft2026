"""Serial ABBA incremental phase timing; no online-time extrapolation."""
import copy,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.verify_attention_campaign import equal
m={v:load(v,'cost') for v in (435,441,442)}
torch.set_num_threads(4)
g=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=True)[0]
qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
with torch.inference_mode():
 base=m[435].hif4_calibration_attention(g['calib'],qh,kh,d);frozen=copy.deepcopy(base)
 rows=[]
 for v in (441,442,442,441):
  start=time.perf_counter();state=getattr(m[v],f'_v{v}_calibrate')(g['calib'],base,qh,kh,d)
  cal=time.perf_counter()-start
  start=time.perf_counter()
  for s in g['test']:m[v].hif4_dynamic_quantize_v(*s['v'],kh,d,state['v_state'])
  dynamic=time.perf_counter()-start
  assert equal(base,frozen)
  rows.append({'version':v,'extra_calibration_seconds':cal,'five_v_seconds':dynamic})
report={'complete':True,'group':0,'rows':rows,'limits':'Single shared parent group, serial ABBA, incremental calibration and full V only. No Kunpeng extrapolation.'}
(ROOT/'docs/v442-serial-cost.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
