"""Observable calibration features; test scores are not routing inputs."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4);torch.set_grad_enabled(False)
bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
rows=[]
for gi in (0,4,14,19,24,34,47,48,49):
    group=bank[gi];dims=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    m=load_solution(ROOT/'solution/solution_v381.py')
    state=m.hif4_calibration_attention(group['calib'],*dims)
    gain=state['q_state'].get('temperature_gain',torch.ones(dims[0]))
    row={'group':gi,'dimensions':dims,'early_temperature_heads':int((gain!=1).sum()),
         'max_gain_delta':float((gain-1).abs().max()),'gain':gain.tolist()}
    rows.append(row);print(row,flush=True)
(ROOT/'docs/temperature-risk-profile.json').write_text(json.dumps(rows,indent=2)+'\n')
