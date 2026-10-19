"""Separate reordered intermediate-state loss from final temperature change."""
import copy,contextlib,io,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_attention
torch.set_num_threads(4)
torch.set_grad_enabled(False)
bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
rows=[]
for gi in (48,49):
    group=bank[gi];dims=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    a=load_solution(ROOT/'solution/solution_v381.py');b=load_solution(ROOT/'solution/solution_v382.py')
    early=a._temp_parent_calibration(group['calib'],*dims)
    a._temp_parent_calibration=lambda *args:copy.deepcopy(early)
    b._vm_parent_calibration=lambda *args:copy.deepcopy(early)
    old=a.hif4_calibration_attention(group['calib'],*dims)
    snapshot=[];fit=b._temp_fit
    def capture(samples,state,*args):snapshot.append(copy.deepcopy(state));return fit(samples,state,*args)
    b._temp_fit=capture
    new=b.hif4_calibration_attention(group['calib'],*dims)
    result={}
    for label,state in [('v381',old),('before_late_temp',snapshot[0]),('v382',new)]:
        b.hif4_calibration_attention=lambda *args,_state=state:copy.deepcopy(_state)
        with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(b,[group])
        result[label]=sum(scores)
    rows.append({'group':gi,'scores':result});print(rows[-1],flush=True)
(ROOT/'docs/v382-stage-diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
