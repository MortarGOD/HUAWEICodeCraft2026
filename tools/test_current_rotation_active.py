"""Force a nonidentity candidate to cover deployment even when guards reject."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_attention_calibration_result,validate_hif4_params
torch.set_num_threads(4)
torch.set_grad_enabled(False)
m=load_solution(ROOT/'solution/solution_v380.py')
group=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[0]
qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
parent=m._lr_parent_calibration(group['calib'],qh,kh,d)
torch.manual_seed(810)
rotation=m._lr_cayley(torch.eye(d).expand(kh,-1,-1),torch.randn(kh,d,d),.03)
trial=m._lr_states(parent,rotation,qh,kh,d)
assert not validate_attention_calibration_result(trial,'active')
changed=0;checks=0
for sample in group['test']:
    for role,nh in [('q',qh),('k',kh)]:
        fn=getattr(m,'hif4_dynamic_quantize_'+role)
        a=fn(*sample[role],nh,d,parent[role+'_state'])
        b=fn(*sample[role],nh,d,trial[role+'_state'])
        assert not validate_hif4_params(b,sample[role][0].shape,role)
        checks+=1;changed+=int(not torch.equal(m._deq_params(a),m._deq_params(b)))
assert changed==checks
print('PASS: forced shared rotation changes all',checks,'Q/K outputs and preserves format/state validity')
