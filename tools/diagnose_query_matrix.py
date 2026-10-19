"""Separate continuous gradient gain from full quantization reconstruction."""
import sys,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
torch.set_num_threads(4)
torch.set_grad_enabled(False)
m=load_solution(ROOT/'solution/solution_v379.py')
bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
report=[]
for gi in (0,14,47,49):
    group=bank[gi];qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    states=m._qm_parent_calibration(group['calib'],qh,kh,d);st=states['q_state']
    data=[m._qm_data(s,states,qh,kh,d,16) for s in group['calib'][:-1][-2:]]
    eye=torch.eye(d).expand(kh,-1,-1).clone()
    norm,gradient,_=m._qm_eval(data,st,eye,qh,kh,d);norm=norm.clamp_min(1e-20)
    direction=(gradient/norm[:,:,None,None]).mean(0)
    direction/=direction.square().sum((-1,-2)).sqrt()[:,None,None].clamp_min(1e-20)
    row={'group':gi,'rates':[]}
    for rate in (.02,.005,.001,.0002):
        delta=-rate*d**.5*direction
        actual,_,_=m._qm_eval(data,st,eye+delta,qh,kh,d)
        continuous=[]
        for item in data:
            base=m._joint_quant_queries(item['sample'],st,qh,kh,d,item['ix']).reshape(-1,qh,d).transpose(0,1)
            q=base+item['x']@delta.repeat_interleave(qh//kh,0)
            loss,_,_=m._qm_loss_gradient(q,item['k'],item['v'],item['target'])
            continuous.append(loss.reshape(kh,qh//kh).mean(-1))
        row['rates'].append({'rate':rate,'continuous':(torch.stack(continuous)/norm).mean(0).tolist(),
                             'full_quantized':(actual/norm).mean(0).tolist()})
    report.append(row);print(row,flush=True)
    (ROOT/'docs/v379-gradient-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
