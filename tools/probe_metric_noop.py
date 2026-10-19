"""Count exact no-op calibration proposals; no approximate thresholds."""
from pathlib import Path
import json
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    rows=[]
    for gi in (0,24,25,34,47,49):
        g=bank[gi];m=load_solution(ROOT/'solution/solution_v375.py');seen={}
        oldq=m._joint_quant_queries;oldt=m._cm_trial
        def query(*a,**k):
            result=oldq(*a,**k);seen['original']=result.clone();return result
        def trial(*a,**k):
            result=oldt(*a,**k)
            v=m._deq_params(result);original=seen['original']
            seen['equal']=torch.equal(v,original)
            seen['changed']=int((v!=original).sum())
            seen['max_delta']=float((v-original).abs().max())
            return result
        m._joint_quant_queries=query;m._cm_trial=trial
        m.hif4_calibration_attention(g['calib'],g['q_num_heads'],g['kv_num_heads'],g['head_dim'])
        row={'group':gi,**{k:v for k,v in seen.items() if k!='original'},'debug':dict(m.CM_DEBUG)}
        rows.append(row);print(row,flush=True)
    (ROOT/'docs/metric-noop-probe.json').write_text(json.dumps({'complete':True,'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
