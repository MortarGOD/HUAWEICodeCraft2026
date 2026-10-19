"""Check reuse on the exact real-projection groups with unpaired run drift."""
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_persistent_covariance import load


def main():
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    report=dict(passed=False,groups=[],warning='Fixed-parent context audit; separate-run parent numerical variability is not corrected or attributed to online.')
    for index in (6,47):
        m=load(316);g=bank[index]
        qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        states,ctx=m._frame_calibrate_parent(g['calib'],qh,kh,d)
        assert ctx
        actual=m._joint_quant_queries(g['calib'][-1],states['q_state'],qh,kh,d,ctx['ix'])
        error=float((actual-ctx['q']).abs().max())
        assert error==0,error
        m._frame_guard(g['calib'],states,qh,kh,d)
        slow=copy.deepcopy(m.FRAME_DEBUG)
        m._frame_guard(g['calib'],states,qh,kh,d,ctx)
        fast=copy.deepcopy(m.FRAME_DEBUG)
        assert slow==fast
        report['groups'].append(dict(group=index,query_max_error=error,choices=fast['choice'],costs_exact=True))
    report['passed']=True
    (ROOT/'docs/v316-real-context-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
