"""Same-parent, same accepted-row iteration budget metric ablation."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_attention
from self_check import validate_attention_calibration_result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    report={'complete':False,'groups':[],'online_prediction':None,'same_parent':True}
    for gi in (0,4,14,19,48,49):
        g=bank[gi];qh,kh,d=g['q_num_heads'],g['kv_num_heads'],g['head_dim']
        base=load_solution(ROOT/'solution/solution_v374.py')
        parent=base._cm_parent_calibration(g['calib'],qh,kh,d)
        centers,bankh=base._cm_fit(g['calib'],parent,qh,kh,d)
        old=base._joint_raw_metric(parent['q_state']).float()
        old=old/old.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None].clamp_min(1e-20)
        rows={}
        for mode in ('parent','conditional','merged','original_metric'):
            m=load_solution(ROOT/'solution/solution_v374.py')
            m._cm_parent_calibration=lambda *a:copy.deepcopy(parent)
            if mode=='parent':m.hif4_calibration_attention=m._cm_parent_calibration
            else:
                h=bankh if mode=='conditional' else (bankh.mean(1,keepdim=True).expand_as(bankh) if mode=='merged' else old[:,None].expand_as(bankh))
                m._cm_fit=lambda *a,_h=h:(centers.clone(),_h.clone())
            original=m.hif4_calibration_attention
            seen={}
            def capture(*a):
                st=original(*a)
                assert not validate_attention_calibration_result(st,'ablation')
                seen.update(m.CM_DEBUG)
                return st
            m.hif4_calibration_attention=capture
            with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,[g])
            rows[mode]={'scores':scores,'debug':seen,'seconds':seconds}
        result={'group':gi,'modes':rows};report['groups'].append(result)
        print(gi,{k:sum(v['scores'])-sum(rows['parent']['scores']) for k,v in rows.items()},flush=True)
        (ROOT/'docs/v374-metric-ablation.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs/v374-metric-ablation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
