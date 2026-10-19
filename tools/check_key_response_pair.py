"""Actual-file v324/v378 comparison with one immutable shared parent per group."""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    report={'complete':False,'same_parent':True,'rows':[],'online_prediction':None,
            'sha256':{str(v):hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() for v in (324,378)}}
    for gi in (0,4,14,19,24,34,47,48,49):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=load_solution(ROOT/'solution/solution_v324.py').hif4_calibration_attention(g['calib'],qh,kh,d)
        versions={}
        for v in (324,378):
            m=load_solution(ROOT/f'solution/solution_v{v}.py');checks=[0]
            if v==324:m.hif4_calibration_attention=lambda *a:copy.deepcopy(parent)
            else:m._kr_parent_calibration=lambda *a:copy.deepcopy(parent)
            original=m.hif4_calibration_attention
            def calibration(*a):
                result=original(*a)
                assert not validate_attention_calibration_result(result,'pair');checks[0]+=1
                return result
            m.hif4_calibration_attention=calibration
            for role in 'qkv':
                fn=getattr(m,'hif4_dynamic_quantize_'+role)
                def dynamic(*a,_fn=fn,_role=role):
                    result=_fn(*a)
                    assert not validate_hif4_params(result,a[0].shape,_role);checks[0]+=1
                    return result
                setattr(m,'hif4_dynamic_quantize_'+role,dynamic)
            with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,[g])
            versions[str(v)]={'scores':scores,'debug':dict(getattr(m,'KR_DEBUG',{})),'checks':checks[0],'seconds':seconds}
        report['rows'].append({'group':gi,'versions':versions})
        print(gi,{v:sum(r['scores'])-sum(versions['324']['scores']) for v,r in versions.items()},flush=True)
        (ROOT/'docs/v378-paired-screen.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs/v378-paired-screen.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

