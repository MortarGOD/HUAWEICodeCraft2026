"""v387/v398 exact final-parent pairing on structural and archived stress banks."""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    report = dict(complete=False,online_prediction=None,rows=[],
                  timing='Shared full v387 calibration recorded separately; local timings are not online predictions.',
                  hashes={str(v):hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() for v in (387,398)})
    paths = [('structural','local_fitted_models_attention_feedback_v2',list(range(10,20))),
             ('stress','local_fitted_archive/attention_feedback_v2_before_structural_20260906',[0,4,14,19,24,47,49])]
    for name,path,groups in paths:
        bank = torch.load(ROOT/path/'attn.pt',weights_only=False)
        manifest = json.loads((ROOT/path/'manifest.json').read_text())
        for gi in groups:
            g=bank[gi]; qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            baseline=load_solution(ROOT/'solution/solution_v387.py')
            started=time.perf_counter()
            parent=baseline.hif4_calibration_attention(g['calib'],qh,kh,d)
            shared_seconds=time.perf_counter()-started
            row=dict(bank=name,path=path,schema=manifest.get('schema','legacy'),group=gi,key=g['key'],
                     shared_parent_seconds=shared_seconds,versions={})
            for version in (387,398):
                m=load_solution(ROOT/f'solution/solution_v{version}.py'); checks=[0]
                if version == 387:m.hif4_calibration_attention=lambda *a:copy.deepcopy(parent)
                else:m._qa_parent_calibration=lambda *a:copy.deepcopy(parent)
                original=m.hif4_calibration_attention
                def calibration(*a):
                    result=original(*a)
                    assert not validate_attention_calibration_result(result,'pair'); checks[0]+=1
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
                row['versions'][str(version)] = dict(scores=scores,seconds=seconds,checks=checks[0],
                                                     debug=dict(getattr(m,'QA_DEBUG',{})))
            report['rows'].append(row)
            print(name,gi,sum(row['versions']['398']['scores'])-sum(row['versions']['387']['scores']),row['versions']['398']['debug'],flush=True)
            (ROOT/'docs/v398-paired-screen.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs/v398-paired-screen.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()



