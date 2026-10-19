"""v400/v402 shared pre-joint parent pairing on structural and archived stress banks."""
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
def main(paths=None, report_name='v402-paired-screen.json'):
    torch.set_num_threads(4)
    report = dict(complete=False,online_prediction=None,rows=[],
                  timing='Shared early v400 calibration recorded separately; local timings are not online predictions.',
                  hashes={str(v):hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() for v in (400,402)})
    paths = paths if paths is not None else [('structural','local_fitted_models_attention_feedback_v2',list(range(10,20))),
             ('stress','local_fitted_archive/attention_feedback_v2_before_structural_20260906',[0,4,14,19,24,47,49])]
    for name,path,groups in paths:
        bank = torch.load(ROOT/path/'attn.pt',weights_only=False)
        manifest = json.loads((ROOT/path/'manifest.json').read_text())
        for gi in groups:
            g=bank[gi]; qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            baseline=load_solution(ROOT/'solution/solution_v400.py')
            started=time.perf_counter()
            parent=baseline._joint_parent_calibration(g['calib'],qh,kh,d)
            shared_seconds=time.perf_counter()-started
            row=dict(bank=name,path=path,schema=manifest.get('schema','legacy'),group=gi,key=g['key'],
                     shared_parent_seconds=shared_seconds,versions={})
            for version in (400,402):
                m=load_solution(ROOT/f'solution/solution_v{version}.py'); checks=[0]
                m._joint_parent_calibration=lambda *a:copy.deepcopy(parent)
                raw_metric=m._joint_raw_metric
                metric_activity=dict(calls=0,from_u=0,from_h=0)
                def measured_metric(st):
                    metric_activity['calls']+=1
                    metric_activity['from_h' if 'H' in st else 'from_u']+=1
                    return raw_metric(st)
                m._joint_raw_metric=measured_metric
                fit=m._vk_fit
                model_activity=dict(fits=0,with_modes=0,mode_squared_norm=0.0)
                def measured_fit(*a):
                    result=fit(*a);model_activity['fits']+=1
                    modes=result.get('query_residual_centers',result.get('query_sigma_modes'))
                    if modes is not None:
                        model_activity['with_modes']+=1
                        model_activity['mode_squared_norm']+=float(modes.float().square().sum())
                    return result
                m._vk_fit=measured_fit
                refine=m._vk_refine
                activity=dict(calls=0,changed=0,values=0,changed_values=0,metadata=0,changed_metadata=0)
                def measured(*a):
                    answer=refine(*a)
                    activity['calls']+=1
                    aa,bb=m._deq_params(answer),m._deq_params(a[1])
                    activity['values']+=aa.numel();activity['changed_values']+=int((aa!=bb).sum())
                    for key in ('scale_factor','scale_lv2','scale_lv3'):
                        if key in answer:
                            activity['metadata']+=answer[key].numel();activity['changed_metadata']+=int((answer[key]!=a[1][key]).sum())
                    activity['changed']+=int(not torch.equal(m._deq_params(answer),m._deq_params(a[1])))
                    return answer
                m._vk_refine=measured
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
                                                     debug=dict(getattr(m,'JOINT_DEBUG',{})), activity=dict(activity),model_activity=dict(model_activity),metric_activity=dict(metric_activity))
            report['rows'].append(row)
            print(name,gi,sum(row['versions']['402']['scores'])-sum(row['versions']['400']['scores']),row['versions']['402']['debug'],flush=True)
            (ROOT/'docs'/report_name).write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs'/report_name).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

