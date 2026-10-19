"""Paired raw-score screen with the exact same complete v402 parent state."""
import contextlib, copy, hashlib, io, json, sys, time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution, evaluate_attention
from self_check import validate_attention_calibration_result, validate_hif4_params


@torch.inference_mode()
def main(paths=None, report_name='v405-direction-screen.json', candidate=None, same_operator=False):
    torch.set_num_threads(4)
    paths=paths or [('structural','local_fitted_models_attention_feedback_v2',[10,12,16,18]),
                   ('stress','local_fitted_archive/attention_feedback_v2_before_structural_20260906',[0,4,24,49])]
    report=dict(complete=False,online_prediction=None,rows=[],same_operator=same_operator,
                source_hash=hashlib.sha256((ROOT/(candidate or 'tools/dual_operator_core.py')).read_bytes()).hexdigest(),
                parent_hash=hashlib.sha256((ROOT/'solution/solution_v402.py').read_bytes()).hexdigest(),
                timing='Same complete parent calibration is recorded separately; fixed version order and single local run are not an online prediction.')
    for bank,path,groups in paths:
        data=torch.load(ROOT/path/'attn.pt',weights_only=False)
        for gi in groups:
            g=data[gi];qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
            base=load_solution(ROOT/'solution/solution_v402.py')
            start=time.perf_counter();parent=base.hif4_calibration_attention(g['calib'],qh,kh,d)
            shared=time.perf_counter()-start
            row=dict(bank=bank,group=gi,key=g['key'],shared_parent_seconds=shared,versions={})
            for version in ('402','405'):
                m=load_solution(ROOT/'solution/solution_v402.py' if version=='402' or candidate is None else ROOT/candidate)
                if version=='405' and candidate is None:
                    exec(compile((ROOT/'tools/dual_operator_core.py').read_text(),'dual_operator_core.py','exec'),m.__dict__)
                if version=='405' and same_operator:
                    probabilities=m._do_probabilities
                    def without_cross(*a):
                        raw,quant=probabilities(*a)
                        return quant,quant
                    m._do_probabilities=without_cross
                if version=='402':m.hif4_calibration_attention=lambda *a:copy.deepcopy(parent)
                else:m._do_parent_calibration=lambda *a:copy.deepcopy(parent)
                checks=[0];cal=m.hif4_calibration_attention;activity=dict(calls=0,changed_values=0,values=0)
                def wrapped_cal(*a):
                    ans=cal(*a);assert not validate_attention_calibration_result(ans,'dual');checks[0]+=1
                    return ans
                m.hif4_calibration_attention=wrapped_cal
                if version=='405':
                    refine=m._do_refine
                    def observed(*a):
                        ans=refine(*a);activity['calls']+=1
                        aa,bb=m._deq_params(ans),m._deq_params(a[1]);activity['values']+=aa.numel()
                        activity['changed_values']+=int((aa!=bb).sum());return ans
                    m._do_refine=observed
                for role in 'qkv':
                    dynamic=getattr(m,'hif4_dynamic_quantize_'+role)
                    def checked(*a,_fn=dynamic,_role=role):
                        ans=_fn(*a);assert not validate_hif4_params(ans,a[0].shape,_role);checks[0]+=1;return ans
                    setattr(m,'hif4_dynamic_quantize_'+role,checked)
                with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,[g])
                row['versions'][version]=dict(scores=scores,seconds=seconds,checks=checks[0],
                    activity=activity,debug=dict(getattr(m,'DO_DEBUG',{})))
            report['rows'].append(row)
            delta=sum(row['versions']['405']['scores'])-sum(row['versions']['402']['scores'])
            print(bank,gi,delta,row['versions']['405']['debug'],flush=True)
            (ROOT/'docs'/report_name).write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs'/report_name).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
