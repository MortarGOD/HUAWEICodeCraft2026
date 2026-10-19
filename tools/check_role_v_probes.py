"""Three role calibrations; compare only final V paths with identical parent states."""
import contextlib,copy,hashlib,io,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params
from tools.check_role_decoy_calibration import role_samples


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    report=dict(complete=False,online_prediction=None,rows=[],
        warning='Unknown judge decoy source; modes are sensitivity hypotheses. Shared parent avoids repeated numerical baseline drift. Timings are local, not online forecasts.',
        hashes={str(v):hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() for v in (400,407,408)})
    for bank,path,indices in [('structural','local_fitted_models_attention_feedback_v2',[10,11]),
                              ('stress','local_fitted_archive/attention_feedback_v2_before_structural_20260906',[0,49])]:
        data=torch.load(ROOT/path/'attn.pt',weights_only=False)
        for gi in indices:
            g=data[gi];qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
            for mode in ('row_permute','block_resample'):
                states={};inputs={};start=time.perf_counter()
                for ri,role in enumerate('qkv'):
                    m=load_solution(ROOT/'solution/solution_v400.py')
                    inputs[role]=role_samples(g['calib'],role,mode,20260906+ri)
                    answer=m.hif4_calibration_attention(inputs[role],qh,kh,d)
                    assert not validate_attention_calibration_result(answer,'parent')
                    states[role+'_state']=copy.deepcopy(answer[role+'_state'])
                shared=time.perf_counter()-start
                row=dict(bank=bank,group=gi,mode=mode,head_dim=d,shared_parent_seconds=shared,versions={})
                for version in (400,407,408):
                    m=load_solution(ROOT/f'solution/solution_v{version}.py');st=copy.deepcopy(states)
                    if version!=400:
                        # Equivalent final replacement, exact same parent states.
                        m._vp_parent_calibration=lambda *a:copy.deepcopy(states)
                        st=m.hif4_calibration_attention(inputs['v'],qh,kh,d)
                    assert not validate_attention_calibration_result(st,'selected')
                    m.hif4_calibration_attention=lambda *a:copy.deepcopy(st)
                    checks=[0];changed=[0];values=[0]
                    for role in 'qkv':
                        fresh=load_solution(ROOT/f'solution/solution_v{version}.py')
                        fn=getattr(fresh,'hif4_dynamic_quantize_'+role)
                        base=load_solution(ROOT/'solution/solution_v400.py')
                        def checked(*a,_fn=fn,_role=role,_base=base):
                            out=_fn(*a);assert not validate_hif4_params(out,a[0].shape,_role);checks[0]+=1
                            if _role=='v':
                                before=_base.hif4_dynamic_quantize_v(*a[:-1],states['v_state'])
                                aa,bb=_base._deq_params(out),_base._deq_params(before)
                                changed[0]+=int((aa!=bb).sum());values[0]+=aa.numel()
                            return out
                        setattr(m,'hif4_dynamic_quantize_'+role,checked)
                    with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,[g])
                    row['versions'][str(version)]=dict(scores=scores,seconds_including_instrumentation=seconds,
                        checks=checks[0],changed_v_values=changed[0],v_values=values[0])
                report['rows'].append(row)
                print(bank,gi,mode,{v:sum(x['scores']) for v,x in row['versions'].items()},flush=True)
                (ROOT/'docs/v407-v408-role-screen.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs/v407-v408-role-screen.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
