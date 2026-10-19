"""Role-separated regression plus explicitly forced VK mechanism coverage."""
import argparse,contextlib,copy,hashlib,io,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params
from tools.check_role_decoy_calibration import role_samples


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,default=413)
    version=parser.parse_args().version
    torch.set_num_threads(4)
    outpath=ROOT/f'docs/v{version}-vk-screen.json'
    report=dict(complete=False,online_prediction=None,rows=[],
        warning='Decoy source unknown. Forced VK bypasses calibration guard only in this harness; it is a mechanism test, not an online-distribution reconstruction. Timing includes extra comparisons.',
        hashes={str(v):hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() for v in (412,version)})
    for bank,path,indices in [('structural','local_fitted_models_attention_feedback_v2',[10,11]),
                              ('stress','local_fitted_archive/attention_feedback_v2_before_structural_20260906',[0,49])]:
        data=torch.load(ROOT/path/'attn.pt',weights_only=False)
        for gi in indices:
            group=data[gi];qh,kh,d=(group[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
            for mode in ('row_permute','block_resample'):
                states={};inputs={}
                for ri,role in enumerate('qkv'):
                    base=load_solution(ROOT/'solution/solution_v412.py')
                    inputs[role]=role_samples(group['calib'],role,mode,20260906+ri)
                    answer=base.hif4_calibration_attention(inputs[role],qh,kh,d)
                    assert not validate_attention_calibration_result(answer,'parent')
                    states[role+'_state']=copy.deepcopy(answer[role+'_state'])
                forced=base._vk_fit(inputs['v'],qh,kh,d)
                for force in (False,True):
                    selected=copy.deepcopy(states)
                    if force:selected['v_state']=copy.deepcopy(forced)
                    row=dict(bank=bank,group=gi,mode=mode,forced_vk=force,versions={})
                    for v in (412,version):
                        prepare_start=time.perf_counter()
                        m=load_solution(ROOT/f'solution/solution_v{v}.py');st=copy.deepcopy(selected)
                        if v!=412:
                            m._vs_parent_calibration=lambda *a:copy.deepcopy(selected)
                            st=m.hif4_calibration_attention(inputs['v'],qh,kh,d)
                        assert not validate_attention_calibration_result(st,'assembled')
                        prepare_seconds=time.perf_counter()-prepare_start
                        m.hif4_calibration_attention=lambda *a:copy.deepcopy(st)
                        checks=[0];changed=[0];dynamic_time=[0.]
                        for role in 'qkv':
                            fresh=load_solution(ROOT/f'solution/solution_v{v}.py')
                            fn=getattr(fresh,'hif4_dynamic_quantize_'+role)
                            reference=load_solution(ROOT/'solution/solution_v412.py')
                            def checked(*a,_fn=fn,_role=role,_ref=reference):
                                start=time.perf_counter();result=_fn(*a)
                                if _role=='v':dynamic_time[0]+=time.perf_counter()-start
                                assert not validate_hif4_params(result,a[0].shape,_role);checks[0]+=1
                                if _role=='v':
                                    original=_ref.hif4_dynamic_quantize_v(*a[:-1],selected['v_state'])
                                    changed[0]+=int((_ref._deq_params(original)!=_ref._deq_params(result)).sum())
                                return result
                            setattr(m,'hif4_dynamic_quantize_'+role,checked)
                        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,[group])
                        row['versions'][str(v)]=dict(scores=scores,checks=checks[0],changed_v_values=changed[0],
                            seconds_including_comparisons=seconds,local_v_dynamic_seconds=dynamic_time[0],
                            state_prepare_seconds_including_load_copy_validate=prepare_seconds)
                    report['rows'].append(row)
                    print(bank,gi,mode,'forced',force,{v:sum(x['scores']) for v,x in row['versions'].items()},flush=True)
                    outpath.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;outpath.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
