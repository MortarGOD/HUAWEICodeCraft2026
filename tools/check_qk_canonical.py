"""Paired and independent-role regression for canonical Q/K coordinates."""
import argparse,copy,contextlib,hashlib,io,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params
from tools.check_role_decoy_calibration import role_samples


def compatibility(m,states,kh,d):
    matrices=[]
    for role in 'qk':
        st={k:v for k,v in states[role+'_state'].items() if k!='temperature_gain'}
        a=m._kt_input(torch.eye(d).repeat(1,kh),st,kh,d,role).reshape(d,kh,d).transpose(0,1)
        matrices.append(a)
    return ((matrices[0]@matrices[1].transpose(1,2)-torch.eye(d)).square().sum((1,2))/d).sqrt().tolist()


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,default=417)
    candidate=parser.parse_args().version
    assert candidate in (417,418)
    path=ROOT/f'docs/v{candidate}-qk-screen.json'
    report=dict(complete=False,online_prediction=None,rows=[],
                warning='Unknown judge decoys. Paired and corruption modes are diagnostics, not online distribution. Top-level linear compatibility excludes temperature and K translation. Timings include comparisons.',
                hashes={str(v):hashlib.sha256((ROOT/f'solution/solution_v{v}.py').read_bytes()).hexdigest() for v in (412,candidate)})
    for bank,folder,indices in [('structural','local_fitted_models_attention_feedback_v2',[10,11]),
                               ('stress','local_fitted_archive/attention_feedback_v2_before_structural_20260906',[0,49])]:
        data=torch.load(ROOT/folder/'attn.pt',weights_only=False)
        for gi in indices:
            g=data[gi];qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
            for mode in ('paired','row_permute','block_resample'):
                states={}
                if mode=='paired':
                    base=load_solution(ROOT/'solution/solution_v412.py')
                    states=base.hif4_calibration_attention(g['calib'],qh,kh,d)
                else:
                    for ri,role in enumerate('qkv'):
                        base=load_solution(ROOT/'solution/solution_v412.py')
                        sample=role_samples(g['calib'],role,mode,20260906+ri)
                        out=base.hif4_calibration_attention(sample,qh,kh,d)
                        assert not validate_attention_calibration_result(out,'parent')
                        states[role+'_state']=copy.deepcopy(out[role+'_state'])
                assert not validate_attention_calibration_result(states,'assembled')
                row=dict(bank=bank,group=gi,mode=mode,versions={})
                for version in (412,candidate):
                    m=load_solution(ROOT/f'solution/solution_v{version}.py');st=copy.deepcopy(states)
                    start=time.perf_counter()
                    if version==candidate:
                        m._vs_parent_calibration=lambda *a:copy.deepcopy(states)
                        st=m.hif4_calibration_attention(g['calib'],qh,kh,d)
                    prepare=time.perf_counter()-start
                    assert not validate_attention_calibration_result(st,'new')
                    compat=compatibility(m,st,kh,d)
                    if version==candidate:assert max(compat)<2e-5
                    m.hif4_calibration_attention=lambda *a:copy.deepcopy(st)
                    changed={r:0 for r in 'qkv'};checks=[0];seconds={r:0. for r in 'qkv'}
                    for role in 'qkv':
                        fresh=load_solution(ROOT/f'solution/solution_v{version}.py')
                        fn=getattr(fresh,'hif4_dynamic_quantize_'+role)
                        ref=load_solution(ROOT/'solution/solution_v412.py');ref_fn=getattr(ref,'hif4_dynamic_quantize_'+role)
                        def checked(*a,_fn=fn,_ref=ref,_rf=ref_fn,_role=role):
                            start=time.perf_counter();out=_fn(*a);seconds[_role]+=time.perf_counter()-start
                            assert not validate_hif4_params(out,a[0].shape,_role);checks[0]+=1
                            old=_rf(*a[:-1],states[_role+'_state'])
                            changed[_role]+=int((_ref._deq_params(old)!=_ref._deq_params(out)).sum())
                            return out
                        setattr(m,'hif4_dynamic_quantize_'+role,checked)
                    with contextlib.redirect_stdout(io.StringIO()):scores,total=evaluate_attention(m,[g])
                    assert changed['v']==0
                    row['versions'][str(version)]=dict(scores=scores,checks=checks[0],changed=changed,
                        compatibility=compat,conversion_seconds_including_copy=prepare,role_dynamic_seconds=seconds,
                        seconds_including_comparisons=total,
                        metric_fallback={r:st[r+'_state'].get('canonical_metric_fallback') for r in 'qk'})
                report['rows'].append(row);print(bank,gi,mode,{v:sum(x['scores']) for v,x in row['versions'].items()},flush=True)
                path.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
