"""Sensitivity hypotheses, not a reconstruction of the unknown judge decoys."""
import copy,contextlib,io,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def role_samples(samples,real_role,mode,seed):
    generator=torch.Generator().manual_seed(seed)
    result=[]
    for sample in samples:
        out={}
        for role in 'qkv':
            q,s=sample[role]
            if role==real_role:
                out[role]=(q.clone(),s.clone())
            elif mode=='row_permute':
                ix=torch.randperm(q.shape[0],generator=generator)
                out[role]=(q[ix].contiguous(),s[ix].contiguous())
            elif mode=='block_resample':
                n,c=q.shape;b=c//16
                ix=torch.randint(n,(n,b),generator=generator);cols=torch.arange(b)[None]
                out[role]=(q.reshape(n,b,16)[ix,cols].reshape(n,c).contiguous(),s.reshape(n,b)[ix,cols].reshape(s.shape).contiguous())
            else:raise ValueError(mode)
        result.append(out)
    return result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    g=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[10]
    qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
    report=dict(complete=False,group=10,bank='local_fitted_models_attention_feedback_v2',
        warning='Role-separated calibration confirmed by user; decoy sources UNKNOWN. Both corruption types below are only sensitivity hypotheses.',rows=[])
    for version in (402,406):
        path=ROOT/f'solution/solution_v{version}.py'
        for mode in ('paired','row_permute','block_resample'):
            states={};debug={};start=time.perf_counter()
            if mode=='paired':
                m=load_solution(path);states=m.hif4_calibration_attention(copy.deepcopy(g['calib']),qh,kh,d)
                debug['paired']=dict(getattr(m,'DO_DEBUG',{}))
            else:
                for ri,role in enumerate('qkv'):
                    m=load_solution(path)
                    samples=role_samples(g['calib'],role,mode,20260906+ri)
                    assert all(torch.equal(s[role][j],t[role][j]) for s,t in zip(samples,g['calib']) for j in (0,1))
                    calibrated=m.hif4_calibration_attention(samples,qh,kh,d)
                    assert not validate_attention_calibration_result(calibrated,'role-calibration')
                    states[role+'_state']=copy.deepcopy(calibrated[role+'_state'])
                    debug[role]=dict(getattr(m,'DO_DEBUG',{}))
            calibration_seconds=time.perf_counter()-start
            assert not validate_attention_calibration_result(states,'assembled')
            m=load_solution(path);m.hif4_calibration_attention=lambda *a:copy.deepcopy(states)
            checks=[0]
            for role in 'qkv':
                role_module=load_solution(path);fn=getattr(role_module,'hif4_dynamic_quantize_'+role)
                def wrapped(*a,_fn=fn,_role=role):
                    result=_fn(*a);assert not validate_hif4_params(result,a[0].shape,_role);checks[0]+=1;return result
                setattr(m,'hif4_dynamic_quantize_'+role,wrapped)
            with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,[g])
            row=dict(version=version,mode=mode,scores=scores,sum=sum(scores),calibration_seconds=calibration_seconds,
                evaluate_seconds=seconds,format_checks=checks[0],debug=debug,
                state_keys={r:sorted(states[r+'_state']) if isinstance(states[r+'_state'],dict) else [] for r in 'qkv'})
            report['rows'].append(row)
            print(version,mode,row['sum'],debug,flush=True)
            (ROOT/'docs/role-decoy-calibration-screen.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs/role-decoy-calibration-screen.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
