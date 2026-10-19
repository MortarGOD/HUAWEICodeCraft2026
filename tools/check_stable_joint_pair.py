"""Same-parent stability, dtype and transform-pair checks."""
import argparse,copy,json,time
import torch
from probe_v319_value import ROOT,load_solution
from stable_joint_source import stable_joint_source
from probe_linear_deployed_joint import tree_equal
from self_check import validate_attention_calibration_result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trace-errors',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    g=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[6]
    m=load_solution(ROOT/'solution/solution_v324.py');original=m._joint_balanced_states;capture={}
    def record(states,*dims):
        capture['state']=copy.deepcopy(states);capture['dims']=dims
        return original(states,*dims)
    m._joint_balanced_states=record
    m.hif4_calibration_attention(g['calib'],*[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')])
    source=stable_joint_source((ROOT/'solution/solution_v324.py').read_text())
    begin=source.index('def _joint_balanced_states(');end=source.index('\ndef _joint_quant_queries(',begin)
    exec(compile(source[begin:end],'double_joint','exec'),m.__dict__)
    stable=m._joint_balanced_states;state=capture['state'];snapshot=copy.deepcopy(state);rows=[];first=None
    if args.trace_errors:
        original_source=(ROOT/'solution/solution_v324.py').read_text()
        start=original_source.index('def _upper_cholesky_inv(')
        end=original_source.index('\ndef ',start+4)
        block=original_source[start:end]
        assert block.count('except Exception:')==1
        block=block.replace('    try:\n        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hd))\n        return torch.linalg.cholesky(Hinv, upper=True)',
            '    stage="first_cholesky"\n    input_copy=Hd.clone()\n    try:\n        factor=torch.linalg.cholesky(Hd)\n        stage="inverse"\n        Hinv=torch.cholesky_inverse(factor)\n        stage="second_cholesky"\n        return torch.linalg.cholesky(Hinv,upper=True)')
        block=block.replace('except Exception:','except Exception as exc:\n        _STABILITY_ERRORS.append(dict(stage=stage,type=type(exc).__name__,message=str(exc),dtype=str(H.dtype),finite=bool(torch.isfinite(H).all()),input_modified=not torch.equal(Hd,input_copy),damped_min_eigenvalue=float(torch.linalg.eigvalsh(input_copy.double()).amin()),input_stride=list(Hd.stride())))')
        m._STABILITY_ERRORS=[]
        exec(compile(block,'traced_cholesky','exec'),m.__dict__)
    for repeat in range(8):
        row={'repeat':repeat}
        for name,fn in (('float',original),('double',stable)):
            if args.trace_errors:m._STABILITY_ERRORS.clear()
            start=time.perf_counter();out=fn(state,*capture['dims']);elapsed=time.perf_counter()-start
            row[name]={'success':out is not None,'seconds':elapsed}
            if args.trace_errors:row[name]['exceptions']=copy.deepcopy(m._STABILITY_ERRORS)
            assert tree_equal(state,snapshot)
            if out is not None:
                assert not validate_attention_calibration_result(out,'pair')
                q=out['q_state']['joint_transform'].double();k=out['k_state']['joint_transform'].double()
                error=float((q@k.transpose(-1,-2)-torch.eye(q.shape[-1])).abs().max())
                row[name]['inverse_pair_error']=error;assert error<1e-4
                assert out['q_state']['u'].dtype==torch.float32 and out['k_state']['u'].dtype==torch.float32
                if name=='double':
                    if first is None:first=copy.deepcopy(out)
                    row[name]['same_as_first_success']=tree_equal(first,out)
        rows.append(row)
    report={'complete':True,'group':6,'same_parent':True,'rows':rows}
    suffix='-input-check' if args.trace_errors else ''
    (ROOT/f'docs/joint-stable-paired-check{suffix}.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
