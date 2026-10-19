"""Repeated candidate construction with an identical captured parent state."""
import argparse,copy,json
import torch
from probe_v319_value import ROOT,load_solution
from probe_linear_deployed_joint import tree_equal


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--double',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    group=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[6]
    m=load_solution(ROOT/'solution/solution_v324.py')
    if args.double:
        from stable_joint_source import stable_joint_source
        source=stable_joint_source((ROOT/'solution/solution_v324.py').read_text())
        begin=source.index('def _joint_balanced_states(');end=source.index('\ndef _joint_quant_queries(',begin)
        exec(compile(source[begin:end],'double_joint','exec'),m.__dict__)
    original=m._joint_balanced_states;captured={}
    def capture(states,*dims):
        captured['states']=copy.deepcopy(states);captured['dims']=dims
        return original(states,*dims)
    m._joint_balanced_states=capture
    m.hif4_calibration_attention(group['calib'],*[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')])
    states=captured['states'];snapshot=copy.deepcopy(states);rows=[];first=None;chol=m._upper_cholesky_inv
    for repeat in range(6):
        calls=[]
        def trace(h):
            result=chol(h)
            calls.append({'shape':list(h.shape),'success':result is not None,
                          'symmetry_error':float((h-h.transpose(-1,-2)).abs().max())})
            return result
        m._upper_cholesky_inv=trace
        result=original(states,*captured['dims'])
        assert tree_equal(states,snapshot)
        if repeat==0:first=copy.deepcopy(result)
        rows.append({'repeat':repeat,'success':result is not None,'same_as_first':tree_equal(first,result),'cholesky':calls})
    report={'group':6,'double':args.double,'rows':rows,'warning':'One captured state, six repeated constructions. Does not prove cross-environment determinism.'}
    suffix='-double' if args.double else ''
    (ROOT/f'docs/joint-candidate-repeat{suffix}.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
