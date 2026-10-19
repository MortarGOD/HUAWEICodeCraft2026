"""Locate first differing intermediate for repeated identical parent inputs."""
import argparse,copy,json
import torch
from probe_v319_value import ROOT,load_solution


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--triangular',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    g=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[6]
    m=load_solution(ROOT/'solution/solution_v324.py');original=m._joint_balanced_states;captured={}
    def capture(st,*dims):
        captured['state']=copy.deepcopy(st);captured['dims']=dims
        return original(st,*dims)
    m._joint_balanced_states=capture
    m.hif4_calibration_attention(g['calib'],*[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')])
    if args.triangular:
        from stable_joint_source import triangular_metric_source
        changed=triangular_metric_source((ROOT/'solution/solution_v324.py').read_text())
        a=changed.index('def _joint_raw_metric(');b=changed.index('\ndef _joint_balanced_states(',a)
        exec(compile(changed[a:b],'triangular_metric','exec'),m.__dict__)
    source=(ROOT/'solution/solution_v324.py').read_text();start=source.index('def _joint_balanced_states(');end=source.index('\ndef _joint_quant_queries(',start)
    block=source[start:end];lines=block.splitlines();output=[]
    for line in lines:
        output.append(line)
        if line.startswith('    ') and not line.startswith('        ') and ' = ' in line:
            name=line.strip().split(' = ')[0]
            if name in ('hk','hq','a','ar','ai','mm','t','ti','gq','gk'):
                output.append(f'    _INTERMEDIATES.append(({name!r}, {name}.clone()))')
    m._INTERMEDIATES=[];exec(compile('\n'.join(output),'trace_joint','exec'),m.__dict__)
    reference=None;rows=[]
    for repeat in range(10):
        m._INTERMEDIATES.clear();result=m._joint_balanced_states(captured['state'],*captured['dims'])
        current=list(m._INTERMEDIATES)
        if reference is None:reference=current
        diffs=[]
        for (name,a),(other,b) in zip(reference,current):
            assert name==other
            if not torch.equal(a,b):diffs.append({'name':name,'max_abs_delta':float((a-b).abs().max())})
        rows.append({'repeat':repeat,'success':result is not None,'differences':diffs})
    report={'group':6,'rows':rows,'warning':'Instrumented local CPU execution, not a universal backend diagnosis.'}
    suffix='-triangular' if args.triangular else ''
    (ROOT/f'docs/joint-intermediate-repeat{suffix}.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
