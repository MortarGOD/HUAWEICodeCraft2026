"""Cross-sequence rescue screen, not test-selected deployment."""
import argparse,contextlib,copy,io,json,types,time
import torch
from probe_v319_value import ROOT,load_solution
from joint_cross_sequence_source import candidate_source,frontier_source
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frontier',action='store_true')
    parser.add_argument('--all-groups',action='store_true')
    parser.add_argument('--stable-joint',action='store_true')
    parser.add_argument('--triangular-metric',action='store_true')
    parser.add_argument('--actual-version',type=int,choices=(352,))
    parser.add_argument('--dataset',choices=('feedback','covariance'),default='feedback')
    args=parser.parse_args()
    assert sum((args.frontier,args.stable_joint,args.triangular_metric))<=1
    if args.actual_version:assert args.triangular_metric
    torch.set_num_threads(4)
    dataset='local_fitted_models_attention_feedback_v2' if args.dataset=='feedback' else 'local_fitted_models_covariance_v1'
    bank=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
    path=ROOT/'solution/solution_v324.py'
    baseline=load_solution(path);candidate=types.ModuleType('joint_cross_sequence')
    source=frontier_source(path.read_text()) if args.frontier else candidate_source(path.read_text())
    if args.stable_joint:
        from stable_joint_source import stable_joint_source
        source=stable_joint_source(path.read_text())
    if args.triangular_metric:
        from stable_joint_source import triangular_metric_source
        source=triangular_metric_source(path.read_text())
    if args.actual_version:
        import ast
        actual=ROOT/f'solution/solution_v{args.actual_version}.py'
        assert [ast.dump(x) for x in ast.parse(source).body[1:]]==[ast.dump(x) for x in ast.parse(actual.read_text()).body[1:]]
        candidate=load_solution(actual)
    else:exec(compile(source,'joint_cross_sequence','exec'),candidate.__dict__)
    calibrators=[baseline.hif4_calibration_attention,candidate.hif4_calibration_attention];rows=[]
    groups=range(len(bank)) if args.all_groups else ((0,1,5,14,19,49) if args.dataset=='feedback' else (0,1,24,25,26,49))
    if (args.stable_joint or args.triangular_metric) and not args.all_groups and args.dataset=='feedback':groups=(0,6,14,20,21,49)
    for gi in groups:
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')];variants={}
        for label,m,cal in zip(('base','auxiliary'),(baseline,candidate),calibrators):
            start=time.perf_counter();state=cal(g['calib'],*dims);elapsed=time.perf_counter()-start
            assert not validate_attention_calibration_result(state,'cross_sequence')
            debug=copy.deepcopy(m.JOINT_DEBUG);assert not debug['error'],debug
            m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
            variants[label]={'scores':scores,'joint_debug':debug,'calibration_seconds':elapsed}
        row={'group':gi,'variants':variants,'delta':sum(variants['auxiliary']['scores'])-sum(variants['base']['scores'])}
        rows.append(row);print(row,flush=True)
    label='triangular' if args.triangular_metric else ('stable' if args.stable_joint else ('frontier' if args.frontier else 'cross-sequence'))
    suffix='full' if args.all_groups else 'screen'
    suffix+=('-covariance' if args.dataset=='covariance' else '')
    if args.actual_version:label=f'v{args.actual_version}-production'
    (ROOT/f'docs/joint-{label}-{suffix}.json').write_text(json.dumps({'complete':True,'frontier':args.frontier,'stable_joint':args.stable_joint,'dataset':dataset,'rows':rows,'delta':sum(r['delta'] for r in rows),
        'warning':('Frontier uses only original held-out costs; auxiliary variant uses an in-training sample. Modeled activations; no online forecast.')},indent=2)+'\n')


if __name__=='__main__':main()
