"""Paired deployable output-residual query selection screen."""
import argparse,contextlib,copy,io,json,time
import torch
from probe_v319_value import ROOT,load_solution
from output_risk_factor_source import output_risk_source,output_risk_weight_source
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--weighted',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);base=load_solution(ROOT/'solution/solution_v353.py');trial=load_solution(ROOT/'solution/solution_v353.py')
    source=output_risk_weight_source() if args.weighted else output_risk_source()
    exec(compile(source,'output_risk_factor','exec'),trial.__dict__)
    rows=[]
    for dataset,groups in (('local_fitted_models_attention_feedback_v2',(0,1,5,14,19,49)),
                           ('local_fitted_models_covariance_v1',(0,1,24,25,26,49))):
        bank=torch.load(ROOT/dataset/'attn.pt',weights_only=False)
        for gi in groups:
            g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')];versions={}
            for label,m in (('base',base),('risk',trial)):
                cal=m.hif4_calibration_attention;start=time.perf_counter();state=cal(g['calib'],*dims);seconds=time.perf_counter()-start
                assert not validate_attention_calibration_result(state,label)
                m.hif4_calibration_attention=lambda *_,_st=state:copy.deepcopy(_st)
                with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
                m.hif4_calibration_attention=cal;versions[label]={'scores':scores,'calibration_seconds':seconds}
            row={'dataset':dataset,'group':gi,'versions':versions,'delta':sum(versions['risk']['scores'])-sum(versions['base']['scores'])}
            rows.append(row);print(row,flush=True)
    report={'complete':True,'rows':rows,'delta':sum(r['delta'] for r in rows),
            'warning':'Modeled activations and repeated research set; no online or full-runtime claim.'}
    suffix='-weighted' if args.weighted else ''
    (ROOT/f'docs/output-risk-query-screen{suffix}.json').write_text(json.dumps(report,indent=2)+'\n');print({'delta':report['delta']})


if __name__=='__main__':main()
