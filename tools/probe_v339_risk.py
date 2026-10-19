"""Shared-parent screening of learned native-block output-risk ranking."""
import argparse,contextlib,copy,io,json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,1,5,14,19,49')
    p.add_argument('--bank',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--output',default='docs/v339-risk-screen.json')
    p.add_argument('--contexts',default='False,True')
    p.add_argument('--oracle-diagnostic',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/a.bank/'attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v324.py')
    exec(compile((ROOT/'tools/attention_risk_rank_core.py').read_text(),'risk-rank','exec'),m.__dict__)
    m._RK_ORACLE_DIAGNOSTIC=a.oracle_diagnostic
    rows=[]
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._rk_parent_calibration(g['calib'],*dims)
        m.hif4_calibration_attention=lambda *_,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(m,[g])
        row=dict(group=gi,baseline=old,modes={})
        for token in a.contexts.split(','):
            context={'False':False,'True':True,'inverse':'inverse'}[token]
            m._RK_CONTEXT=context
            st=m._rk_guard(g['calib'],parent,*dims)
            assert not validate_attention_calibration_result(st,'risk')
            m.hif4_calibration_attention=lambda *_,_st=st:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):score,_=evaluate_attention(m,[g])
            for s in g['test']:
                out=m.hif4_dynamic_quantize_v(*s['v'],dims[1],dims[2],st['v_state'])
                assert not validate_hif4_params(out,s['v'][0].shape,'v')
            row['modes'][str(context)]=dict(delta=sum(score)-sum(old),scores=score,debug=copy.deepcopy(m.RK_DEBUG))
        rows.append(row);print(gi,{k:(round(v['delta'],6),v['debug']['adopted']) for k,v in row['modes'].items()},flush=True)
    report=dict(complete=True,bank=a.bank,rows=rows,totals={k:sum(r['modes'][k]['delta'] for r in rows) for k in rows[0]['modes']},
                warning='Modeled activation bank. No online score prediction; held sequence also used by parent calibration.')
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print(report['totals'],flush=True)


if __name__=='__main__':main()
