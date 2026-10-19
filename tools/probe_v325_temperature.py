"""Shared-parent final-temperature screen; no online-score extrapolation."""
import argparse,contextlib,copy,io,json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,1,7,11,14,19,47,48,49');p.add_argument('--modes',default='1,2')
    p.add_argument('--bank',default='local_fitted_models_attention_feedback_v2');p.add_argument('--output',default='docs/v325-temperature-screen.json');a=p.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/a.bank/'attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v324.py');base=load_solution(ROOT/'solution/solution_v324.py')
    exec(compile((ROOT/'tools/attention_final_temperature_core.py').read_text(),'final-temperature','exec'),m.__dict__)
    ids=range(len(bank)) if a.groups=='all' else map(int,a.groups.split(','));rows=[]
    for gi in ids:
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._tg_parent_calibration(g['calib'],*dims)
        base.hif4_calibration_attention=lambda *unused,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(base,[g])
        row=dict(group=gi,baseline=old,modes={})
        for mode in map(int,a.modes.split(',')):
            m._TG_MODE=mode;st=m._tg_guard(g['calib'],parent,*dims)
            assert not validate_attention_calibration_result(st,'attention')
            m.hif4_calibration_attention=lambda *unused,_st=st:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
            row['modes'][mode]=dict(scores=scores,delta=sum(scores)-sum(old),debug=copy.deepcopy(m.TG_DEBUG))
        rows.append(row);print(gi,{k:round(v['delta'],6) for k,v in row['modes'].items()},flush=True)
    result=dict(complete=True,bank=a.bank,rows=rows,totals={k:sum(r['modes'][k]['delta'] for r in rows) for k in rows[0]['modes']},score_prediction_allowed=False)
    (ROOT/a.output).write_text(json.dumps(result,indent=2)+'\n');print(result['totals'],flush=True)


if __name__=='__main__':main()
