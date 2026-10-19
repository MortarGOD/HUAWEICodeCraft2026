"""Fixed-data paired screening of Key sensitivity weighting."""
import argparse
import contextlib
import copy
import io
import json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def configured():
    m=load_solution(ROOT/'solution/solution_v322.py')
    exec(compile((ROOT/'tools/attention_sensitivity_origin_core.py').read_text(),'sensitivity-research','exec'),m.__dict__)
    return m


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--groups',default='0,4,7,11,14,19,47,48,49')
    parser.add_argument('--modes',default='1,2,3');parser.add_argument('--output',default='docs/v323-sensitivity-screen.json')
    parser.add_argument('--bank',default='local_fitted_models_attention_feedback_v2')
    parser.add_argument('--probability-gate',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/args.bank/'attn.pt',weights_only=False)
    ids=range(len(bank)) if args.groups=='all' else map(int,args.groups.split(','))
    m=configured();m._SS_PROB_GATE=args.probability_gate
    base=load_solution(ROOT/'solution/solution_v322.py');rows=[]
    for gi in ids:
        group=bank[gi];dims=[group[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._ss_parent_calibration(group['calib'],*dims)
        base.hif4_calibration_attention=lambda *unused,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):original,_=evaluate_attention(base,[group])
        row=dict(group=gi,baseline=original,modes={})
        for mode in map(int,args.modes.split(',')):
            m._SS_MODE=mode;state=m._ss_guard(group['calib'],parent,*dims)
            assert not validate_attention_calibration_result(state,'attention')
            m.hif4_calibration_attention=lambda *unused,_st=state:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[group])
            for sample in group['test']:
                out=m.hif4_dynamic_quantize_k(*sample['k'],dims[1],dims[2],state['k_state'])
                assert not validate_hif4_params(out,sample['k'][0].shape,'k')
            row['modes'][mode]=dict(scores=scores,delta=sum(scores)-sum(original),debug=copy.deepcopy(m.SS_DEBUG))
        rows.append(row);print(gi,{k:round(v['delta'],6) for k,v in row['modes'].items()},flush=True)
    result=dict(complete=True,bank=args.bank,rows=rows,score_prediction_allowed=False,probability_gate=args.probability_gate,
                totals={k:sum(r['modes'][k]['delta'] for r in rows) for k in rows[0]['modes']})
    (ROOT/args.output).write_text(json.dumps(result,indent=2)+'\n');print(result['totals'],flush=True)


if __name__=='__main__':main()
