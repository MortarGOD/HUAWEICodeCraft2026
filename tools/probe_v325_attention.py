"""Shared-parent fixed-data Key-grid screening."""
import argparse,contextlib,copy,io,json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def configured(core='grid'):
    m=load_solution(ROOT/'solution/solution_v324.py')
    name='attention_key_grid_core.py' if core=='grid' else 'attention_key_regression_core.py'
    source=(ROOT/'tools'/name).read_text()
    if core!='grid':source=source.replace('_kc','_kg').replace('_KC','_KG').replace('KC_DEBUG','KG_DEBUG')
    exec(compile(source,'key-research','exec'),m.__dict__)
    return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,1,7,11,14,19,47,48,49');p.add_argument('--modes',default='1,2')
    p.add_argument('--core',choices=['grid','regression'],default='grid')
    p.add_argument('--bank',default='local_fitted_models_attention_feedback_v2');p.add_argument('--output',default='docs/v325-key-grid-screen.json');args=p.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/args.bank/'attn.pt',weights_only=False)
    ids=range(len(bank)) if args.groups=='all' else map(int,args.groups.split(','))
    m=configured(args.core);base=load_solution(ROOT/'solution/solution_v324.py');rows=[]
    for gi in ids:
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._kg_parent_calibration(g['calib'],*dims)
        base.hif4_calibration_attention=lambda *unused,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(base,[g])
        row=dict(group=gi,baseline=old,modes={})
        for mode in map(int,args.modes.split(',')):
            m._KG_MODE=mode;st=m._kg_guard(g['calib'],parent,*dims)
            assert not validate_attention_calibration_result(st,'attention')
            m.hif4_calibration_attention=lambda *unused,_st=st:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
            for sample in g['test']:
                out=m.hif4_dynamic_quantize_k(*sample['k'],dims[1],dims[2],st['k_state'])
                assert not validate_hif4_params(out,sample['k'][0].shape,'key')
            row['modes'][mode]=dict(scores=scores,delta=sum(scores)-sum(old),debug=copy.deepcopy(m.KG_DEBUG))
        rows.append(row);print(gi,{k:round(v['delta'],6) for k,v in row['modes'].items()},flush=True)
    report=dict(complete=True,bank=args.bank,rows=rows,totals={k:sum(r['modes'][k]['delta'] for r in rows) for k in rows[0]['modes']},score_prediction_allowed=False)
    (ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n');print(report['totals'],flush=True)


if __name__=='__main__':main()
