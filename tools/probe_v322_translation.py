"""Shared-parent screening; no data generation or online-score fitting."""
import argparse
import contextlib
import copy
import io
import json
import time
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def configured():
    m=load_solution(ROOT/'solution/solution_v321.py')
    exec(compile((ROOT/'tools/attention_translation_search_core.py').read_text(),'translation-research','exec'),m.__dict__)
    return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,1,7,11,14,19,47,48,49')
    p.add_argument('--variants',default='1,2,3');p.add_argument('--output',default='docs/v322-translation-screen.json')
    p.add_argument('--bank',default='local_fitted_models_attention_feedback_v2');args=p.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/args.bank/'attn.pt',weights_only=False)
    indices=list(range(len(bank))) if args.groups=='all' else [int(x) for x in args.groups.split(',')]
    m=configured();base=load_solution(ROOT/'solution/solution_v321.py');rows=[]
    for gi in indices:
        g=bank[gi];dims=[g[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._ts_parent_calibration(g['calib'],*dims)
        base.hif4_calibration_attention=lambda *unused,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,secs=evaluate_attention(base,[g])
        row=dict(group=gi,base=old,base_seconds=secs,variants={})
        for variant in map(int,args.variants.split(',')):
            m._TS_VARIANT=variant;t=time.perf_counter();st=m._ts_guard(g['calib'],parent,*dims);elapsed=time.perf_counter()-t
            assert not validate_attention_calibration_result(st,'attention')
            m.hif4_calibration_attention=lambda *unused,_st=st:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,secs=evaluate_attention(m,[g])
            for sample in g['test']:
                assert not validate_hif4_params(m.hif4_dynamic_quantize_k(*sample['k'],dims[1],dims[2],st['k_state']),sample['k'][0].shape,'key')
            row['variants'][variant]=dict(scores=scores,delta=sum(scores)-sum(old),seconds=secs,guard_seconds=elapsed,debug=copy.deepcopy(m.TS_DEBUG))
        rows.append(row);print(gi,{k:round(v['delta'],6) for k,v in row['variants'].items()},flush=True)
    report=dict(complete=True,bank=args.bank,rows=rows,score_prediction_allowed=False,
                totals={v:sum(r['variants'][v]['delta'] for r in rows) for v in rows[0]['variants']})
    (ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n');print(report['totals'],flush=True)


if __name__=='__main__':main()
