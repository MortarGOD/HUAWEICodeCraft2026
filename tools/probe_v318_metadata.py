"""Paired-parent fixed-bank screen; no production modification."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution,evaluate_attention
from self_check import validate_frozen_state,validate_hif4_params


def candidate_source():
    source=(ROOT/'solution/solution_v317.py').read_text()
    old='    return _values_to_params(values, p) if values is not None else p'
    assert source.count(old)==1
    source=source.replace(old,'    p = _values_to_params(values, p) if values is not None else p\n    return _meta_refine(x, p, state, num_heads, head_dim, x.shape[0])')
    old='    return _deq_params(_values_to_params(values, p))'
    assert source.count(old)==1
    source=source.replace(old,'    p = _values_to_params(values, p)\n    return _deq_params(_meta_refine(x, p, state, qh, d, rows))')
    return source+'\n\n'+(ROOT/'tools/attention_metadata_core.py').read_text()


def configured():
    module=load_solution(ROOT/'solution/solution_v317.py')
    exec(compile(candidate_source(),'v318-research','exec'),module.__dict__)
    return module


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,19,47,49')
    p.add_argument('--output',default='docs/v318-metadata-screen.json')
    p.add_argument('--no-rerank',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4)
    bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    indices=list(range(len(bank))) if a.groups=='all' else list(map(int,a.groups.split(',')))
    report=dict(dataset=a.dataset,groups=indices,paired_parent=True,results=[],complete=False,
                warning='Checkpoint projection + modeled activations; not a validated online prediction dataset.')
    m=configured();m._META_RERANK=not a.no_rerank
    report['full_hessian_scale_rerank']=m._META_RERANK
    for gi in indices:
        g=bank[gi];args=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        t=time.perf_counter();parent=m._meta_parent_calibration(g['calib'],*args);pt=time.perf_counter()-t
        t=time.perf_counter();candidate=m._meta_guard(g['calib'],parent,*args);ct=time.perf_counter()-t
        assert not validate_frozen_state(candidate,'candidate')
        item=dict(group=gi,parent_calibration_seconds=pt,extra_calibration_seconds=ct,diagnostics=copy.deepcopy(m.META_DEBUG),modes={})
        for name,st in (('parent',parent),('candidate',candidate)):
            m.hif4_calibration_attention=lambda *unused,_st=st: copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):
                scores,seconds=evaluate_attention(m,[g])
            item['modes'][name]=dict(score=sum(scores),cases=scores,seconds=seconds)
        item['delta']=item['modes']['candidate']['score']-item['modes']['parent']['score']
        report['results'].append(item)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(gi,item['delta'],pt,ct,{k:v for k,v in item['diagnostics'].items() if k!='costs'},flush=True)
    report.update(complete=True,delta=sum(g['delta'] for g in report['results']))
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
    print('TOTAL',report['delta'],flush=True)


if __name__=='__main__':main()
