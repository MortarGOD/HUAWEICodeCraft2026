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


def configured():
    m=load_solution(ROOT/'solution/solution_v318.py')
    exec(compile((ROOT/'tools/attention_strata_core.py').read_text(),'strata-research','exec'),m.__dict__)
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,19,47,49')
    p.add_argument('--bins',default='4,8,16')
    p.add_argument('--output',default='docs/v319-strata-screen.json')
    a=p.parse_args();torch.set_num_threads(4)
    bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    indices=list(range(len(bank))) if a.groups=='all' else list(map(int,a.groups.split(',')))
    report=dict(dataset=a.dataset,groups=indices,results=[],complete=False,warning='Diagnostic modeled activations, not an online predictor.')
    m=configured()
    report['configuration']={key:getattr(m,key) for key in ('_VM_VARIANCE','_VM_INNER_ROUNDS','_VM_CAP') if hasattr(m,key)}
    for gi in indices:
        g=bank[gi];args=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        t=time.perf_counter();parent=m._strata_parent_calibration(g['calib'],*args)
        item=dict(group=gi,parent_calibration_seconds=time.perf_counter()-t,modes={})
        variants=[('parent',parent)]
        for bins in map(int,a.bins.split(',')):
            m._STRATA_BINS=bins;t=time.perf_counter()
            candidate=m._strata_guard(g['calib'],parent,*args)
            assert not validate_frozen_state(candidate,'strata')
            item['modes'][str(bins)]=dict(extra_calibration_seconds=time.perf_counter()-t,diagnostics=copy.deepcopy(m.STRATA_DEBUG))
            variants.append((str(bins),candidate))
        for name,st in variants:
            m.hif4_calibration_attention=lambda *unused,_st=st:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,[g])
            item['modes'].setdefault(name,{}).update(score=sum(scores),cases=scores,seconds=seconds)
        for name in a.bins.split(','):
            item['modes'][name]['delta']=item['modes'][name]['score']-item['modes']['parent']['score']
        report['results'].append(item)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(gi,{k:v.get('delta',0) for k,v in item['modes'].items()},flush=True)
    report.update(complete=True,totals={name:sum(r['modes'][name]['delta'] for r in report['results']) for name in a.bins.split(',')})
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print(report['totals'],flush=True)


if __name__=='__main__':main()
