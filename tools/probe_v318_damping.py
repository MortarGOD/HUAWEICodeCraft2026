"""Fixed-bank paired-parent damping screen, isolated from metadata prototype."""
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
from self_check import validate_frozen_state


def configured(affine=False):
    m=load_solution(ROOT/'solution/solution_v317.py')
    filename='attention_value_affine_core.py' if affine else 'attention_damping_core.py'
    exec(compile((ROOT/'tools'/filename).read_text(),'damping-research','exec'),m.__dict__)
    if affine:
        m._damp_parent_calibration=m._va_parent_calibration
        m._damp_guard=m._va_guard
        m.DAMP_DEBUG=m.VA_DEBUG
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,19,47,49')
    p.add_argument('--output',default='docs/v318-damping-screen.json')
    p.add_argument('--affine',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    indices=list(range(len(bank))) if a.groups=='all' else list(map(int,a.groups.split(',')))
    m=configured(a.affine); report=dict(dataset=a.dataset,groups=indices,paired_parent=True,results=[],complete=False,affine=a.affine,
        warning='Checkpoint projection plus modeled hidden states, not a calibrated online forecast.')
    for gi in indices:
        g=bank[gi];args=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        t=time.perf_counter();parent=m._damp_parent_calibration(g['calib'],*args);pt=time.perf_counter()-t
        t=time.perf_counter();candidate=m._damp_guard(g['calib'],parent,*args);ct=time.perf_counter()-t
        assert not validate_frozen_state(candidate,'candidate')
        item=dict(group=gi,parent_calibration_seconds=pt,extra_calibration_seconds=ct,diagnostics=copy.deepcopy(m.DAMP_DEBUG),modes={})
        for name,st in (('parent',parent),('candidate',candidate)):
            m.hif4_calibration_attention=lambda *unused,_st=st:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,[g])
            item['modes'][name]=dict(score=sum(scores),cases=scores,seconds=seconds)
        item['delta']=item['modes']['candidate']['score']-item['modes']['parent']['score']
        report['results'].append(item)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(gi,item['delta'],pt,ct,item['diagnostics'].get('choices',item['diagnostics'].get('choice')),flush=True)
    report.update(complete=True,delta=sum(g['delta'] for g in report['results']))
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print('TOTAL',report['delta'],flush=True)


if __name__=='__main__':main()
