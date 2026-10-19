"""Real-GPTQ-forward calibration rotation search; no production edits."""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_persistent_covariance import load
from evaluate_local import evaluate_attention
from self_check import validate_hif4_params,validate_frozen_state


def configured(rate,steps=4,queries=16,objective='output',block=0,head_cap=1.05):
    m=load(314)
    if rate:
        source=(ROOT/'tools/gptq_covariance_core.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_cov_transform')
        exec(compile(ast.get_source_segment(source,node),'rotation_frame','exec'),m.__dict__)
        exec(compile((ROOT/'tools/learned_rotation_core.py').read_text(),'learned_rotation','exec'),m.__dict__)
        m._LR_RATE=rate;m._LR_STEPS=steps
        m._LR_QUERIES=queries;m._LR_OBJECTIVE=objective;m._LR_BLOCK=block
        m._LR_HEAD_CAP=head_cap
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,49')
    p.add_argument('--rates',default='0,.03,.1')
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--queries',type=int,default=16)
    p.add_argument('--objective',choices=('output','centered_logit'),default='output')
    p.add_argument('--block',type=int,default=0)
    p.add_argument('--head-cap',type=float,default=1.05,help='0 disables only the individual-Q-head cap; both aggregate folds remain required')
    p.add_argument('--output',default='docs/learned-rotation-screen.json')
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(dataset=a.dataset,groups=a.groups,steps=a.steps,queries=a.queries,objective=a.objective,block=a.block,head_cap=a.head_cap,results={},warning='STE gives an approximate direction; every trial uses real GPTQ forward. Modeled inputs, no online guarantee.')
    for rate in map(float,a.rates.split(',')):
        m=configured(rate,a.steps,a.queries,a.objective,a.block,a.head_cap);parent=m.hif4_calibration_attention;diagnostics=[];checks=dict(states=0,returns=0)
        for role in 'qkv':
            original=getattr(m,'hif4_dynamic_quantize_'+role)
            def checked(*args,_original=original,_role=role):
                result=_original(*args)
                errors=validate_hif4_params(result,args[0].shape,'rotation_'+_role)
                assert not errors,errors
                checks['returns']+=1;return result
            setattr(m,'hif4_dynamic_quantize_'+role,checked)
        def capture(*args):
            state=parent(*args);errors=validate_frozen_state(state,'rotation_state')
            assert not errors,errors
            diagnostics.append(dict(getattr(m,'LR_DEBUG',{})));checks['states']+=1;return state
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        report['results'][str(rate)]=dict(score=sum(scores),seconds=seconds,cases=scores,diagnostics=diagnostics,checks=checks)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(rate,sum(scores),seconds,diagnostics,flush=True)


if __name__=='__main__':main()
