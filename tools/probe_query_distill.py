"""Calibration output distillation, independent-Q dynamic interface."""
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
from self_check import validate_frozen_state,validate_hif4_params


def configured(mode,rate=.02,steps=6,queries=32):
    m=load(314)
    if mode!='none':
        source=(ROOT/'tools/gptq_covariance_core.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_cov_transform')
        exec(compile(ast.get_source_segment(source,node),'query_coordinates','exec'),m.__dict__)
        exec(compile((ROOT/'tools/query_distill_core.py').read_text(),'query_distill','exec'),m.__dict__)
        m._QD_MODE=mode;m._QD_RATE=rate;m._QD_STEPS=steps;m._QD_QUERIES=queries
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,49')
    p.add_argument('--modes',default='none,both')
    p.add_argument('--rate',type=float,default=.02)
    p.add_argument('--steps',type=int,default=6)
    p.add_argument('--queries',type=int,default=32)
    p.add_argument('--output',default='docs/query-distill-screen.json')
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(dataset=a.dataset,groups=a.groups,rate=a.rate,steps=a.steps,queries=a.queries,results={},
                warning='Modeled inputs. Real-GPTQ forward, STE direction; neither a score forecast nor cross-role dynamic access.')
    for mode in a.modes.split(','):
        m=configured(mode,a.rate,a.steps,a.queries);parent=m.hif4_calibration_attention
        diagnostics=[];checks=dict(states=0,returns=0)
        for role in 'qkv':
            original=getattr(m,'hif4_dynamic_quantize_'+role)
            def checked(*args,_original=original,_role=role):
                result=_original(*args)
                errors=validate_hif4_params(result,args[0].shape,'query_distill_'+_role)
                assert not errors,errors
                checks['returns']+=1;return result
            setattr(m,'hif4_dynamic_quantize_'+role,checked)
        def capture(*args):
            states=parent(*args);errors=validate_frozen_state(states,'query_distill_state')
            assert not errors,errors
            diagnostics.append(dict(getattr(m,'QD_DEBUG',{})));checks['states']+=1;return states
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        report['results'][mode]=dict(score=sum(scores),seconds=seconds,cases=scores,diagnostics=diagnostics,checks=checks)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(mode,sum(scores),seconds,[(x.get('adopted'),x.get('max_gain'),x.get('max_normalized_bias')) for x in diagnostics],flush=True)


if __name__=='__main__':main()
