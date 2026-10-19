"""Calibration-statistics ablation; submission source remains unchanged."""
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


def configured(mode):
    m=load(314)
    if mode!='none':
        source=(ROOT/'tools/pv_full_metric_core.py').read_text()
        guard=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_pvfull_guard')
        exec(compile(ast.get_source_segment(source,guard),'cov_guard','exec'),m.__dict__)
        exec(compile((ROOT/'tools/gptq_covariance_core.py').read_text(),'covariance_core','exec'),m.__dict__)
        m._COV_MODE=mode.removesuffix('_fp32');m._COV_H_FP32=mode.endswith('_fp32')
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,26,37,49')
    p.add_argument('--modes',default='none,legacy,all,centered,centered_fp32')
    p.add_argument('--output',default='docs/gptq-covariance-screen.json')
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(dataset=a.dataset,groups=a.groups,results={},warning='Diagnostic modeled activations; unknown online Attention lengths. No high-score forecast.')
    for mode in a.modes.split(','):
        m=configured(mode);parent=m.hif4_calibration_attention;diagnostics=[]
        checks=dict(states=0,returns=0)
        for role in 'qkv':
            original=getattr(m,'hif4_dynamic_quantize_'+role)
            def checked(*args,_original=original,_role=role):
                result=_original(*args)
                errors=validate_hif4_params(result,args[0].shape,'covariance_'+_role)
                assert not errors,errors
                checks['returns']+=1;return result
            setattr(m,'hif4_dynamic_quantize_'+role,checked)
        def capture(*args):
            states=parent(*args);errors=validate_frozen_state(states,'covariance_state')
            assert not errors,errors
            checks['states']+=1;diagnostics.append(dict(getattr(m,'COV_DEBUG',{})));return states
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        report['results'][mode]=dict(score=sum(scores),seconds=seconds,cases=scores,diagnostics=diagnostics,checks=checks)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(mode,sum(scores),seconds,[(d.get('q_heads'),d.get('k_heads'),d.get('key_mean_trace_fraction'),d.get('error')) for d in diagnostics],flush=True)


if __name__=='__main__':main()
