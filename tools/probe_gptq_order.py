"""Static source builder and offline order probe; no runtime source loading in submissions."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import types
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def candidate_source():
    s=(ROOT/'solution/solution_v314.py').read_text()
    old='qs = _gptq_quantize_batched(xs, us, u_full)'
    assert s.count(old)==1
    s=s.replace(old,'qs = _order_quant(xs, us, u_full, state)')
    start=s.index('def _joint_quant_queries(');end=s.index('def _joint_output(',start)
    body=s[start:end]
    assert body.count('_gptq_quantize_batched(')==1
    body=body.replace('_gptq_quantize_batched(','_order_quant(')
    old='unit.reshape(-1, qh, d).transpose(0, 1).contiguous(), u)'
    assert body.count(old)==1
    body=body.replace(old,'unit.reshape(-1, qh, d).transpose(0, 1).contiguous(), u, state)')
    return s[:start]+body+s[end:]+'\n'+(ROOT/'tools/gptq_order_core.py').read_text()


def configured(mode):
    if mode=='none':return load(314)
    m=types.ModuleType('gptq_order')
    exec(compile(candidate_source(),'gptq_order','exec'),m.__dict__)
    m._ORDER_MODE=mode
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,26,37,49')
    p.add_argument('--modes',default='none,conditional,diagonal,reverse')
    p.add_argument('--output',default='docs/gptq-order-screen.json')
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(dataset=a.dataset,groups=a.groups,results={},warning='Diagnostic proxy, not an online score prediction.')
    for mode in a.modes.split(','):
        m=configured(mode);diagnostics=[];parent=m.hif4_calibration_attention
        def capture(*args):
            st=parent(*args);diagnostics.append(dict(getattr(m,'ORDER_DEBUG',{})));return st
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        report['results'][mode]=dict(score=sum(scores),seconds=seconds,cases=scores,diagnostics=diagnostics)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(mode,sum(scores),seconds,diagnostics,flush=True)


if __name__=='__main__':main()
