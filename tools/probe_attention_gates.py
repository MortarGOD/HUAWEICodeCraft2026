"""Full-key parent candidate-bank ablation without production modifications."""
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


def configured(mode,head_cap=1.01):
    m=load(314)
    if mode!='none':
        source=(ROOT/'tools/gptq_covariance_core.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_cov_transform')
        exec(compile(ast.get_source_segment(source,node),'gate_coordinates','exec'),m.__dict__)
        exec(compile((ROOT/'tools/attention_gate_core.py').read_text(),'attention_gates','exec'),m.__dict__)
        m._GATE_VALUE=mode
        m._GATE_HEAD_CAP=head_cap
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,49')
    p.add_argument('--modes',default='none,raw,deployed')
    p.add_argument('--head-cap',type=float,default=1.01,help='0 removes the per-Q-head cap, not the two aggregate-fold guards')
    p.add_argument('--output',default='docs/attention-gates-screen.json')
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(dataset=a.dataset,groups=a.groups,head_cap=a.head_cap,results={},warning='Modeled activations, not an online score estimate. Both gates retain all keys; no test label selection.')
    for mode in a.modes.split(','):
        m=configured(mode,a.head_cap);parent=m.hif4_calibration_attention;diags=[];checks=dict(states=0,returns=0)
        for role in 'qkv':
            original=getattr(m,'hif4_dynamic_quantize_'+role)
            def checked(*args,_original=original,_role=role):
                out=_original(*args);errors=validate_hif4_params(out,args[0].shape,'gate_'+_role)
                assert not errors,errors
                checks['returns']+=1;return out
            setattr(m,'hif4_dynamic_quantize_'+role,checked)
        def capture(*args):
            states=parent(*args);errors=validate_frozen_state(states,'gate_state')
            assert not errors,errors
            checks['states']+=1;diags.append(dict(getattr(m,'GATE_DEBUG',{})));return states
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        report['results'][mode]=dict(score=sum(scores),cases=scores,seconds=seconds,diagnostics=diags,checks=checks)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(mode,sum(scores),seconds,[(x.get('q_changed'),x.get('k_changed')) for x in diags],flush=True)


if __name__=='__main__':main()
