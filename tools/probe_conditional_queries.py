"""Calibration query/error-metric experiment with independent dynamic inputs."""
import argparse
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


def configured(queries,ridge=.01,metadata=False):
    m=load(314)
    if queries:
        for name in ('full_key_predictor.py','conditional_query_core.py'):
            exec(compile((ROOT/'tools'/name).read_text(),name,'exec'),m.__dict__)
        m._CQ_QUERIES=queries;m._CQ_RIDGE=ridge
        if metadata:
            exec(compile((ROOT/'tools/conditional_metadata_core.py').read_text(),'conditional_metadata','exec'),m.__dict__)
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,26,37,49')
    p.add_argument('--queries',default='0,32,128')
    p.add_argument('--ridge',type=float,default=.01)
    p.add_argument('--metadata',action='store_true')
    p.add_argument('--output',default='docs/conditional-query-screen.json')
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(dataset=a.dataset,groups=a.groups,ridge=a.ridge,metadata=a.metadata,results={},warning='Predicted K only defines a residual metric. Modeled hidden states, no reliable online score forecast.')
    for n in map(int,a.queries.split(',')):
        m=configured(n,a.ridge,a.metadata);parent=m.hif4_calibration_attention;diagnostics=[]
        checks=dict(states=0,returns=0)
        for role in 'qkv':
            original=getattr(m,'hif4_dynamic_quantize_'+role)
            def checked(*args,_original=original,_role=role):
                result=_original(*args)
                errors=validate_hif4_params(result,args[0].shape,'conditional_'+_role)
                assert not errors,errors
                checks['returns']+=1
                return result
            setattr(m,'hif4_dynamic_quantize_'+role,checked)
        def capture(*args):
            st=parent(*args);errors=validate_frozen_state(st,'conditional_state')
            assert not errors,errors
            checks['states']+=1;diagnostics.append(dict(getattr(m,'CQ_DEBUG',{})));return st
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        report['results'][str(n)]=dict(score=sum(scores),seconds=seconds,cases=scores,diagnostics=diagnostics,checks=checks)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(n,sum(scores),seconds,diagnostics,flush=True)


if __name__=='__main__':main()
