"""Whole-scale-tree Value optimization screen; frozen datasets and production."""
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


def configured(variance,inner=False,cap=1.01,bank=False,row=False):
    m=load(314)
    if variance:
        exec(compile((ROOT/'tools/value_metadata_core.py').read_text(),'value_metadata','exec'),m.__dict__)
        m._VM_VARIANCE=variance
        m._VM_CAP=cap
        if inner or bank or row:
            exec(compile((ROOT/'tools/value_metadata_inner_core.py').read_text(),'value_metadata_inner','exec'),m.__dict__)
        if bank:
            exec(compile((ROOT/'tools/value_metadata_bank_core.py').read_text(),'value_metadata_bank','exec'),m.__dict__)
        if row:
            exec(compile((ROOT/'tools/value_metadata_row_core.py').read_text(),'value_metadata_row','exec'),m.__dict__)
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,26,37,49')
    p.add_argument('--variance',default='0,1,8')
    p.add_argument('--output',default='docs/value-metadata-screen.json')
    p.add_argument('--inner',action='store_true')
    p.add_argument('--cap',type=float,default=1.01)
    p.add_argument('--bank',action='store_true')
    p.add_argument('--row',action='store_true')
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(dataset=a.dataset,groups=a.groups,inner=a.inner,cap=a.cap,bank=a.bank,row=a.row,results={},warning='Modeled activation diagnostic; no online-score guarantee.')
    for variance in map(float,a.variance.split(',')):
        m=configured(variance,a.inner,a.cap,a.bank,a.row);parent=m.hif4_calibration_attention;diagnostics=[]
        validation=dict(states=0,returns=0)
        for role in 'qkv':
            original=getattr(m,'hif4_dynamic_quantize_'+role)
            def checked(*args,_original=original,_role=role):
                result=_original(*args)
                errors=validate_hif4_params(result,args[0].shape,'metadata_'+_role)
                assert not errors,errors
                validation['returns']+=1
                return result
            setattr(m,'hif4_dynamic_quantize_'+role,checked)
        def capture(*args):
            states=parent(*args)
            errors=validate_frozen_state(states,'metadata_states')
            assert not errors,errors
            validation['states']+=1
            diagnostics.append(dict(getattr(m,'VM_DEBUG',{})));return states
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        report['results'][str(variance)]=dict(score=sum(scores),seconds=seconds,cases=scores,diagnostics=diagnostics,validation=validation)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(variance,sum(scores),seconds,diagnostics,flush=True)


if __name__=='__main__':main()
