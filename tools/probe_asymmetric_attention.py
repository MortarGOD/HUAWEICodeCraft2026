"""Offline asymmetric reconstruction experiments; never submitted."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def configured(strength):
    m=load(314)
    if strength:
        exec(compile((ROOT/'tools/asymmetric_attention_core.py').read_text(),
                     'asymmetric_attention_core','exec'),m.__dict__)
        m._ASYM_STRENGTH=strength
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_covariance_v1')
    p.add_argument('--groups',default='2,7,27,32')
    p.add_argument('--strengths',default='0,0.5,1')
    p.add_argument('--output',default='docs/v315-asymmetric-screen.json')
    a=p.parse_args()
    bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    groups=bank if a.groups=='all' else [bank[int(i)] for i in a.groups.split(',')]
    result={'dataset':a.dataset,'groups':a.groups,'warning':'Diagnostic proxies, not calibrated online predictions.','results':{}}
    for strength in map(float,a.strengths.split(',')):
        m=configured(strength)
        parent=m.hif4_calibration_attention
        diagnostics=[]
        def capture(*args):
            st=parent(*args)
            diagnostics.append(dict(getattr(m,'ASYM_DEBUG',{})))
            return st
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        result['results'][str(strength)]={'score':sum(scores),'seconds':seconds,'cases':scores,'diagnostics':diagnostics}
        (ROOT/a.output).write_text(json.dumps(result,indent=2)+'\n')
        print(strength,sum(scores),seconds,diagnostics,flush=True)


if __name__=='__main__':main()
