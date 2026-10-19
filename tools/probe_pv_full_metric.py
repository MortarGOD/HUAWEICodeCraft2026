"""Read-only datasets, offline full-key PV metric experiment."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def configured(blend, version=314, recovery='solve'):
    m = load(version)
    if blend:
        exec(compile((ROOT/'tools/pv_full_metric_core.py').read_text(),'pv_full_metric','exec'),m.__dict__)
        m._PVFULL_BLEND = blend
        m._PVFULL_SAFE_RECOVERY = recovery=='solve'
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,26,37,49')
    p.add_argument('--blends',default='0,0.5,1')
    p.add_argument('--version',type=int,default=314)
    p.add_argument('--recovery',choices=('solve','upper'),default='solve')
    p.add_argument('--output',default='docs/pv-full-metric-screen.json')
    a=p.parse_args()
    bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(dataset=a.dataset,groups=a.groups,parent=a.version,recovery=a.recovery,results={},
                warning='Modeled activation diagnostic; online 22000+ is not verified.')
    for blend in map(float,a.blends.split(',')):
        m=configured(blend,a.version,a.recovery)
        diagnostics=[]
        parent=m.hif4_calibration_attention
        def capture(*args):
            st=parent(*args);diagnostics.append(dict(getattr(m,'PVFULL_DEBUG',{})));return st
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        report['results'][str(blend)]=dict(score=sum(scores),cases=scores,seconds=seconds,diagnostics=diagnostics)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(blend,sum(scores),seconds,diagnostics,flush=True)


if __name__=='__main__':main()
