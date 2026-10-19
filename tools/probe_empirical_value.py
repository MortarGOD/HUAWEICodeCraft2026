"""Expected-query surrogate experiment on frozen datasets, no submission changes."""
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


def configured(queries):
    m=load(314)
    if queries:
        exec(compile((ROOT/'tools/empirical_value_core.py').read_text(),'empirical_value','exec'),m.__dict__)
        m._EMPIRICAL_QUERIES=queries
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,14,26,49')
    p.add_argument('--queries',default='0,8,32')
    p.add_argument('--output',default='docs/empirical-value-screen.json')
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(dataset=a.dataset,groups=a.groups,results={},warning='Modeled activations. Calibration identity K=V must be verified, never assumed online.')
    for queries in map(int,a.queries.split(',')):
        m=configured(queries);parent=m.hif4_calibration_attention;diagnostics=[]
        def capture(*args):
            st=parent(*args);diagnostics.append(dict(getattr(m,'EMPIRICAL_DEBUG',{})));return st
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        report['results'][str(queries)]=dict(score=sum(scores),seconds=seconds,cases=scores,diagnostics=diagnostics)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(queries,sum(scores),seconds,diagnostics,flush=True)


if __name__=='__main__':main()
