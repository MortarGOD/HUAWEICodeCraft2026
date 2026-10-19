"""Serial pure interface timing; no scorer or I/O inside timed intervals."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_persistent_covariance import load


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--groups',default='1,5,14,19,47,49')
    p.add_argument('--output',default='docs/v316-interface-timing.json')
    a=p.parse_args()
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(groups=a.groups,runs=[],warning='Local CPU, modeled activations. Interface-only seconds; 96-core online runtime not measured.')
    for version in (315,316,316,315):
        m=load(version)
        times=dict(calibration=0.,q=0.,k=0.,v=0.)
        choices=[]
        for g in groups:
            qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            start=time.perf_counter()
            st=m.hif4_calibration_attention(g['calib'],qh,kh,d)
            times['calibration']+=time.perf_counter()-start
            choices.append(dict(getattr(m,'FRAME_DEBUG',{})).get('choice'))
            for sample in g['test']:
                for role,heads in (('q',qh),('k',kh),('v',kh)):
                    start=time.perf_counter()
                    out=getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],heads,d,st[role+'_state'])
                    times[role]+=time.perf_counter()-start
                    del out
        report['runs'].append(dict(version=version,times=times,total=sum(times.values()),choices=choices))
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(version,times,sum(times.values()),flush=True)


if __name__=='__main__':main()
