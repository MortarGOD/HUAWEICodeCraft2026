"""Serial pure-interface timing; fixed inputs, reverse order, no scorer in timer."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from probe_v318_release import configured


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='1,5,19')
    p.add_argument('--output',default='docs/v318-interface-timing.json')
    a=p.parse_args();torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')]
    report=dict(groups=a.groups,runs=[],note='Local four-thread CPU, reverse-order serial interface calls; unknown online shapes/96-core behavior.')
    for name in ('parent','candidate','candidate','parent'):
        m=load_solution(ROOT/'solution/solution_v317.py') if name=='parent' else configured()
        timing=dict(calibration=0.,q=0.,k=0.,v=0.)
        for g in groups:
            qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            t=time.perf_counter();state=m.hif4_calibration_attention(g['calib'],qh,kh,d);timing['calibration']+=time.perf_counter()-t
            for sample in g['test']:
                for role,heads in (('q',qh),('k',kh),('v',kh)):
                    t=time.perf_counter();out=getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],heads,d,state[role+'_state'])
                    timing[role]+=time.perf_counter()-t;del out
        report['runs'].append(dict(mode=name,times=timing,total=sum(timing.values())))
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print(name,timing,sum(timing.values()),flush=True)


if __name__=='__main__':main()
