"""Serial, reverse-order timing of actual v318 and static v319 candidate."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_v319_candidate import candidate_source


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--groups',default='0,4,19,47')
    p.add_argument('--output',default='docs/v319-interface-timing.json')
    p.add_argument('--order',default='parent,candidate,candidate,parent')
    a=p.parse_args();torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    groups=bank if a.groups=='all' else [bank[int(i)] for i in a.groups.split(',')]
    report=dict(groups=a.groups,runs=[],complete=False,note='Local 4-thread serial interfaces, not 96-core online timing.')
    for name in a.order.split(','):
        m=load_solution(ROOT/'solution/solution_v318.py')
        if name=='candidate':exec(compile(candidate_source(),'v319-static','exec'),m.__dict__)
        timing=dict(calibration=0.,q=0.,k=0.,v=0.)
        for g in groups:
            qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            t=time.perf_counter();st=m.hif4_calibration_attention(g['calib'],qh,kh,d);timing['calibration']+=time.perf_counter()-t
            for sample in g['test']:
                for role,heads in (('q',qh),('k',kh),('v',kh)):
                    t=time.perf_counter();out=getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],heads,d,st[role+'_state'])
                    timing[role]+=time.perf_counter()-t;del out
        report['runs'].append(dict(mode=name,times=timing,total=sum(timing.values())))
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print(name,timing,sum(timing.values()),flush=True)
    ratio=sum(r['total'] for r in report['runs'] if r['mode']=='candidate')/sum(r['total'] for r in report['runs'] if r['mode']=='parent')
    report.update(complete=True,ratio=ratio,naive_online_seconds=249*ratio)
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print('ratio',ratio,249*ratio)


if __name__=='__main__':main()
