"""Serial paired pure-interface timing; no score computation in timed sections."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_v320_candidate import candidate_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='all')
    p.add_argument('--order',default='parent,candidate')
    p.add_argument('--paired',action='store_true',help='Alternate order within each group to limit drift')
    p.add_argument('--output',default='docs/v320-full-interface-timing.json');a=p.parse_args()
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    groups=bank if a.groups=='all' else [bank[int(i)] for i in a.groups.split(',')]
    report=dict(groups=a.groups,runs=[],complete=False,online_parent_seconds=245,
        warning='Serial local 4-thread Attention timing, not online Linux/96-core timing or a timeout guarantee.')
    if a.paired:
        modules={name:load_solution(ROOT/'solution/solution_v319.py') for name in ('parent','candidate')}
        exec(compile(candidate_source(),'v320-static','exec'),modules['candidate'].__dict__)
        times={name:dict(calibration=0.,q=0.,k=0.,v=0.) for name in modules}
        report['paired_groups']=[]
        for gi,g in enumerate(groups):
            qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            row=dict(group=gi)
            for mode in (('parent','candidate') if gi%2==0 else ('candidate','parent')):
                m=modules[mode];tt=dict(calibration=0.,q=0.,k=0.,v=0.)
                start=time.perf_counter();st=m.hif4_calibration_attention(g['calib'],qh,kh,d)
                tt['calibration']+=time.perf_counter()-start
                for s in g['test']:
                    for role,heads in (('q',qh),('k',kh),('v',kh)):
                        start=time.perf_counter();out=getattr(m,'hif4_dynamic_quantize_'+role)(*s[role],heads,d,st[role+'_state'])
                        tt[role]+=time.perf_counter()-start;del out
                row[mode]=tt
                for key in tt:times[mode][key]+=tt[key]
            report['paired_groups'].append(row)
            (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
            if gi%10==9:print('paired',gi+1,{k:sum(v.values()) for k,v in times.items()},flush=True)
        report['runs']=[dict(mode=mode,times=tt,total=sum(tt.values())) for mode,tt in times.items()]
        ratio=sum(times['candidate'].values())/sum(times['parent'].values())
        report.update(complete=True,ratio=ratio,naive_online_seconds=245*ratio)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print('paired ratio',ratio,245*ratio)
        return
    for mode in a.order.split(','):
        m=load_solution(ROOT/'solution/solution_v319.py')
        if mode=='candidate':exec(compile(candidate_source(),'v320-static','exec'),m.__dict__)
        tt=dict(calibration=0.,q=0.,k=0.,v=0.);states=[]
        for g in groups:
            qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            start=time.perf_counter();st=m.hif4_calibration_attention(g['calib'],qh,kh,d)
            tt['calibration']+=time.perf_counter()-start
            if mode=='candidate':states.append(dict(m.VR_DEBUG))
            for s in g['test']:
                for role,heads in (('q',qh),('k',kh),('v',kh)):
                    start=time.perf_counter();out=getattr(m,'hif4_dynamic_quantize_'+role)(*s[role],heads,d,st[role+'_state'])
                    tt[role]+=time.perf_counter()-start;del out
        report['runs'].append(dict(mode=mode,times=tt,total=sum(tt.values()),diagnostics=states))
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(mode,tt,sum(tt.values()),flush=True)
    total=lambda mode:sum(r['total'] for r in report['runs'] if r['mode']==mode)/sum(r['mode']==mode for r in report['runs'])
    ratio=total('candidate')/total('parent')
    report.update(complete=True,ratio=ratio,naive_online_seconds=245*ratio)
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print('ratio',ratio,'naive',245*ratio)


if __name__=='__main__':main()
