"""Paired normal-bank or repeated-long invocation timing, without scoring."""
import argparse
import hashlib
import json
import time
import torch
from probe_v319_value import ROOT,load_solution
from build_v323_candidate import candidate_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--long',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);source=candidate_source()
    assert source==(ROOT/'solution/solution_v323.py').read_text()==(ROOT/'solution/solution.py').read_text()
    modules={name:load_solution(ROOT/'solution'/file) for name,file in (('parent','solution_v322.py'),('candidate','solution_v323.py'))}
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    ids=(4,14,19,48,49) if args.long else range(len(bank))
    totals={name:dict(calibration=0.,q=0.,k=0.,v=0.) for name in modules};rows=[]
    for gi in ids:
        g=bank[gi];qh,kh,d=[g[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
        sets={key:g[key] for key in ('calib','test')}
        if args.long:
            sets={key:[{role:tuple(torch.cat([x]*4,dim=0) for x in sample[role]) for role in 'qkv'} for sample in value] for key,value in sets.items()}
        row=dict(group=gi,calibration_rows=[s['k'][0].shape[0] for s in sets['calib']],test_rows=[s['k'][0].shape[0] for s in sets['test']])
        for name in (('parent','candidate') if gi%2==0 else ('candidate','parent')):
            m=modules[name];tt=dict(calibration=0.,q=0.,k=0.,v=0.)
            t=time.perf_counter();st=m.hif4_calibration_attention(sets['calib'],qh,kh,d);tt['calibration']=time.perf_counter()-t
            for sample in sets['test']:
                for role in 'qkv':
                    t=time.perf_counter();out=getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],qh if role=='q' else kh,d,st[role+'_state'])
                    tt[role]+=time.perf_counter()-t;del out
            row[name]=tt
            for key,value in tt.items():totals[name][key]+=value
        rows.append(row)
        if args.long or (gi+1)%10==0:print(gi+1,{n:sum(t.values()) for n,t in totals.items()},flush=True)
    ratio=sum(totals['candidate'].values())/sum(totals['parent'].values())
    report=dict(complete=True,sha256=hashlib.sha256(source.encode()).hexdigest(),long_stress=args.long,totals=totals,rows=rows,
                ratio=ratio,naive_online_seconds=251*ratio,warning='Mac 4-thread pure interface timing; long mode repeats tokens and is not an online composition estimate.')
    output=ROOT/('docs/v323-long-interface-timing.json' if args.long else 'docs/v323-paired-interface-timing.json')
    if output.exists():
        previous=json.loads(output.read_text())
        if previous.get('sha256')!=report['sha256']:
            archive=output.with_name(output.stem+'-'+previous['sha256'][:8]+'.json')
            if not archive.exists():archive.write_text(json.dumps(previous,indent=2)+'\n')
    output.write_text(json.dumps(report,indent=2)+'\n');print(totals,ratio,251*ratio,flush=True)


if __name__=='__main__':main()
