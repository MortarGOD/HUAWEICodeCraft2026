"""Serial alternating actual-file Attention timing without scoring."""
import argparse,hashlib,json,time
import torch
from build_v325_candidate import ROOT,candidate_source
from probe_v319_value import load_solution


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='all');p.add_argument('--output',default='docs/v325-interface-timing.json');a=p.parse_args()
    torch.set_num_threads(4);src=candidate_source()
    assert src==(ROOT/'solution/solution_v325.py').read_text()==(ROOT/'solution/solution.py').read_text()
    modules={n:load_solution(ROOT/'solution'/f'solution_v{v}.py') for n,v in (('parent',324),('candidate',325))}
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    ids=range(len(bank)) if a.groups=='all' else list(map(int,a.groups.split(',')))
    totals={k:dict(calibration=0.,q=0.,k=0.,v=0.) for k in modules};rows=[]
    for gi in ids:
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')];row=dict(group=gi)
        for name in (('parent','candidate') if gi%2==0 else ('candidate','parent')):
            m=modules[name];times=dict(calibration=0.,q=0.,k=0.,v=0.)
            start=time.perf_counter();st=m.hif4_calibration_attention(g['calib'],*dims);times['calibration']=time.perf_counter()-start
            for sample in g['test']:
                for role in 'qkv':
                    start=time.perf_counter();out=getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],dims[0] if role=='q' else dims[1],dims[2],st[role+'_state'])
                    times[role]+=time.perf_counter()-start;del out
            row[name]=times
            for key,value in times.items():totals[name][key]+=value
        rows.append(row)
        if gi%10==9:print(gi+1,{k:sum(v.values()) for k,v in totals.items()},flush=True)
    ratio=sum(totals['candidate'].values())/sum(totals['parent'].values())
    report=dict(complete=True,sha256=hashlib.sha256(src.encode()).hexdigest(),rows=rows,totals=totals,ratio=ratio,
        warning='Mac 4-thread Attention only. No scoring/in-loop writes. Not an online whole-runtime multiplier or a 96-core guarantee.')
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n');print(totals,ratio,flush=True)


if __name__=='__main__':main()
