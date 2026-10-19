"""Alternating same-process paired timing with no scoring or in-loop writes."""
import hashlib
import json
import time
import torch
from probe_v322_translation import ROOT
from probe_v319_value import load_solution
from build_v322_candidate import candidate_source


def main():
    torch.set_num_threads(4);source=candidate_source()
    assert source==(ROOT/'solution/solution_v322.py').read_text()
    assert source==(ROOT/'solution/solution.py').read_text()
    modules={k:load_solution(ROOT/'solution/solution_v321.py') for k in ('parent','candidate')}
    exec(compile(source,'v322-timing','exec'),modules['candidate'].__dict__)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    totals={k:dict(calibration=0.,q=0.,k=0.,v=0.) for k in modules};rows=[]
    for gi,g in enumerate(bank):
        dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')];row=dict(group=gi)
        for name in (('parent','candidate') if gi%2==0 else ('candidate','parent')):
            m=modules[name];times=dict(calibration=0.,q=0.,k=0.,v=0.)
            t=time.perf_counter();st=m.hif4_calibration_attention(g['calib'],*dims)
            times['calibration']=time.perf_counter()-t
            for sample in g['test']:
                for role in 'qkv':
                    t=time.perf_counter();out=getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],dims[0] if role=='q' else dims[1],dims[2],st[role+'_state'])
                    times[role]+=time.perf_counter()-t;del out
            row[name]=times
            for key,value in times.items():totals[name][key]+=value
        rows.append(row)
        if gi%10==9:print(gi+1,{n:sum(t.values()) for n,t in totals.items()},flush=True)
    ratio=sum(totals['candidate'].values())/sum(totals['parent'].values())
    report=dict(complete=True,sha256=hashlib.sha256(source.encode()).hexdigest(),totals=totals,rows=rows,
                ratio=ratio,online_parent_seconds=246,naive_online_seconds=246*ratio,
                warning='Mac 4-thread Attention only. Not a 96-core online guarantee; Linear unchanged.')
    output=ROOT/'docs/v322-paired-interface-timing.json'
    if output.exists():
        previous=json.loads(output.read_text())
        if previous.get('sha256')!=report['sha256']:
            archive=ROOT/('docs/v322-paired-interface-timing-'+previous['sha256'][:8]+'.json')
            if not archive.exists():archive.write_text(json.dumps(previous,indent=2)+'\n')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(totals,ratio,246*ratio,flush=True)


if __name__=='__main__':main()
