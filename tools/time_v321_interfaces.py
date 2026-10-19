"""Alternate parent/candidate order per group; no scoring inside timing."""
import json
import hashlib
import time
import torch
from probe_v321_homogeneous import ROOT,load_solution
from build_v321_candidate import candidate_source


def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    modules={k:load_solution(ROOT/'solution/solution_v320.py') for k in ('parent','candidate')}
    source=candidate_source()
    assert source==(ROOT/'solution/solution_v321.py').read_text()
    exec(compile(source,'v321-timing','exec'),modules['candidate'].__dict__)
    total={k:dict(calibration=0.,q=0.,k=0.,v=0.) for k in modules}
    report=dict(complete=False,paired_groups=[],online_parent_seconds=242,
                sha256=hashlib.sha256(source.encode()).hexdigest(),
                warning='Local Mac 4-thread Attention timing, not online 96-core timeout guarantee.')
    output=ROOT/'docs/v321-paired-interface-timing.json'
    archive_pending=None
    if output.exists():
        previous=json.loads(output.read_text())
        suffix=previous.get('sha256','initial')[:8]
        archive=ROOT/('docs/v321-paired-interface-timing-'+suffix+'.json')
        if previous.get('complete') and not archive.exists():archive_pending=(archive,previous)
    for gi,g in enumerate(bank):
        qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')];row=dict(group=gi)
        for mode in (('parent','candidate') if gi%2==0 else ('candidate','parent')):
            m=modules[mode];tt=dict(calibration=0.,q=0.,k=0.,v=0.)
            start=time.perf_counter();st=m.hif4_calibration_attention(g['calib'],qh,kh,d)
            tt['calibration']=time.perf_counter()-start
            for sample in g['test']:
                for role,heads in (('q',qh),('k',kh),('v',kh)):
                    start=time.perf_counter();out=getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],heads,d,st[role+'_state'])
                    tt[role]+=time.perf_counter()-start;del out
            row[mode]=tt
            for key in tt:total[mode][key]+=tt[key]
        report['paired_groups'].append(row)
        # Write once after timing: watched in-repo progress files can trigger
        # background indexing during the benchmark and perturb the comparison.
        if gi%10==9:print(gi+1,{k:sum(v.values()) for k,v in total.items()},flush=True)
    ratio=sum(total['candidate'].values())/sum(total['parent'].values())
    report.update(complete=True,totals=total,ratio=ratio,naive_online_seconds=242*ratio)
    if archive_pending:archive_pending[0].write_text(json.dumps(archive_pending[1],indent=2)+'\n')
    (ROOT/'docs/v321-paired-interface-timing.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['totals'],ratio,242*ratio,flush=True)


if __name__=='__main__':main()
