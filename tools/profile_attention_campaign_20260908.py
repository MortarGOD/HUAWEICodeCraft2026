"""Serial, uninstrumented phase timings; no conversion to online runtime."""
import json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load


def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    versions=(422,430,431,432,433,434);modules={v:load(v,'profile') for v in versions};rows=[]
    for m in modules.values():m._quantize_linear_values(torch.zeros(2,64))
    for repeat in range(2):
        for gi in (0,24,49):
            g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            for v in (versions if repeat==0 else tuple(reversed(versions))):
                m=modules[v];start=time.perf_counter();st=m.hif4_calibration_attention(g['calib'],qh,kh,d)
                calibration=time.perf_counter()-start;start=time.perf_counter()
                for sample in g['test']:
                    for role in 'vkq':
                        getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],qh if role=='q' else kh,d,st[role+'_state'])
                rows.append({'repeat':repeat,'group':gi,'version':v,'calibration':calibration,'dynamic':time.perf_counter()-start})
                print(rows[-1],flush=True)
    totals={str(v):{phase:sum(r[phase] for r in rows if r['version']==v) for phase in ('calibration','dynamic')} for v in versions}
    report={'groups':[0,24,49],'repetitions':2,'threads':4,'rows':rows,'totals':totals,
            'warning':'Local serial uninstrumented timing only. Two repetitions and three groups do not establish an online runtime guarantee; no Kunpeng conversion.'}
    (ROOT/'docs/attention-campaign-serial-profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(totals,flush=True)


if __name__=='__main__':
    with torch.inference_mode():main()
