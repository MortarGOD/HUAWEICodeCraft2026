"""Serial incremental interface timings with one shared parent calibration."""
import json,time
import torch
from build_output_risk_versions import ROOT
from probe_v319_value import load_solution


def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    modules={v:load_solution(ROOT/f'solution/solution_v{v}.py') for v in (324,339,340)}
    rows=[]
    for gi in (0,1,5,14,19,49):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        start=time.perf_counter();root=modules[324].hif4_calibration_attention(g['calib'],qh,kh,d);cal=time.perf_counter()-start
        states={324:root};extra={324:0.}
        for version in (339,340):
            m=modules[version];guard=m._rk_guard if version==339 else m._rs_guard
            start=time.perf_counter();states[version]=guard(g['calib'],root,qh,kh,d);extra[version]=time.perf_counter()-start
        common=0.;vt={v:0. for v in modules}
        for s in g['test']:
            for role in 'qk':
                start=time.perf_counter();getattr(modules[324],'hif4_dynamic_quantize_'+role)(*s[role],qh if role=='q' else kh,d,root[role+'_state']);common+=time.perf_counter()-start
            for version in ((324,339,340) if gi%2==0 else (340,339,324)):
                start=time.perf_counter();modules[version].hif4_dynamic_quantize_v(*s['v'],kh,d,states[version]['v_state']);vt[version]+=time.perf_counter()-start
        rows.append(dict(group=gi,base_calibration=cal,common_qk=common,extra_calibration=extra,value=vt,
                         total={v:cal+common+extra[v]+vt[v] for v in modules}))
        print(gi,rows[-1]['total'],flush=True)
    totals={v:sum(r['total'][v] for r in rows) for v in modules}
    (ROOT/'docs/v339-v340-interface-timing.json').write_text(json.dumps(dict(complete=True,rows=rows,totals=totals,
        warning='Six-group shared-parent incremental Attention timing, not independent full 300-case timing or online guarantee.'),indent=2)+'\n')
    print(totals,flush=True)


if __name__=='__main__':main()
