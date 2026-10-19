"""Coordinate-correct role interventions, offline only; not attainable bounds."""
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from hif4_quant import quantize_to_hif4

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    m=load_solution(ROOT/'solution/solution_v387.py')
    report=dict(complete=False,deployable=False,query_count=32,rows=[],source_sha256=hashlib.sha256((ROOT/'solution/solution_v387.py').read_bytes()).hexdigest())
    for name,path,groups in [('structural','local_fitted_models_attention_feedback_v2',[10,12,16,18]),('stress','local_fitted_archive/attention_feedback_v2_before_structural_20260906',[0,4,24,49])]:
        bank=torch.load(ROOT/path/'attn.pt',weights_only=False)
        for gi in groups:
            g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
            states=m.hif4_calibration_attention(g['calib'],qh,kh,d)
            for ti,s in enumerate(g['test']):
                raw={r:m.dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
                quant={r:m._deq_params(getattr(m,'hif4_dynamic_quantize_'+r)(*s[r],qh if r=='q' else kh,d,states[r+'_state'])).float() for r in 'qkv'}
                ideal={r:m._kt_input(raw[r],states[r+'_state'],qh if r=='q' else kh,d,r) for r in 'qk'};ideal['v']=raw['v']
                t=raw['q'].shape[0];ix=torch.linspace(0,t-1,min(32,t)).long()
                target=m._joint_output(raw['q'],raw['k'],raw['v'],qh,kh,d,ix).double()
                standard={r:m._deq_params(quantize_to_hif4(raw[r])).float() for r in 'qkv'}
                std=(m._joint_output(standard['q'],standard['k'],standard['v'],qh,kh,d,ix).double()-target).square().mean().clamp_min(1e-30)
                costs={}
                for mask in range(8):
                    tensors=[ideal[r] if mask&(1<<j) else quant[r] for j,r in enumerate('qkv')]
                    err=m._joint_output(*tensors,qh,kh,d,ix).double()-target
                    label=''.join(r for j,r in enumerate('qkv') if mask&(1<<j)) or 'parent'
                    costs[label]=float(err.square().mean()/std)
                p=m._vr_probs(raw['q'][ix],raw['k'],qh,kh,d).double()
                pq=m._vr_probs(quant['q'][ix],quant['k'],qh,kh,d).double()
                vr=raw['v'].reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0).double()
                vq=quant['v'].reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0).double()
                a=(pq-p)@vr;b=p@(vq-vr);c=(pq-p)@(vq-vr)
                exact=pq@vq-p@vr
                assert torch.allclose(a+b+c,exact,atol=1e-9,rtol=1e-8)
                row=dict(bank=name,group=gi,test=ti,t=t,qh=qh,kh=kh,d=d,costs=costs,fix_gains={r:costs['parent']-costs[r] for r in ('q','k','v','qk')},components={label:float(x.square().mean()/std) for label,x in [('qk',a),('v',b),('cross',c)]},inner_terms=float(2*(a*b+a*c+b*c).mean()/std))
                report['rows'].append(row)
            selected=report['rows'][-5:]
            print(name,gi,{r:sum(x['fix_gains'][r] for x in selected)/5 for r in ('q','k','v','qk')},flush=True)
            (ROOT/'docs/v399-role-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs/v399-role-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
