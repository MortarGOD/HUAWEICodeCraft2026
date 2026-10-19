"""Full-key/query-sampled counterfactuals, not attainable-score upper bounds."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_persistent_covariance import load
from hif4_quant import dequantize_nvfp4,quantize_to_hif4


def probabilities(q,k,qh,kh,d,ix):
    q=q[ix].reshape(-1,qh,d).transpose(0,1)
    k=k.reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    return torch.softmax(q@k.transpose(1,2)/d**.5,-1).double()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,26,37,49')
    p.add_argument('--version',type=int,default=314)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--output',default='docs/attention-error-budget-feedback.json')
    a=p.parse_args();torch.set_num_threads(a.threads);m=load(a.version)
    bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    report=dict(dataset=a.dataset,version=a.version,cases=[],complete=False,
                warning='32 queries, all keys. Counterfactual interventions are not separately attainable bounds; interactions may cancel. Modeled activations only.')
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        states=m.hif4_calibration_attention(g['calib'],qh,kh,d)
        for ti,sample in enumerate(g['test']):
            raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            quant={r:m._deq_params(getattr(m,'hif4_dynamic_quantize_'+r)(*sample[r],qh if r=='q' else kh,d,states[r+'_state'])) for r in 'qkv'}
            standard={r:m._deq_params(quantize_to_hif4(raw[r])) for r in 'qkv'}
            rows=raw['q'].shape[0]
            ix=torch.linspace(0,rows-1,min(rows,32)).long()
            pr=probabilities(raw['q'],raw['k'],qh,kh,d,ix)
            pq=probabilities(quant['q'],quant['k'],qh,kh,d,ix)
            pb=probabilities(standard['q'],standard['k'],qh,kh,d,ix)
            vr=raw['v'].reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0).double()
            vq=quant['v'].reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0).double()
            vb=standard['v'].reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0).double()
            ea=(pq-pr)@vr
            ev=pr@(vq-vr)
            ec=(pq-pr)@(vq-vr)
            error=pq@vq-pr@vr
            baseline=float((pb@vb-pr@vr).square().mean())
            baseline=max(baseline,1e-30)
            mse=lambda x:float(x.square().mean())/baseline
            reconstruction=float((ea+ev+ec-error).abs().max())
            assert reconstruction<1e-8*max(float(error.abs().max()),1.),reconstruction
            record=dict(group=gi,test=ti,rows=rows,qh=qh,kh=kh,d=d,
                        parent=mse(error),qk_component=mse(ea),v_component=mse(ev),interaction_component=mse(ec),
                        cross_inner_product=2*float((ea*ev+ea*ec+ev*ec).mean())/baseline,
                        fix_v_gain=mse(error)-mse(ea),fix_qk_gain=mse(error)-mse(ev),
                        reconstruction_error=reconstruction,
                        probability_concentration=float(pr.square().sum(-1).mean()))
            report['cases'].append(record)
        print(gi,'complete',flush=True)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
    fields=('parent','qk_component','v_component','interaction_component','cross_inner_product','fix_v_gain','fix_qk_gain')
    report['mean']={k:sum(c[k] for c in report['cases'])/len(report['cases']) for k in fields}
    report['complete']=True
    (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(report['mean'])


if __name__=='__main__':main()
