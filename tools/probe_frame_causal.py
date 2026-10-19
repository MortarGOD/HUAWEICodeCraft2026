"""Forced frame interventions, not test-selected submission strategies."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_attention_gates import configured
from frame_causal_core import variants,geometry
from hif4_quant import dequantize_nvfp4,quantize_to_hif4
from self_check import validate_hif4_params,validate_frozen_state


def measure(m,sample,states,qh,kh,d,queries):
    rows=sample['q'][0].shape[0];ix=torch.linspace(0,rows-1,min(rows,queries)).long()
    raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
    target=m._joint_output(*[raw[r] for r in 'qkv'],qh,kh,d,ix)
    standard=[m._deq_params(quantize_to_hif4(raw[r])) for r in 'qkv']
    baseline=(m._joint_output(*standard,qh,kh,d,ix)-target).double().square().mean().clamp_min(1e-30)
    result={};checks=0
    for label,st in states.items():
        q=m._gate_query_values(sample,st['q_state'],qh,kh,d,ix)
        pk=m._gate_parent_k(*sample['k'],kh,d,st['k_state'])
        pv=m.hif4_dynamic_quantize_v(*sample['v'],kh,d,st['v_state'])
        for role,p in (('k',pk),('v',pv)):
            assert not validate_hif4_params(p,sample[role][0].shape,'frame_causal_'+role)
            checks+=1
        out=m._joint_output(q,m._deq_params(pk),m._deq_params(pv),qh,kh,d,ix,query_subset=True)
        loss=(out-target).double().square().mean()
        # Geometric error without HiF4: transform both original inputs exactly
        # as deployed, so BF16 reciprocal mismatch is included.
        xq=m._cov_transform(raw['q'][ix],st['q_state'],qh,kh,d,'q')
        xk=m._cov_transform(raw['k'],st['k_state'],kh,kh,d,'k')
        xq=xq.transpose(0,1).reshape(-1,qh*d)
        xk=xk.transpose(0,1).reshape(-1,kh*d)
        before_quant=m._joint_output(xq,xk,raw['v'],qh,kh,d,ix,query_subset=True)
        result[label]=dict(error_over_baseline=float(loss/baseline),
                           geometry_output_over_baseline=float((before_quant-target).double().square().mean()/baseline))
    for value in result.values():
        value['gain_vs_parent']=result['parent']['error_over_baseline']-value['error_over_baseline']
    return dict(rows=rows,query_rows=len(ix),results=result),checks


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='5,13,14,19,23,47')
    p.add_argument('--queries',type=int,default=64)
    p.add_argument('--output',default='docs/frame-causal-screen.json')
    a=p.parse_args();m=configured('frames',0)
    bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    report=dict(dataset=a.dataset,groups=a.groups,query_budget=a.queries,records=[],checks=dict(states=0,returns=0),
                warning='Forced calibration-fitted interventions scored on sampled queries/full keys. No test-label fitting or online score claim; no held-out guard in these interventions.')
    for gi in map(int,a.groups.split(',')):
        started=time.perf_counter();g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._gate_parent_calibration(g['calib'],qh,kh,d)
        st=variants(m,g['calib'],parent,qh,kh,d)
        for s in st.values():
            assert not validate_frozen_state(s,'frame_causal');report['checks']['states']+=1
        held,n=measure(m,g['calib'][-1],st,qh,kh,d,a.queries);report['checks']['returns']+=n
        tests=[]
        for s in g['test']:
            value,n=measure(m,s,st,qh,kh,d,a.queries);tests.append(value);report['checks']['returns']+=n
        record=dict(group=gi,geometry={k:geometry(m,v,kh,d) for k,v in st.items()},hold=held,tests=tests,
                    seconds=time.perf_counter()-started,parent=dict(scale='qs' in parent['q_state'],dense='joint_transform' in parent['q_state'],rot=parent['q_state'].get('rot')))
        report['records'].append(record)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(gi,{k:round(sum(t['results'][k]['gain_vs_parent'] for t in tests),6) for k in st},flush=True)


if __name__=='__main__':main()
