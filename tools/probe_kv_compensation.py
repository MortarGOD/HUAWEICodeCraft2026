"""Cal-query K/V compensation vs a clearly labeled test-query oracle.

The deployable arm uses current K only (with calibration-confirmed K=V),
calibration Q samples, and frozen parent states. Actual test Q/V are read by the
scorer only. The oracle arm is a diagnostic and must never be submitted.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_persistent_covariance import load
from kv_compensation_math import refine_keys
from hif4_quant import dequantize_nvfp4,quantize_to_hif4
from self_check import validate_hif4_params


def transform_keys(m,x,state,kh,d):
    y=x.reshape(-1,kh,d)
    if 'ks' in state:y=y*state['ks'].float()[None]
    if 'joint_transform' in state:
        y=(y.transpose(0,1)@state['joint_transform'].float()).transpose(0,1)
    elif state.get('rot'):
        rotation=m._make_R(d,int(state.get('rseed') or 0))
        if rotation is not None:y=y@rotation
    return y.transpose(0,1).contiguous()


def calibration_queries(m,group,states,count):
    qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    pieces=[]
    for sample in group['calib'][:-1]:
        if sample['q'][0].shape[0]<=1:continue
        q=m._deq_params(m.hif4_dynamic_quantize_q(*sample['q'],qh,d,states['q_state']))
        ix=torch.linspace(0,q.shape[0]-1,min(q.shape[0],256)).long()
        pieces.append(q[ix].reshape(-1,qh,d))
    q=torch.cat(pieces)
    ix=torch.linspace(0,q.shape[0]-1,min(q.shape[0],count)).long()
    return q[ix].transpose(0,1).reshape(kh,-1,d).contiguous()


def candidate(m,pair,states,queries,kh,d,rounds):
    raw=dequantize_nvfp4(*pair).float()
    params=m.hif4_dynamic_quantize_k(*pair,kh,d,states['k_state'])
    kq=m._deq_params(params).reshape(-1,kh,d).transpose(0,1).contiguous()
    # V is inferred from this same K input, never from the test V argument.
    vq=m._deq_params(m.hif4_dynamic_quantize_v(*pair,kh,d,states['v_state']))
    vq=vq.reshape(-1,kh,d).transpose(0,1).contiguous()
    vr=raw.reshape(-1,kh,d).transpose(0,1).contiguous()
    step=(m._params_unit_flat(params)*.25).reshape(-1,kh,d).transpose(0,1).contiguous()
    result,stats=refine_keys(queries,transform_keys(m,raw,states['k_state'],kh,d),kq,vr,vq,step,rounds,2.)
    output=m._values_to_params(result.transpose(0,1).reshape(raw.shape).contiguous(),params)
    assert not validate_hif4_params(output,raw.shape,'key_compensation')
    return output,stats


def evaluate(m,sample,states,params,qh,kh,d,mask=None):
    raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
    base={r:m._deq_params(quantize_to_hif4(raw[r])) for r in 'qkv'}
    play={r:m._deq_params(getattr(m,'hif4_dynamic_quantize_'+r)(*sample[r],qh if r=='q' else kh,d,states[r+'_state'])) for r in 'qkv'}
    changed=m._deq_params(params)
    if mask is not None:
        changed=torch.where(mask[None,:,None],changed.reshape(-1,kh,d),play['k'].reshape(-1,kh,d)).reshape(changed.shape)
    choices=torch.arange(1,raw['q'].shape[0],2)
    ix=choices[torch.linspace(0,choices.numel()-1,min(choices.numel(),32)).long()]
    target=m._joint_output(*[raw[r] for r in 'qkv'],qh,kh,d,ix)
    refbase=m._joint_output(*[base[r] for r in 'qkv'],qh,kh,d,ix)
    parent=m._joint_output(*[play[r] for r in 'qkv'],qh,kh,d,ix)
    result=m._joint_output(play['q'],changed,play['v'],qh,kh,d,ix)
    baseline=float((refbase-target).double().square().mean())
    costs=[]
    for out in (parent,result):
        error=(out-target).double().square().mean((-1,-2))
        costs.append(torch.stack((error[::2].mean(0),error[1::2].mean(0))))
    gain=float((costs[0]-costs[1]).mean())/max(baseline,1e-30)
    return gain,costs,play['q']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--groups',default='1,49')
    p.add_argument('--tests',default='0,4')
    p.add_argument('--queries',type=int,default=8)
    p.add_argument('--rounds',type=int,default=8)
    p.add_argument('--output',default='docs/kv-compensation-screen.json')
    a=p.parse_args();m=load(314)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    report=dict(queries=a.queries,rounds=a.rounds,groups=[],
                warning='Query-sampled diagnostic. Oracle reads even-row test queries and is NOT deployable; evaluation uses disjoint odd queries. No 22000+ claim.')
    for gi in map(int,a.groups.split(',')):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        assert all(torch.equal(dequantize_nvfp4(*s['k']),dequantize_nvfp4(*s['v'])) for s in g['calib'])
        states=m.hif4_calibration_attention(g['calib'],qh,kh,d)
        queries=calibration_queries(m,g,states,a.queries)
        proposal,_=candidate(m,g['calib'][-1]['k'],states,queries,kh,d,a.rounds)
        _,costs,_=evaluate(m,g['calib'][-1],states,proposal,qh,kh,d)
        mask=(costs[1]<costs[0]*.99).all(0)
        record=dict(group=gi,accepted_heads=int(mask.sum()),tests=[])
        for ti in map(int,a.tests.split(',')):
            s=g['test'][ti]
            start=time.perf_counter()
            proposal,stats=candidate(m,s['k'],states,queries,kh,d,a.rounds)
            rawgain,_,q=evaluate(m,s,states,proposal,qh,kh,d)
            guarded,_,_=evaluate(m,s,states,proposal,qh,kh,d,mask)
            even=torch.arange(0,q.shape[0],2)
            ix=even[torch.linspace(0,even.numel()-1,min(a.queries,even.numel())).long()]
            oracle=q[ix].reshape(-1,qh,d).transpose(0,1).reshape(kh,-1,d).contiguous()
            oracle_params,oracle_stats=candidate(m,s['k'],states,oracle,kh,d,a.rounds)
            oracle_gain,_,_=evaluate(m,s,states,oracle_params,qh,kh,d)
            result=dict(test=ti,rows=s['k'][0].shape[0],ungated_gain=rawgain,guarded_gain=guarded,
                        oracle_disjoint_query_gain=oracle_gain,
                        surrogate_initial=stats['initial'].tolist(),surrogate_final=stats['final'].tolist(),
                        assumption_holds=torch.equal(dequantize_nvfp4(*s['k']),dequantize_nvfp4(*s['v'])),
                        seconds=time.perf_counter()-start)
            record['tests'].append(result)
            print(gi,ti,'cal',round(rawgain,5),'guarded',round(guarded,5),'oracle',round(oracle_gain,5),flush=True)
        report['groups'].append(record)
        (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
