"""Factorial QK/V ablation under alternative scoring assumptions.

These scorer perturbations are counterexamples, not assertions about the judge.
No data, seed, or scenario weights are fitted to the observed online score.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_persistent_covariance import load
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4
from evaluate_local import mse,score


def output(x,qh,kh,d,temperature=1.0,causal=False,sink_bias=0.0):
    t=x['q'].shape[0]
    q=x['q'].reshape(t,qh,d).transpose(0,1)
    k=x['k'].reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    v=x['v'].reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    z=q@k.transpose(1,2)*(temperature/d**0.5)
    if sink_bias:
        z[...,0]+=sink_bias
    if causal:
        z.masked_fill_(torch.ones(t,t,dtype=torch.bool).triu(1),float('-inf'))
    return torch.softmax(z,-1)@v


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--groups-per-model',type=int,default=1)
    p.add_argument('--scenarios',default='full,causal,sharp16,sharp64,causal_sharp16')
    args=p.parse_args()
    bank=torch.load(ROOT/'local_fitted_models_covariance_v1/attn.pt',weights_only=False)
    groups=bank[:args.groups_per_model]+bank[25:25+args.groups_per_model]
    mods={'base':load(308),'qk_only':load(309),'v_feedback_only':load(308),
          'v_polish_only':load(308),'combined':load(311)}
    mods['v_feedback_only'].hif4_dynamic_quantize_v=load(310).hif4_dynamic_quantize_v
    mods['v_polish_only'].hif4_dynamic_quantize_v=load(311).hif4_dynamic_quantize_v
    choices={'full':(1,False,0),'causal':(1,True,0),'sharp16':(16,False,0),
             'sharp64':(64,False,0),'causal_sharp16':(16,True,0),
             'sink8':(1,False,8),'sink16':(1,False,16)}
    scenarios={name:choices[name] for name in args.scenarios.split(',')}
    report={'groups':len(groups),'dataset':'local_fitted_models_covariance_v1',
            'warning':'alternative scorer assumptions only; not online evidence or a recalibrated main bank',
            'cases':[],'hessian_diagnostics':[],
            'totals':{s:{m:0.0 for m in mods} for s in scenarios}}
    for gi,g in enumerate(groups):
        qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        states={name:m.hif4_calibration_attention(g['calib'],qh,kh,d) for name,m in mods.items()}
        for name in ('base','qk_only'):
            for role in 'qk':
                st=states[name][role+'_state'];h=st.get('H') if isinstance(st,dict) else None
                if isinstance(h,torch.Tensor):
                    eigen=torch.linalg.eigvalsh(h.float())
                    report['hessian_diagnostics'].append({'group':gi,'variant':name,'role':role,
                        'minimum_eigenvalue':float(eigen.min()),'negative_fraction':float((eigen<0).float().mean()),
                        'most_negative_relative_to_max':float(eigen.min()/eigen.abs().max())})
        for ti,sample in enumerate(g['test']):
            original={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            standard={r:dequantize_hif4(quantize_to_hif4(x),tuple(x.shape)) for r,x in original.items()}
            players={}
            for name,m in mods.items():
                players[name]={}
                for r in 'qkv':
                    params=getattr(m,'hif4_dynamic_quantize_'+r)(*sample[r],qh if r=='q' else kh,d,states[name][r+'_state'])
                    players[name][r]=dequantize_hif4(params,tuple(original[r].shape))
            result={'group':gi,'test':ti,'rows':original['q'].shape[0],'scores':{}}
            for scenario,(temperature,causal,sink_bias) in scenarios.items():
                target=output(original,qh,kh,d,temperature,causal,sink_bias)
                bm=mse(target,output(standard,qh,kh,d,temperature,causal,sink_bias))
                result['scores'][scenario]={}
                for name,x in players.items():
                    val=score(bm,mse(target,output(x,qh,kh,d,temperature,causal,sink_bias)))
                    result['scores'][scenario][name]=val
                    report['totals'][scenario][name]+=val
            report['cases'].append(result)
            Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
            print('done',gi,ti,flush=True)
    print(json.dumps(report['totals'],indent=2),flush=True)


if __name__=='__main__':main()
