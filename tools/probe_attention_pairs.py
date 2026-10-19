"""Measure pair traps in real parent outputs; only Q gets an output diagnostic."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_gptq_covariance import configured
from attention_pair_math import correlated_pairs,pair_refine
from hif4_quant import dequantize_nvfp4,quantize_to_hif4
from self_check import validate_hif4_params


def metric_from_state(state,damp):
    if 'H' in state:
        h=state['H'].float()
    else:
        u=state['u'].float();d=u.shape[-1]
        inverse_u=torch.linalg.solve_triangular(u,torch.eye(d).expand_as(u),upper=True)
        damped=inverse_u@inverse_u.transpose(1,2)
        ridge=damped.diagonal(dim1=-2,dim2=-1).mean(-1)*(damp/(1+damp))
        h=damped-ridge[:,None,None]*torch.eye(d)
    return (h+h.transpose(1,2))*.5


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='1,5,14,49')
    p.add_argument('--tests',default='0,2,4')
    p.add_argument('--pairs',default='64,512')
    p.add_argument('--rows',type=int,default=64)
    p.add_argument('--rounds',type=int,default=1)
    p.add_argument('--exact',action='store_true')
    p.add_argument('--output',default='docs/attention-pair-trap-screen.json')
    a=p.parse_args();m=configured('legacy')
    bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    report=dict(dataset=a.dataset,exact=a.exact,rounds=a.rounds,records=[],warning='Sampled rows. H improvement is not online gain; Q diagnostic uses full keys, K has no partial-key score claim.')
    for gi in map(int,a.groups.split(',')):
        group=bank[gi];qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        states=m._cov_parent_calibration(group['calib'],qh,kh,d)
        for ti in map(int,a.tests.split(',')):
            sample=group['test'][ti];rows=sample['q'][0].shape[0]
            ix=torch.linspace(0,rows-1,min(a.rows,rows)).long()
            raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            params={r:getattr(m,'hif4_dynamic_quantize_'+r)(*sample[r],qh if r=='q' else kh,d,states[r+'_state']) for r in 'qkv'}
            values={r:m._deq_params(params[r]) for r in 'qkv'}
            base={r:m._deq_params(quantize_to_hif4(raw[r])) for r in 'qkv'}
            target=m._joint_output(*[raw[r] for r in 'qkv'],qh,kh,d,ix)
            standard=m._joint_output(*[base[r] for r in 'qkv'],qh,kh,d,ix)
            parent=m._joint_output(*[values[r] for r in 'qkv'],qh,kh,d,ix)
            baseline=(standard-target).double().square().mean().clamp_min(1e-30)
            old_error=(parent-target).double().square().mean()
            for role,nh in (('q',qh),('k',kh)):
                state=states[role+'_state']
                if 'u' not in state:continue
                h=metric_from_state(state,m.GPTQ_DAMP)
                h=h.repeat_interleave(nh//h.shape[0],0)
                x=m._cov_transform(raw[role][ix],state,nh,kh,d,role)
                y=values[role][ix].reshape(-1,nh,d).transpose(0,1).contiguous()
                step=(m._params_unit_flat(params[role])[ix]*.25).reshape(-1,nh,d).transpose(0,1).contiguous()
                for limit in map(int,a.pairs.split(',')):
                    started=time.perf_counter()
                    changed,stats=pair_refine(x,y,step,h,correlated_pairs(h,limit),a.rounds,not a.exact,exact=a.exact)
                    pp=m._values_to_params(changed.transpose(0,1).reshape(-1,nh*d).contiguous(),{k:v[ix] for k,v in params[role].items()})
                    assert not validate_hif4_params(pp,(len(ix),nh*d),'pair_'+role)
                    record=dict(group=gi,test=ti,original_rows=rows,role=role,pairs=limit,stats=stats,seconds=time.perf_counter()-started)
                    if role=='q':
                        out=m._joint_output(m._deq_params(pp),values['k'],values['v'],qh,kh,d,ix,query_subset=True)
                        record['query_sample_gain']=float((old_error-(out-target).double().square().mean())/baseline)
                    report['records'].append(record)
            print(gi,ti,'done',flush=True)
            (ROOT/a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
