"""Post-push shared-parent V-only verification with role-local selection."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.check_role_decoy_calibration import role_samples
from tools.verify_attention_campaign import equal
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4
from self_check import validate_hif4_params,validate_attention_calibration_result


@torch.inference_mode()
def run():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True);ap.add_argument('--groups',required=True)
    ap.add_argument('--mode',choices=['paired','row_permute','block_resample'],default='paired')
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    torch.set_num_threads(4)
    base,child=load(435,'443parent'),load(443,'443child')
    path=ROOT/args.dataset/'attn.pt';bank=torch.load(path,weights_only=True)
    report={'complete':False,'source_sha256':hashlib.sha256((ROOT/'solution/solution_v443.py').read_bytes()).hexdigest(),
            'dataset':args.dataset,'mode':args.mode,'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'groups':[],'return_checks':0,'state_checks':0,
            'limits':'Mechanism-only temperature recalibration from retained parent states using V context; not full new calibration wrapper or role-independent Q fitting. Full dynamic inputs, <=64 query scoring. No online score estimate; local timing includes verification overhead separately.'}
    for gi in map(int,args.groups.split(',')):
        g=bank[gi];saved=copy.deepcopy(g)
        qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        if args.mode=='paired':
            context=base.hif4_calibration_attention(g['calib'],qh,kh,d)
            states=copy.deepcopy(context);cal=g['calib']
        else:
            states={}
            for r in 'qkv':
                samples=role_samples(g['calib'],r,args.mode,829+ord(r))
                result=base.hif4_calibration_attention(samples,qh,kh,d)
                states[r+'_state']=copy.deepcopy(result[r+'_state'])
                if r=='v':context,cal=result,samples
        frozen=copy.deepcopy((context,states))
        fitting={'eligible':not bool(context['q_state'].get('gq')),'errors':0}
        start=time.perf_counter();trial=copy.deepcopy(context)
        if fitting['eligible'] and len(cal)>=2 and d%64==0 and 4<=len(cal[-1]['v'][0])<=4096:
            try:
                delta,_=child._temp_fit(cal,trial,qh,kh,d)
                trial=child._temp_guard(cal[-1],trial,delta,qh,kh,d)
            except (RuntimeError,ValueError):fitting['errors']+=1
        seconds=time.perf_counter()-start
        assert not validate_attention_calibration_result(trial,'443')
        new_state=trial['v_state'];state_copy=copy.deepcopy(new_state)
        model=None
        record={'group':gi,'fitting':fitting,'extra_calibration_seconds':seconds,
                'active_heads':int((trial['q_state'].get('temperature_gain',torch.ones(qh))!=1).sum()),'cases':[]}
        for ti,s in enumerate(g['test']):
            raw={r:dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
            qp=base.hif4_dynamic_quantize_q(*s['q'],qh,d,states['q_state'])
            kp=base.hif4_dynamic_quantize_k(*s['k'],kh,d,states['k_state'])
            start=time.perf_counter();vp=base.hif4_dynamic_quantize_v(*s['v'],kh,d,states['v_state']);pt=time.perf_counter()-start
            start=time.perf_counter();vc=child.hif4_dynamic_quantize_v(*s['v'],kh,d,new_state);ct=time.perf_counter()-start
            for name,params in (('q',qp),('k',kp),('v',vp),('v',vc)):
                assert not validate_hif4_params(params,raw[name].shape,name);report['return_checks']+=1
            qc=child.hif4_dynamic_quantize_q(*s['q'],qh,d,trial['q_state'])
            assert not validate_hif4_params(qc,raw['q'].shape,'qc');report['return_checks']+=1
            newq=dequantize_hif4(qc,raw['q'].shape)
            q=dequantize_hif4(qp,raw['q'].shape);k=dequantize_hif4(kp,raw['k'].shape)
            old=dequantize_hif4(vp,raw['v'].shape);new=dequantize_hif4(vc,raw['v'].shape)
            ix=torch.linspace(0,len(raw['q'])-1,min(64,len(raw['q']))).long()
            target=base._joint_output(raw['q'],raw['k'],raw['v'],qh,kh,d,ix)
            output=lambda value:base._joint_output(q,k,value,qh,kh,d,ix)
            a=float((output(old)-target).square().mean());b=float((base._joint_output(newq,k,new,qh,kh,d,ix)-target).square().mean())
            standard={r:dequantize_hif4(quantize_to_hif4(raw[r]),raw[r].shape) for r in 'qkv'}
            bm=float((base._joint_output(standard['q'],standard['k'],standard['v'],qh,kh,d,ix)-target).square().mean())
            row={'test':ti,'mse':[a,b],'baseline_mse':bm,'quality_delta':(a-b)/max(bm,1e-30),
                 'mse_ratio':b/max(a,1e-30),'changed':int((old!=new).sum()+(q!=newq).sum()),'v_seconds':[pt,ct]}
            fresh=load(443,'fresh443')
            assert equal(vc,fresh.hif4_dynamic_quantize_v(*s['v'],kh,d,copy.deepcopy(new_state)))
            assert equal(state_copy,new_state);report['state_checks']+=1
            record['cases'].append(row)
        assert equal(saved,g) and equal(frozen,(context,states))
        report['groups'].append(record)
        (ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n')
        print({'group':gi,'models':fitting,'active':record['active_heads'],
               'delta':sum(r['quality_delta'] for r in record['cases'])},flush=True)
    rows=[r for g in report['groups'] for r in g['cases']]
    report['summary']={'cases':len(rows),'better':sum(r['mse'][1]<r['mse'][0] for r in rows),
                       'worse':sum(r['mse'][1]>r['mse'][0] for r in rows),
                       'changed':sum(r['changed']>0 for r in rows),'quality_delta':sum(r['quality_delta'] for r in rows),
                       'mean_mse_ratio':sum(r['mse_ratio'] for r in rows)/len(rows)}
    report['complete']=True;(ROOT/args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(report['summary'])


if __name__=='__main__':run()
