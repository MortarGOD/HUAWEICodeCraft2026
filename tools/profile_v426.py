"""Uninstrumented paired phase timing and unchanged Linear contracts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.verify_v426 import equal
from self_check import validate_hif4_params, validate_linear_calibration_result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--output',default='docs/v426-profile.json');a=p.parse_args()
    report={'source_sha256':hashlib.sha256((ROOT/'solution/solution_v426.py').read_bytes()).hexdigest(),
        'completed':False,'attention':[],'linear':[],'synthetic_helper':[],
        'warning':'This machine, 4 torch threads; no online score or Kunpeng wall-time extrapolation. Synthetic helper tests force refinement.'}
    target=ROOT/a.output
    def save():target.write_text(json.dumps(report,indent=2)+'\n')
    modules={v:load(v,'phase') for v in (422,426)}
    groups=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    for repeat in range(2):
        for gi in (0,25,49):
            g=groups[gi];qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
            initial=torch.get_rng_state().clone()
            for ver in ((422,426) if (repeat+gi)%2==0 else (426,422)):
                m=modules[ver];torch.set_rng_state(initial)
                start=time.perf_counter();states=m.hif4_calibration_attention(g['calib'],qh,kh,d);cal=time.perf_counter()-start
                dt={r:0. for r in 'qkv'}
                for sample in g['test']:
                    for role in 'qkv':
                        start=time.perf_counter()
                        getattr(m,'hif4_dynamic_quantize_'+role)(*sample[role],qh if role=='q' else kh,d,states[role+'_state'])
                        dt[role]+=time.perf_counter()-start
                report['attention'].append({'group':gi,'repeat':repeat,'version':ver,'calibration':cal,**dt})
            save()
    linear=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/linear.pt',weights_only=False)
    for gi,g in enumerate(linear):
        before=copy.deepcopy(g);states={};times={};initial=torch.get_rng_state().clone()
        for ver in ((422,426) if gi%2==0 else (426,422)):
            torch.set_rng_state(initial);start=time.perf_counter()
            states[ver]=modules[ver].hif4_calibration_and_quantize_weight(*g['weight'],g['calib_activation_list'])
            times[ver]={'calibration':time.perf_counter()-start,'dynamic':0.}
            errors=validate_linear_calibration_result(states[ver],g['weight'][0].shape,'weight');assert not errors,errors
        assert equal(states[422],states[426]),('Linear calibration',gi)
        snapshot=copy.deepcopy(states)
        for ai,pair in enumerate(g['test_activation_list']):
            answers={}
            for ver in (422,426):
                start=time.perf_counter();answers[ver]=modules[ver].hif4_dynamic_quantize_activation(*pair,states[ver]['activation_state'])
                times[ver]['dynamic']+=time.perf_counter()-start
                assert not validate_hif4_params(answers[ver],pair[0].shape,'activation')
            assert equal(answers[422],answers[426]),('Linear dynamic',gi,ai)
        assert equal(snapshot,states) and equal(before,g)
        report['linear'].append({'group':gi,'cases':len(g['test_activation_list']),'weight_state_activation_exact':True,'timing':times})
        save()
    rng=torch.Generator().manual_seed(426500)
    for t,b,d in ((64,4,64),(512,4,256),(2048,2,256),(4096,2,512)):
        x=torch.randn(t,b*d,generator=rng);unit=torch.ones_like(x)*2;val=(x/.5).round().clamp(-7,7)*.5
        z=torch.randn(b,d,d,generator=rng);h=z@z.transpose(1,2)/d+torch.eye(d)[None]*.1
        rounds=32 if t<=128 else (4 if t>1024 else 8)
        times={422:[],426:[]}
        for rep in range(3):
            for ver in ((422,426) if rep%2==0 else (426,422)):
                kw={'quotient':True} if ver==426 else {}
                start=time.perf_counter();modules[ver]._refine_attn_heads(x,val,unit,h,b,1,total_rounds=rounds,**kw)
                times[ver].append(time.perf_counter()-start)
        report['synthetic_helper'].append({'tokens':t,'heads':b,'dim':d,'rounds':rounds,'seconds':times})
        save()
    report['completed']=True;save();print('PASS: phase timing and all Linear identities',flush=True)


if __name__=='__main__':main()
