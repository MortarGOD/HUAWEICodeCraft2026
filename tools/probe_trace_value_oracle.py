"""Offline-only true-probability V search; prohibited information for deployment."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.attention_token_transfer import frozen_token_step
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4
from self_check import validate_hif4_params

@torch.inference_mode()
def run():
    torch.set_num_threads(4);m=load(435,'trace_oracle')
    bank=torch.load(ROOT/'docs/papers/hif4-public-references/test/datasets/qwen3_5_35b_a3b_v1/attn.pt',weights_only=True)
    records=[]
    for gi in (0,2,4,6,9):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        state=m.hif4_calibration_attention(g['calib'],qh,kh,d)
        for ti,s in enumerate(g['test']):
            raw={r:dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
            params={r:getattr(m,'hif4_dynamic_quantize_'+r)(*s[r],qh if r=='q' else kh,d,state[r+'_state']) for r in 'qkv'}
            decoded={r:dequantize_hif4(params[r],raw[r].shape) for r in 'qkv'}
            t=len(raw['v'])
            p=m._vr_probs(raw['q'],raw['k'],qh,kh,d).reshape(kh,-1,t)
            pq=m._vr_probs(decoded['q'],decoded['k'],qh,kh,d).reshape(kh,-1,t)
            x=raw['v'].reshape(t,kh,d).transpose(0,1)
            y=decoded['v'].reshape(t,kh,d).transpose(0,1)
            step=(m._params_unit_flat(params['v'])*.25).reshape(t,kh,d).transpose(0,1)
            h=pq[:,::2].transpose(-1,-2)@pq[:,::2]/pq[:,::2].shape[1]
            trial=y.clone()
            for _ in range(32):trial=frozen_token_step(x,trial,step,h)
            candidate=m._values_to_params(trial.transpose(0,1).reshape(t,-1).contiguous(),params['v'])
            assert not validate_hif4_params(candidate,raw['v'].shape,'oracle')
            actual=dequantize_hif4(candidate,raw['v'].shape).reshape(t,kh,d).transpose(0,1)
            target=p@x;before=pq@y-target;after=pq@actual-target
            records.append({'group':gi,'test':ti,'fit_mse':[float(before[:,::2].square().mean()),float(after[:,::2].square().mean())],
                'heldout_mse':[float(before[:,1::2].square().mean()),float(after[:,1::2].square().mean())],
                'changed_values':int((actual!=y).sum())})
    ratios=[r['heldout_mse'][1]/r['heldout_mse'][0] for r in records]
    report={'completed':True,'cases':records,'mean_heldout_ratio':sum(ratios)/len(ratios),
            'better':sum(v<1 for v in ratios),'worse':sum(v>1 for v in ratios),
            'warning':'Offline true test Q/K only; unavailable to independent V API. 32 fixed-scale steps, 0.7H plus 0.3I metric. Query split is not sequence generalization or an achievable online bound.'}
    (ROOT/'docs/qwen35-value-oracle.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='cases'})
if __name__=='__main__':run()
