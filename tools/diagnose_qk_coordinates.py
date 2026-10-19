"""Reproduce continuous-coordinate versus temperature-only error mechanism."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.verify_attention_campaign import role_calibration
from evaluate_local import mse
from hif4_quant import dequantize_nvfp4


@torch.inference_mode()
def run():
    m=load(435,'coord');torch.set_num_threads(4)
    group=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[49]
    qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    st=role_calibration(m,group,True,'block_resample')
    sample=group['test'][0]
    raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
    ix=torch.linspace(0,len(raw['q'])-1,32).long()
    target=m._joint_output(raw['q'],raw['k'],raw['v'],qh,kh,d,ix)
    q=m._kt_input(raw['q'],st['q_state'],qh,d,'q')
    k=m._kt_input(raw['k'],st['k_state'],kh,d,'k')
    gain=st['q_state'].get('temperature_gain',torch.ones(qh))
    qgain=(raw['q'].reshape(-1,qh,d)*gain[None,:,None]).reshape(raw['q'].shape)
    out={'shape':[qh,kh,d],'temperature':gain.tolist(),
         'continuous_parent_mse':mse(m._joint_output(q,k,raw['v'],qh,kh,d,ix),target),
         'temperature_only_mse':mse(m._joint_output(qgain,raw['k'],raw['v'],qh,kh,d,ix),target)}
    (ROOT/'docs/v438-coordinate-mechanism.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))

if __name__=='__main__':run()
