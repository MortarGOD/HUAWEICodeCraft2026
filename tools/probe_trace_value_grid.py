"""Offline V-only scale-grid intervention preserving all v435 postprocessing."""
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.verify_attention_campaign import equal
from hif4_quant import dequantize_nvfp4,dequantize_hif4
from self_check import validate_hif4_params

@torch.inference_mode()
def run():
    torch.set_num_threads(4);m=load(435,'trace_grid')
    bank=torch.load(ROOT/'docs/papers/hif4-public-references/test/datasets/qwen3_5_35b_a3b_v1/attn.pt',weights_only=True)
    original=m._dyn_table;grid=tuple((off,sig) for off in range(-1,3) for sig in m.E6M2_SIG)
    def wide(x,state,has_scale):
        return m._quantize_weighted(x,torch.ones(1,x.shape[1]),grid=grid)
    rows=[]
    for gi,g in enumerate(bank):
        qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        st=m.hif4_calibration_attention(g['calib'],qh,kh,d);frozen=copy.deepcopy(st)
        for ti,s in enumerate(g['test']):
            raw={r:dequantize_nvfp4(*s[r]).float() for r in 'qkv'}
            quant={r:getattr(m,'hif4_dynamic_quantize_'+r)(*s[r],qh if r=='q' else kh,d,st[r+'_state']) for r in 'qkv'}
            try:
                m._dyn_table=wide
                candidate=m.hif4_dynamic_quantize_v(*s['v'],kh,d,st['v_state'])
            finally:m._dyn_table=original
            assert not validate_hif4_params(candidate,raw['v'].shape,'wide')
            decoded={r:dequantize_hif4(quant[r],raw[r].shape) for r in 'qkv'}
            value=dequantize_hif4(candidate,raw['v'].shape)
            ix=torch.arange(len(raw['v']))
            target=m._joint_output(raw['q'],raw['k'],raw['v'],qh,kh,d,ix)
            before=m._joint_output(decoded['q'],decoded['k'],decoded['v'],qh,kh,d,ix)
            after=m._joint_output(decoded['q'],decoded['k'],value,qh,kh,d,ix)
            native=dequantize_hif4(original(raw['v'],None,False),raw['v'].shape)
            expanded=dequantize_hif4(wide(raw['v'],None,False),raw['v'].shape)
            base=float((native-raw['v']).square().mean());new=float((expanded-raw['v']).square().mean())
            assert new<=base+max(1e-10,base*1e-6)
            a=float((before-target).square().mean());b=float((after-target).square().mean())
            rows.append({'group':gi,'test':ti,'mse':[a,b],'native_tensor_ratio':new/max(base,1e-30),
                         'output_ratio':b/max(a,1e-30),'changed_values':int((value!=decoded['v']).sum())})
        assert equal(frozen,st)
    report={'completed':True,'cases':rows,'mean_output_ratio':sum(r['output_ratio'] for r in rows)/len(rows),
        'mean_native_tensor_ratio':sum(r['native_tensor_ratio'] for r in rows)/len(rows),
        'better':sum(r['output_ratio']<1 for r in rows),'worse':sum(r['output_ratio']>1 for r in rows),
        'changed_cases':sum(r['changed_values']>0 for r in rows),
        'warning':'Offline serial hook only; no production mutable dispatch. Shared parent calibration, only V primitive grid differs. No acceptance gate; this is not an online score prediction.'}
    (ROOT/'docs/qwen35-value-grid.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='cases'})
if __name__=='__main__':run()
