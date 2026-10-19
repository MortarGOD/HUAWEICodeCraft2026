"""Attribution control for known repeated-calibration Q-state instability."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.verify_attention_campaign import equal,role_calibration
from hif4_quant import dequantize_hif4,dequantize_nvfp4,quantize_to_hif4
from evaluate_local import gqa_attention,mse
from self_check import validate_hif4_params


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    parent,child=load(430,'fixed'),load(435,'fixed')
    groups=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    report={'completed':False,'groups':[0,24,49],'decoy':'block_resample','cases':[],
            'note':'One parent role-isolated calibration per group; V-only fixed-state attribution, not independent full calibration.',
            'source_sha256':hashlib.sha256((ROOT/'solution/solution_v435.py').read_bytes()).hexdigest()}
    path=ROOT/'docs/v435-fixed-state-block_resample.json'
    for gi in report['groups']:
        group=groups[gi]
        qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        state=role_calibration(parent,group,True,'block_resample')
        frozen=copy.deepcopy(state)
        child_state={'v435_content_parent':copy.deepcopy(state['v_state'])}
        for ti,sample in enumerate(group['test']):
            raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            baseline={r:dequantize_hif4(quantize_to_hif4(raw[r]),raw[r].shape) for r in 'qkv'}
            target=gqa_attention(raw['q'],raw['k'],raw['v'],qh,kh,d)
            bm=mse(target,gqa_attention(baseline['q'],baseline['k'],baseline['v'],qh,kh,d))
            deployed={r:dequantize_hif4(getattr(parent,'hif4_dynamic_quantize_'+r)(
                *sample[r],qh if r=='q' else kh,d,state[r+'_state']),raw[r].shape) for r in 'qkv'}
            params=child.hif4_dynamic_quantize_v(*sample['v'],kh,d,child_state)
            assert not validate_hif4_params(params,raw['v'].shape,'V')
            value=dequantize_hif4(params,raw['v'].shape)
            m0=mse(target,gqa_attention(deployed['q'],deployed['k'],deployed['v'],qh,kh,d))
            m1=mse(target,gqa_attention(deployed['q'],deployed['k'],value,qh,kh,d))
            report['cases'].append({'group':gi,'test':ti,'delta':(m0-m1)/bm if bm>0 else 0.,
                                    'parent_mse':m0,'child_mse':m1,'baseline_mse':bm,
                                    'changed_v':int((value!=deployed['v']).sum())})
        assert equal(frozen,state) and equal(child_state['v435_content_parent'],state['v_state'])
        path.write_text(json.dumps(report,indent=2)+'\n')
        print({'group':gi,'delta':sum(r['delta'] for r in report['cases'])},flush=True)
    rows=report['cases']
    report['summary']={'cases':len(rows),'delta':sum(r['delta'] for r in rows),
                       'better':sum(r['delta']>1e-12 for r in rows),'worse':sum(r['delta']<-1e-12 for r in rows),
                       'equal':sum(abs(r['delta'])<=1e-12 for r in rows)}
    report['completed']=True;path.write_text(json.dumps(report,indent=2)+'\n');print(report['summary'])


if __name__=='__main__':main()
