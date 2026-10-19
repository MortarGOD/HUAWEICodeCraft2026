"""Post-push negative-case attribution using the exact parent state for both arms."""
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.test_v422 import load
from tools.verify_v426 import equal
from evaluate_local import gqa_attention,mse
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4


@torch.inference_mode()
def main():
    torch.set_num_threads(4);parent,child=load(422,'fixed'),load(426,'fixed')
    groups=torch.load(ROOT/'local_fitted_models_covariance_v1/attn.pt',weights_only=False)
    previous=json.loads((ROOT/'docs/v426-attention-covariance.json').read_text())
    recorded={(c['group'],c['test']):c for c in previous['cases']}
    results=[]
    for gi in (3,7):
        g=groups[gi];qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
        states=parent.hif4_calibration_attention(g['calib'],qh,kh,d);snapshot=copy.deepcopy(states)
        for ti,sample in enumerate(g['test']):
            raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
            decoded={r:dequantize_hif4(getattr(parent,'hif4_dynamic_quantize_'+r)(*sample[r],qh if r=='q' else kh,d,states[r+'_state']),raw[r].shape) for r in 'qkv'}
            fixed_k=dequantize_hif4(child.hif4_dynamic_quantize_k(*sample['k'],kh,d,copy.deepcopy(states['k_state'])),raw['k'].shape)
            ref=gqa_attention(raw['q'],raw['k'],raw['v'],qh,kh,d)
            pm=mse(ref,gqa_attention(decoded['q'],decoded['k'],decoded['v'],qh,kh,d))
            cm=mse(ref,gqa_attention(decoded['q'],fixed_k,decoded['v'],qh,kh,d))
            old=recorded[(gi,ti)];same_parent=abs(pm-old['mse'][0])<=1e-7*max(pm,old['mse'][0],1e-12)
            assert same_parent,('parent reproduction mismatch',gi,ti,pm,old['mse'][0])
            results.append({'group':gi,'test':ti,'tokens':raw['q'].shape[0],'parent_mse':pm,'fixed_parent_state_child_k_mse':cm,
                'full_child_mse':old['mse'][1],'fixed_state_delta':(pm-cm)/old['baseline_mse'],
                'full_child_delta':old['delta'],'parent_reproduced':same_parent,
                'key_values_changed':not torch.equal(decoded['k'],fixed_k)})
        assert equal(states,snapshot)
    report={'completed':True,'selection':'Groups 3 and 7 selected after observing negative covariance results; diagnostic only, never a deployment route.',
        'scope':'All Q/K/V use the same parent coordinate system and frozen states. Only the child K quantizer is substituted; no mismatched learned transforms are mixed.',
        'cases':results,'summary':{'cases':len(results),'fixed_state_delta':sum(x['fixed_state_delta'] for x in results),'full_child_delta':sum(x['full_child_delta'] for x in results)}}
    (ROOT/'docs/v426-fixed-state-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary']))


if __name__=='__main__':main()
