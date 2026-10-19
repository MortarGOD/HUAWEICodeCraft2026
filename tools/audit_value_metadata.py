"""Separate genuine metadata changes from integer-code-only changes."""
import json
from pathlib import Path
import torch
from probe_value_metadata import configured
from self_check import validate_hif4_params,validate_frozen_state


def main():
    root=Path(__file__).resolve().parents[1]
    m=configured(8,cap=1.1,row=True)
    group=torch.load(root/'local_fitted_models_covariance_v1/attn.pt',weights_only=False)[7]
    qh,kh,d=[group[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
    states=m.hif4_calibration_attention(group['calib'],qh,kh,d)
    assert not validate_frozen_state(states,'metadata_audit')
    state=states['v_state'];fallback=state.get('fallback') if 'vm_active' in state else state
    cases=[]
    for sample in group['test']:
        pair=sample['v'];x=m.dequantize_nvfp4(*pair).float()
        parent=m._vm_parent_dynamic(*pair,kh,d,fallback)
        candidate=m.hif4_dynamic_quantize_v(*pair,kh,d,state)
        assert not validate_hif4_params(candidate,x.shape,'metadata_audit')
        e0=x-m._deq_params(parent);e1=x-m._deq_params(candidate)
        mse0=e0.square().mean(0);mse1=e1.square().mean(0)
        cases.append(dict(rows=x.shape[0],changed={k:int((parent[k]!=candidate[k]).sum()) for k in parent},
                          changed_effective_steps=int((m._params_unit_flat(parent)!=m._params_unit_flat(candidate)).sum()),
                          max_channel_mse_ratio=float((mse1/mse0.clamp_min(1e-30)).max())))
    report=dict(group=7,dataset='local_fitted_models_covariance_v1',variance=8,cap=1.1,cases=cases,
                warning='Checks parameter changes, not an isolated online attribution.')
    (root/'docs/value-metadata-parameter-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
