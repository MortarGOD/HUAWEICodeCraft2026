"""Focused invariants for the statically embedded joint Attention candidate."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_attention_joint import configured
from self_check import validate_attention_calibration_result, validate_hif4_params


def main():
    m=configured()
    bank=torch.load(ROOT/'local_fitted_models_covariance_v1/attn.pt',weights_only=False)
    records=[]
    for gi in (5,7,25,30):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        state=m.hif4_calibration_attention(g['calib'],qh,kh,d)
        errors=validate_attention_calibration_result(state,'joint')
        assert not errors, errors
        # Test the learned branch even when the calibration guard rejects it.
        vk=m._vk_fit(g['calib'],qh,kh,d)
        for ti,sample in enumerate(g['test']):
            x=m.dequantize_nvfp4(*sample['v']).float()
            p0=m._vk_parent_v(*sample['v'],None,kh,d)
            p1=m._vk_refine(x,p0,vk,kh,d)
            errors=validate_hif4_params(p1,x.shape,'forced_kernel')
            assert not errors, errors
            a,b=m._deq_params(p0),m._deq_params(p1)
            mse0,mse1=(x-a).square().mean(0),(x-b).square().mean(0)
            assert bool((mse1<=mse0*1.05001+1e-18).all())
            assert all(torch.equal(p0[k],p1[k]) for k in ('scale_factor','scale_lv2','scale_lv3'))
            if x.shape[0]==1:assert torch.equal(a,b)
            off=dict(vk,active=torch.zeros(kh,dtype=torch.bool))
            assert torch.equal(m._deq_params(m._vk_refine(x,p0,off,kh,d)),a)
        proposed=m._joint_balanced_states(state,qh,kh,d)
        if proposed is not None:
            tq=proposed['q_state']['joint_transform']
            tk=proposed['k_state']['joint_transform']
            inverse_error=float((tq@tk.transpose(-1,-2)-torch.eye(d)).abs().max())
            assert inverse_error<1e-4,inverse_error
        else:inverse_error=None
        hold=g['calib'][-1];ix=torch.linspace(0,hold['q'][0].shape[0]-1,64).long()
        for st in (state,proposed):
            if st is None:continue
            full=m._deq_params(m.hif4_dynamic_quantize_q(*hold['q'],qh,d,st['q_state']))[ix]
            small=m._joint_quant_queries(hold,st['q_state'],qh,kh,d,ix)
            rel=float((full-small).square().mean().sqrt()/full.square().mean().sqrt())
            assert rel<0.02,rel  # Different GEMM dispatch may change boundary codes.
            records.append(dict(group=gi,query_subset_relative_rms=rel,transform_inverse_error=inverse_error))
    for version in (310,311,312):
        from probe_persistent_covariance import load
        z=load(version)
        for rows in (2,10,128,1024):
            params=z.hif4_dynamic_quantize_v(torch.zeros(rows,512),torch.ones(rows,32),1,512,None)
            assert not validate_hif4_params(params,(rows,512),'zero')
    report=dict(passed=True,forced_v_cases=20,zero_boundary_cases=12,details=records,
                warning='Numerical boundary checks did not reproduce the online v311 loss; no per-case receipt available.')
    (ROOT/'docs/v314-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
