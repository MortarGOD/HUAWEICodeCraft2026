"""Sensitivity identity, normalization, state isolation and legal output checks."""
import copy
import json
import torch
from probe_v323_sensitivity import ROOT,configured
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params
from test_v320_residual import same


def main():
    torch.set_num_threads(4);rng=torch.Generator().manual_seed(323);m=configured();other=configured()
    z=torch.randn(13,generator=rng,dtype=torch.float64,requires_grad=True)
    p=z.softmax(-1);jac=torch.autograd.functional.jacobian(lambda x:x.softmax(-1),z)
    formula=p.square()*(1-2*p+p.square().sum())
    assert torch.allclose(jac.square().sum(0),formula,atol=1e-14)
    for probs in (torch.softmax(torch.randn(2,9,17,generator=rng),-1),
                  torch.ones(2,9,17)/17,torch.nn.functional.one_hot(torch.zeros(2,9,dtype=torch.long),17).float()):
        for mode in (1,2,3):
            w=m._ss_weights(probs,mode)
            assert bool(torch.isfinite(w).all()) and bool((w>=0).all())
            assert torch.allclose(w.sum(-1),torch.ones(2),atol=1e-6)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    g=bank[4];qh,kh,d=[g[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
    parent=m._ss_parent_calibration(g['calib'],qh,kh,d);saved=copy.deepcopy(parent);checks=0
    for mode in (1,2,3):
        m._SS_MODE=mode;answer=m._ss_guard(g['calib'],parent,qh,kh,d)
        assert same(parent,saved) and same(answer['q_state'],parent['q_state']) and same(answer['v_state'],parent['v_state'])
        base=parent['k_state'];st=dict(base,ss_mode=mode,ss_active=torch.arange(kh)%2==0,ss_fallback=base)
        if 'kt_probe' not in st:st['kt_probe']=torch.randn(kh,32,d,generator=rng)/d**.5
        assert not validate_frozen_state(st,'key')
        for t in (1,17,129,4097):
            pair=quantize_nvfp4(torch.randn(t,kh*d,generator=rng)*.4);snapshot=copy.deepcopy((pair,st));random=torch.get_rng_state().clone()
            out=m.hif4_dynamic_quantize_k(*pair,kh,d,st)
            assert not validate_hif4_params(out,pair[0].shape,'k')
            assert same(out,other.hif4_dynamic_quantize_k(*pair,kh,d,copy.deepcopy(st)))
            assert same(snapshot,(pair,st)) and torch.equal(random,torch.get_rng_state())
            old=m._ss_parent_k(*pair,kh,d,base);inactive=(~st['ss_active']).repeat_interleave(d//64)
            for name,value in out.items():assert torch.equal(value[:,inactive],old[name][:,inactive])
            if t in (1,4097):assert same(out,old)
            checks+=1
    report=dict(passed=True,checks=checks,jacobian_column_squared_norm_error=float((jac.square().sum(0)-formula).abs().max().detach()),
                normalized_finite_extremes=True,independent_modules=True,inactive_and_boundary_exact=True,inputs_rng_qv_unchanged=True)
    (ROOT/'docs/v323-sensitivity-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
