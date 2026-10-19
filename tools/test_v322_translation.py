"""Mathematical, format, frozen-state and independent-interface checks."""
import ast
import copy
import json
import torch
from probe_v322_translation import ROOT,configured
from probe_v319_value import load_solution
from test_v320_residual import same
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_hif4_params,validate_frozen_state


def main():
    torch.set_num_threads(4);rng=torch.Generator().manual_seed(322)
    # Constant residual is exactly invisible to softmax; weighted covariance
    # must be nonnegative and invariant under another constant offset.
    e=torch.randn(19,2,8,generator=rng,dtype=torch.float64)
    w=torch.softmax(torch.randn(19,2,1,generator=rng,dtype=torch.float64),0)
    a=torch.randn(2,8,8,generator=rng,dtype=torch.float64);h=a@a.transpose(-1,-2)
    def cost(x):
        z=x-(x*w).sum(0,keepdim=True)
        return ((z.transpose(0,1)@h).transpose(0,1)*z*w).sum((0,2))
    c=torch.randn(1,2,8,generator=rng,dtype=torch.float64)
    assert torch.allclose(cost(e),cost(e+c),atol=1e-12) and bool((cost(e)>=0).all())
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    g=bank[47];qh,kh,d=[g[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
    m=configured();other=configured();p=load_solution(ROOT/'solution/solution_v321.py')
    state=m._ts_parent_calibration(g['calib'],qh,kh,d);checks=0
    for variant in (1,2,3):
        st=dict(state['k_state'],ts_variant=variant,ts_active=torch.arange(kh)%2==0,
                kt_active=torch.ones(kh,dtype=torch.bool),kt_mode=2)
        for rows in (1,17,129,4097):
            pair=quantize_nvfp4(torch.randn(rows,kh*d,generator=rng)*.4)
            saved=copy.deepcopy((pair,st));random_state=torch.get_rng_state().clone()
            out=m.hif4_dynamic_quantize_k(*pair,kh,d,st)
            assert not validate_hif4_params(out,pair[0].shape,'k')
            assert not validate_frozen_state(st,'state')
            assert same(saved,(pair,st)) and torch.equal(random_state,torch.get_rng_state())
            assert same(out,other.hif4_dynamic_quantize_k(*pair,kh,d,copy.deepcopy(st)))
            baseline=p.hif4_dynamic_quantize_k(*pair,kh,d,st)
            inactive=(~st['ts_active']).repeat_interleave(d//64)
            for k,v in out.items():assert torch.equal(v[:,inactive],baseline[k][:,inactive])
            if rows in (1,4097):assert same(out,baseline)
            checks+=1
        snap=copy.deepcopy(state);m._TS_VARIANT=variant
        answer=m._ts_guard(g['calib'],state,qh,kh,d)
        assert same(state,snap) and same(state['q_state'],answer['q_state']) and same(state['v_state'],answer['v_state'])
    report=dict(passed=True,checks=checks,covariance_constant_invariant=True,inputs_rng_unchanged=True,
                qv_unchanged=True,inactive_heads_exact=True,independent_modules_exact=True)
    (ROOT/'docs/v322-translation-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
