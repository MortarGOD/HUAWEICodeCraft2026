"""Compare the analytic full-key metric with exact autograd Jacobians."""
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_pv_full_metric import configured
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_attention_calibration_result, validate_hif4_params


def main():
    torch.set_num_threads(4)
    m=configured(.5)
    generator=torch.Generator().manual_seed(31609)
    d,n,t=8,3,11
    q=torch.randn(n,d,dtype=torch.float64,generator=generator)
    k=torch.randn(t,d,dtype=torch.float64,generator=generator)
    v=torch.randn(t,d,dtype=torch.float64,generator=generator)
    gq,gk=m._pvfull_head_metrics(q,k,v,torch.eye(d,dtype=torch.float64))
    aq,ak=torch.zeros_like(gq),torch.zeros_like(gk)
    for i in range(n):
        jq=torch.autograd.functional.jacobian(lambda x:torch.softmax(x@k.T/d**.5,-1)@v,q[i])
        jk=torch.autograd.functional.jacobian(lambda x:torch.softmax(q[i]@x.T/d**.5,-1)@v,k)
        aq += jq.T@jq/n
        ak += torch.einsum('otd,ote->de',jk,jk)/n
    assert torch.allclose(gq,aq,rtol=1e-10,atol=1e-12), (gq-aq).abs().max()
    assert torch.allclose(gk,ak,rtol=1e-10,atol=1e-12), (gk-ak).abs().max()
    shifted=m._pvfull_head_metrics(q,k,v+7.,torch.eye(d,dtype=torch.float64))
    assert all(torch.allclose(a,b,rtol=1e-10,atol=1e-12) for a,b in zip((gq,gk),shifted))
    zeros=m._pvfull_head_metrics(q,k,torch.ones_like(v),torch.eye(d,dtype=torch.float64))
    assert all(float(x.abs().max())<1e-12 for x in zeros)
    # Per-Q-head metrics must survive GQA state expansion and actual quantization.
    d,qh,kh=64,4,2
    h=torch.eye(d).repeat(kh,1,1)
    state=dict(kvh=kh,rot=0,gq=1,u=m._upper_cholesky_inv(h),rf=1,H=h.bfloat16())
    metric=torch.randn(qh,d,d,generator=generator)
    metric=metric@metric.transpose(1,2)
    qs=m._pvfull_proposal(state,metric,qh)
    ks=m._pvfull_proposal(state,metric.reshape(kh,qh//kh,d,d).sum(1),kh)
    assert not validate_attention_calibration_result(dict(q_state=qs,k_state=ks,v_state=None),'proposed')
    for role,nh,st in (('q',qh,qs),('k',kh,ks)):
        pair=quantize_nvfp4(torch.randn(137,nh*d,generator=generator))
        result=getattr(m,'hif4_dynamic_quantize_'+role)(*pair,nh,d,st)
        assert not validate_hif4_params(result,pair[0].shape,role)
    report=dict(passed=True,exact_q_jacobian_max_error=float((gq-aq).abs().max()),
                exact_k_diagonal_block_max_error=float((gk-ak).abs().max()),
                constant_value_invariance=True,gqa_state_and_format=True,
                warning='The sketched metric and calibration transfer still require empirical validation.')
    (ROOT/'docs/pv-full-metric-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
