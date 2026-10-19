"""Exact inactive heads, calibration separation, gradients and valid outputs."""
import copy
import json
from pathlib import Path
import torch
from probe_query_distill import configured
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params


def main():
    torch.set_num_threads(4);m=configured('both',.02,2,16)
    rng=torch.Generator().manual_seed(32373);qh,kh,d=4,2,64
    a=torch.randn(kh,d,d,generator=rng)
    h=a.transpose(1,2)@a+torch.eye(d)[None]
    u=m._upper_cholesky_inv(h)
    st=dict(kvh=kh,gq=1,u=u,H=h.bfloat16(),rf=1,rot=1,rseed=0)
    states=dict(q_state=st,k_state=copy.deepcopy(st),v_state=None)
    samples=[]
    for rows in (17,24,32):
        samples.append({r:quantize_nvfp4(torch.randn(rows,(qh if r=='q' else kh)*d,generator=rng)) for r in 'qkv'})
    saved=copy.deepcopy(samples);global_rng=torch.random.get_rng_state().clone()
    gain,bias=m._qd_train(samples,states,qh,kh,d)
    assert torch.equal(global_rng,torch.random.get_rng_state())
    history=copy.deepcopy(m.QD_DEBUG['training_history'])
    assert all(all(b<=a+1e-6 for a,b in zip(history[i],history[i+1])) for i in range(len(history)-1))
    assert not validate_frozen_state(dict(gain=gain,bias=bias),'query_distill')
    assert not gain.requires_grad and not bias.requires_grad
    assert float(gain.abs().max())<=m._QD_CAP+1e-6
    altered=copy.deepcopy(samples)
    for r in 'qkv':
        altered[-1][r]=quantize_nvfp4(torch.randn(altered[-1][r][0].shape,generator=rng)*5)
    with torch.inference_mode():
        again=m._qd_train(altered,states,qh,kh,d)
    assert torch.equal(gain,again[0]) and torch.equal(bias,again[1])
    for now,old in zip(samples,saved):
        assert all(torch.equal(a,b) for r in 'qkv' for a,b in zip(now[r],old[r]))
    checks=0;changed=False;subset_error=0.
    for rows in (1,65,129,1025):
        pair=quantize_nvfp4(torch.randn(rows,qh*d,generator=rng))
        base=m._qd_parent_q(*pair,qh,d,st)
        zero=dict(st,qd_gain=torch.zeros(qh,d),qd_bias=torch.zeros(qh,d))
        off=m.hif4_dynamic_quantize_q(*pair,qh,d,zero)
        assert all(torch.equal(base[k],off[k]) for k in base)
        active=dict(st,qd_gain=torch.zeros(qh,d),qd_bias=torch.zeros(qh,d))
        active['qd_gain'][0]=.08;active['qd_bias'][0]=.02
        out=m.hif4_dynamic_quantize_q(*pair,qh,d,active)
        assert not validate_hif4_params(out,pair[0].shape,'query_distill')
        for k in base:
            assert torch.equal(base[k][:,d//64:],out[k][:,d//64:])
        if rows==1:assert all(torch.equal(base[k],out[k]) for k in base)
        else:changed=changed or any(not torch.equal(base[k],out[k]) for k in base)
        if rows>1:
            ix=torch.linspace(0,rows-1,min(rows,16)).long()
            x=m._cov_transform(m.dequantize_nvfp4(*pair).float()[ix],active,qh,kh,d,'q')
            subset=m._qd_quant(x,{k:v[ix] for k,v in base.items()},active,qh,d,rows,
                               active['qd_gain'],active['qd_bias'])
            subset_error=max(subset_error,float((m._deq_params(subset)-m._deq_params(out)[ix]).abs().max()))
        checks+=1
    assert changed
    assert subset_error==0
    joint=dict(st,joint_transform=torch.eye(d)[None].repeat(kh,1,1)*1.1,
               qs=torch.full((kh,d),.9),qd_gain=torch.full((qh,d),.03),qd_bias=torch.full((qh,d),.01))
    joint_out=m.hif4_dynamic_quantize_q(*samples[0]['q'],qh,d,joint)
    assert not validate_hif4_params(joint_out,samples[0]['q'][0].shape,'joint_query_distill')
    checks+=1
    # Compare the smooth surrogate's derivative to an independently constructed
    # output expression; this is not a claim that STE is a true discrete gradient.
    x=torch.randn(qh,3,d,generator=rng,dtype=torch.float64)
    k=torch.randn(qh,5,d,generator=rng,dtype=torch.float64)
    v=torch.randn(qh,5,d,generator=rng,dtype=torch.float64)
    target=torch.randn(qh,3,d,generator=rng,dtype=torch.float64)
    ga=torch.zeros(qh,d,dtype=torch.float64,requires_grad=True)
    bi=torch.zeros_like(ga,requires_grad=True)
    smooth=x*(1+ga[:,None])+bi[:,None]
    quant=(x*4).round()/4
    straight=quant+(smooth-smooth.detach())
    assert torch.equal(straight,quant)
    loss=m._qd_loss(straight,dict(k=k,v=v,target=target),d).sum()
    gradient=torch.autograd.grad(loss,(ga,bi))
    q=quant.detach().requires_grad_(True)
    output=torch.softmax(q@k.transpose(1,2)/d**.5,-1)@v
    direct=torch.autograd.grad((output-target).square().mean((1,2)).sum(),q)[0]
    gradient_error=max(float((gradient[0]-(direct*x).sum(1)).abs().max()),float((gradient[1]-direct.sum(1)).abs().max()))
    assert gradient_error<1e-12
    report=dict(passed=True,format_returns=checks,inactive_heads_exact=True,zero_parameters_exact=True,
                single_token_exact=True,active_branch_changes=True,holdout_excluded=True,calibration_immutable=True,
                global_rng_unchanged=True,inference_mode_supported=True,state_no_grad=True,training_monotone=True,
                training_history=history,ste_forward_exact=True,surrogate_gradient_error=gradient_error,
                query_subset_error=subset_error,joint_transform_and_scales=True)
    (Path(__file__).resolve().parents[1]/'docs/query-distill-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
