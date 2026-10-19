"""Gauge invariance, calibration coverage, deployment frame and state checks."""
import copy
import json
from pathlib import Path
import torch
from probe_gptq_covariance import configured
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params


def main():
    torch.set_num_threads(4);m=configured('centered_fp32')
    rng=torch.Generator().manual_seed(32073)
    q=torch.randn(5,16,generator=rng,dtype=torch.float64)
    k=torch.randn(19,16,generator=rng,dtype=torch.float64)+2
    shift=torch.randn(16,generator=rng,dtype=torch.float64)*3
    p=torch.softmax(q@k.T/4,-1)
    shifted=torch.softmax(q@(k+shift).T/4,-1)
    assert torch.allclose(p,shifted,atol=1e-14,rtol=1e-12)
    kc=k-k.mean(0);h=kc.T@kc
    delta=torch.randn(5,16,generator=rng,dtype=torch.float64)*.02
    logits=delta@k.T;centered=logits-logits.mean(1,keepdim=True)
    direct=centered.square().sum();quadratic=((delta@h)*delta).sum()
    assert torch.allclose(direct,quadratic,atol=1e-12,rtol=1e-12)
    kh,qh,d=2,4,64
    r=m._make_R(d,0)
    s=torch.rand(kh,d,generator=rng)+.6
    states=dict(q_state=dict(kvh=kh,rot=1,rseed=0,qs=s.bfloat16(),joint_transform=r[None].repeat(kh,1,1)),
                k_state=dict(kvh=kh,rot=1,rseed=0,ks=(1/s).bfloat16(),joint_transform=r[None].repeat(kh,1,1)),v_state=None)
    samples=[]
    for rows in (31,1057,64):
        samples.append({role:quantize_nvfp4(torch.randn(rows,(qh if role=='q' else kh)*d,generator=rng)) for role in 'qkv'})
    saved=copy.deepcopy(samples)
    # Compare exactly against the transform executed by the dynamic function.
    original=m._dyn_table;frame_errors={}
    for role,nh in (('q',qh),('k',kh)):
        observed=[]
        def capture(x,*args,**kwargs):
            observed.append(x.clone());return original(x,*args,**kwargs)
        m._dyn_table=capture
        pair=samples[0][role]
        result=getattr(m,'hif4_dynamic_quantize_'+role)(*pair,nh,d,states[role+'_state'])
        assert not validate_hif4_params(result,pair[0].shape,'frame')
        x=m.dequantize_nvfp4(*pair).float()
        expected=m._cov_transform(x,states[role+'_state'],nh,kh,d,role).transpose(0,1).reshape_as(x)
        frame_errors[role]=float((expected-observed[0]).abs().max())
        assert torch.equal(expected,observed[0])
    m._dyn_table=original
    m._COV_MODE='legacy';legacy,_=m._cov_fit(samples,states,qh,kh,d)
    assert m.COV_DEBUG['skipped']==[1057]
    m._COV_MODE='all';all_h,_=m._cov_fit(samples,states,qh,kh,d)
    assert len(m.COV_DEBUG['included'])==2 and m.COV_DEBUG['skipped']==[]
    assert not torch.equal(legacy,all_h)
    m._COV_MODE='centered';center_h,k_h=m._cov_fit(samples,states,qh,kh,d)
    h0=center_h.clone()
    changed=copy.deepcopy(samples)
    changed[-1]['k']=quantize_nvfp4(torch.randn(64,kh*d,generator=rng)*10)
    again,_=m._cov_fit(changed,states,qh,kh,d)
    assert torch.equal(h0,again)
    proposed=m._cov_proposal(dict(states['q_state'],gq=1,H=all_h.bfloat16()),center_h,qh)
    assert proposed is not None and not validate_frozen_state(proposed,'proposed')
    assert proposed['u'].shape==(qh,d,d) and proposed['H'].dtype==torch.float32
    dynamic=getattr(m,'hif4_dynamic_quantize_q')(*samples[0]['q'],qh,d,proposed)
    assert not validate_hif4_params(dynamic,samples[0]['q'][0].shape,'gqa_proposal')
    for old,current in zip(saved,samples):
        assert all(torch.equal(a,b) for role in 'qkv' for a,b in zip(old[role],current[role]))
    report=dict(passed=True,gauge_error=float((p-shifted).abs().max()),quadratic_error=float((direct-quadratic).abs()),
                deployed_frame_error=frame_errors,long_sample_included=True,holdout_excluded=True,
                state_valid=True,fp32_H_valid=True,gqa_valid=True,calibration_immutable=True)
    (Path(__file__).resolve().parents[1]/'docs/gptq-covariance-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
