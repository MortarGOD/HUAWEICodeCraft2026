"""Reciprocal geometry, independent interventions and legal dynamic states."""
import copy
import json
from pathlib import Path
import torch
from frame_causal_core import variants,geometry
from probe_attention_gates import configured
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params


def main():
    torch.set_num_threads(4);m=configured('frames',0)
    rng=torch.Generator().manual_seed(32573);qh,kh,d=4,2,64
    r=m._make_R(d,0);t=torch.diag_embed(torch.rand(kh,d,generator=rng)+.6)
    s=(torch.rand(kh,d,generator=rng)+.7).bfloat16()
    a=torch.randn(kh,d,d,generator=rng);h=a.transpose(1,2)@a+torch.eye(d)[None]
    u=m._upper_cholesky_inv(h)
    common=dict(kvh=kh,rot=1,rseed=0,gq=1,rf=1,u=u,H=h.bfloat16())
    parent=dict(q_state=dict(common,qs=s,joint_transform=r@t),
                k_state=dict(common,ks=s.float().reciprocal().bfloat16(),joint_transform=r@torch.linalg.inv(t).transpose(1,2)),v_state=None)
    samples=[{role:quantize_nvfp4(torch.randn(rows,(qh if role=='q' else kh)*d,generator=rng)) for role in 'qkv'} for rows in (13,21,32)]
    saved=copy.deepcopy(samples);before=copy.deepcopy(parent);global_rng=torch.random.get_rng_state().clone()
    states=variants(m,samples,parent,qh,kh,d)
    assert states['parent'] is parent
    assert states['reciprocal_only']['q_state'] is parent['q_state']
    for field in parent['k_state']:
        if field!='ks':
            x=parent['k_state'][field];y=states['reciprocal_only']['k_state'][field]
            assert torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y
    old=geometry(m,parent,kh,d);new=geometry(m,states['reciprocal_only'],kh,d)
    assert old['max_abs_identity_error']>1e-4
    assert new['max_abs_identity_error']<1e-6
    assert new['max_abs_identity_error']<old['max_abs_identity_error']/1000
    assert 'qs' not in states['no_scale']['q_state'] and 'ks' not in states['no_scale']['k_state']
    assert torch.equal(states['no_scale']['q_state']['joint_transform'],parent['q_state']['joint_transform'])
    assert 'joint_transform' not in states['no_dense']['q_state']
    assert torch.equal(states['no_dense']['q_state']['qs'],parent['q_state']['qs'])
    altered=copy.deepcopy(samples)
    for role in 'qkv':altered[-1][role]=quantize_nvfp4(torch.randn(altered[-1][role][0].shape,generator=rng)*10)
    again=variants(m,altered,parent,qh,kh,d)
    returns=0
    for label,st in states.items():
        assert not validate_frozen_state(st,'causal_'+label)
        for role in 'qk':
            for key,value in st[role+'_state'].items():
                rhs=again[label][role+'_state'][key]
                assert torch.equal(value,rhs) if isinstance(value,torch.Tensor) else value==rhs
            heads=qh if role=='q' else kh
            out=getattr(m,'_gate_parent_'+role)(*samples[0][role],heads,d,st[role+'_state'])
            assert not validate_hif4_params(out,samples[0][role][0].shape,'causal_'+role)
            returns+=1
    assert torch.equal(global_rng,torch.random.get_rng_state())
    for role in 'qk':
        for key,value in parent[role+'_state'].items():
            original=before[role+'_state'][key]
            assert torch.equal(value,original) if isinstance(value,torch.Tensor) else value==original
    assert all(torch.equal(a,b) for now,old_sample in zip(samples,saved) for role in 'qkv' for a,b in zip(now[role],old_sample[role]))
    report=dict(passed=True,old_geometry=old,corrected_geometry=new,reciprocal_only_single_field=True,
                separate_scale_dense_interventions=True,holdout_excluded=True,parent_immutable=True,
                samples_immutable=True,global_rng_unchanged=True,format_returns=returns,states=len(states))
    (Path(__file__).resolve().parents[1]/'docs/frame-causal-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
