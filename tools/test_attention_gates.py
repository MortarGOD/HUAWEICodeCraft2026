"""Candidate routing, full-key scoring and calibration-statistic invariants."""
import copy
import json
from pathlib import Path
import torch
from probe_attention_gates import configured
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params


def main():
    torch.set_num_threads(4);m=configured('deployed')
    rng=torch.Generator().manual_seed(32473);qh,kh,d=4,2,64
    a=torch.randn(kh,d,d,generator=rng);h=a.transpose(1,2)@a+torch.eye(d)[None]
    st=dict(kvh=kh,rot=1,rseed=0,gq=1,u=m._upper_cholesky_inv(h),H=h.bfloat16(),rf=1)
    bank,labels=m._gate_bank(st,h,128)
    long_bank,_=m._gate_bank(st,h,2048)
    assert all(x['rf']==1 for x in long_bank)
    assert bank[0] is st and torch.equal(bank[0]['u'],st['u'])
    empty,empty_labels=m._gate_bank(dict(kvh=kh,rot=0),torch.zeros_like(h),128)
    assert len(empty)==1 and empty_labels==['parent']
    checks=0;changed=False;subset_error=0.
    for role,heads in (('q',qh),('k',kh)):
        parent=getattr(m,'_gate_parent_'+role)
        for rows in (1,129,1025):
            pair=quantize_nvfp4(torch.randn(rows,heads*d,generator=rng))
            p=parent(*pair,heads,d,st)
            zero=dict(gate_bank=bank,gate_choice=torch.zeros(heads,dtype=torch.int64))
            z=m._gate_dynamic(*pair,heads,d,zero,parent)
            assert all(torch.equal(p[k],z[k]) for k in p)
            choice=torch.arange(heads)%len(bank)
            state=dict(gate_bank=bank,gate_choice=choice)
            assert not validate_frozen_state(state,'gate')
            out=m._gate_dynamic(*pair,heads,d,state,parent)
            assert not validate_hif4_params(out,pair[0].shape,'gate_'+role)
            for head in range(heads):
                expected=p if rows==1 else parent(*pair,heads,d,bank[int(choice[head])])
                sl=slice(head*d//64,(head+1)*d//64)
                assert all(torch.equal(out[k][:,sl],expected[k][:,sl]) for k in out)
            changed=changed or any(not torch.equal(p[k],out[k]) for k in out)
            checks+=1
            if role=='q' and rows>1:
                ix=torch.linspace(0,rows-1,min(rows,16)).long()
                for variant in bank:
                    full=m._deq_params(parent(*pair,heads,d,variant))[ix]
                    subset=m._gate_query_values(dict(q=pair),variant,qh,kh,d,ix)
                    subset_error=max(subset_error,float((full-subset).abs().max()))
    assert changed
    assert subset_error==0
    calls=[]
    def counted(*args):
        calls.append(args[-1]);return m._gate_parent_q(*args)
    only=dict(gate_bank=bank,gate_choice=torch.ones(qh,dtype=torch.int64))
    pair=quantize_nvfp4(torch.randn(9,qh*d,generator=rng))
    m._gate_dynamic(*pair,qh,d,only,counted)
    assert len(calls)==1 and calls[0] is bank[1]
    # A shared K proposal helps only when a different Q variant is chosen.
    costs=torch.full((2,2,2,qh),2.)
    costs[0,0]=1.;costs[1,0,:,:2]=.9
    costs[0,1,:,:2]=1.1;costs[1,1,:,:2]=.7
    costs[1,1,:,2:]=float('nan')
    qi,ki=m._gate_select(costs,kh)
    assert qi.tolist()==[1,1,0,0] and ki.tolist()==[1,0]
    original=torch.full((2,2,2,qh),1.)
    q0,k0=m._gate_select(original,kh)
    assert not bool(q0.any() or k0.any())
    # One fold getting worse must reject a candidate even if its mean improves.
    original[1,0,0,:]=.1;original[1,0,1,:]=1.1
    q0,k0=m._gate_select(original,kh)
    assert not bool(q0.any() or k0.any())
    states=dict(q_state=st,k_state=copy.deepcopy(st),v_state=None)
    samples=[]
    for rows in (17,24,32):
        samples.append({r:quantize_nvfp4(torch.randn(rows,(qh if r=='q' else kh)*d,generator=rng)) for r in 'qkv'})
    saved=copy.deepcopy(samples);global_rng=torch.random.get_rng_state().clone()
    gram=m._gate_fit(samples,states,qh,kh,d)
    altered=copy.deepcopy(samples)
    for role in 'qkv':altered[-1][role]=quantize_nvfp4(torch.randn(altered[-1][role][0].shape,generator=rng)*10)
    second=m._gate_fit(altered,states,qh,kh,d)
    assert all(torch.equal(x,y) for x,y in zip(gram,second))
    assert torch.equal(global_rng,torch.random.get_rng_state())
    assert all(torch.equal(x,y) for now,old in zip(samples,saved) for r in 'qkv' for x,y in zip(now[r],old[r]))
    frames,frame_labels=m._gate_frame_candidates(samples,states,qh,kh,d,32)
    other,other_labels=m._gate_frame_candidates(altered,states,qh,kh,d,32)
    assert frame_labels==other_labels and len(frames)>=3
    assert not validate_frozen_state(frames,'frame_candidates')
    for f1,f2 in zip(frames,other):
        for role in 'qk':
            for key,value in f1[role+'_state'].items():
                rhs=f2[role+'_state'][key]
                assert torch.equal(value,rhs) if isinstance(value,torch.Tensor) else value==rhs
    frame_costs=torch.ones(3,2,qh)
    frame_costs[1,:,:2]=.8;frame_costs[2,:,2:]=.7
    frame_choice=m._gate_select_frames(frame_costs,kh)
    assert frame_choice.tolist()==[1,2]
    # A single-head cap is optional; the scored sum and both folds are not.
    tradeoff=torch.ones(2,2,qh)
    tradeoff[1,:,:2]=torch.tensor([.5,1.05])
    strict=m._gate_select_frames(tradeoff,kh)
    assert strict.tolist()==[0,0]
    m._GATE_HEAD_CAP=0
    relaxed=m._gate_select_frames(tradeoff,kh)
    assert relaxed.tolist()==[1,0]
    tradeoff[1,1,:2]=torch.tensor([1.1,1.1])
    assert m._gate_select_frames(tradeoff,kh).tolist()==[0,0]
    m._GATE_HEAD_CAP=1.01
    qs=dict(gate_bank=[f['q_state'] for f in frames],gate_choice=frame_choice.repeat_interleave(qh//kh))
    ks=dict(gate_bank=[f['k_state'] for f in frames],gate_choice=frame_choice)
    assert torch.equal(qs['gate_choice'].reshape(kh,qh//kh),ks['gate_choice'][:,None].expand(kh,qh//kh))
    for role,state,heads in (('q',qs,qh),('k',ks,kh)):
        output=getattr(m,'hif4_dynamic_quantize_'+role)(*samples[0][role],heads,d,state)
        assert not validate_hif4_params(output,samples[0][role][0].shape,'frame_'+role)
        checks+=1
    # Removing keys changes the normalization even though Q rows are independent.
    q=torch.tensor([[1.,0.]],dtype=torch.float64)
    k=torch.tensor([[1.,0.],[2.,0.],[3.,0.]],dtype=torch.float64)
    v=torch.tensor([[1.,0.],[0.,1.],[2.,2.]],dtype=torch.float64)
    full=torch.softmax(q@k.T,-1)@v
    shortened=torch.softmax(q@k[::2].T,-1)@v[::2]
    assert not torch.allclose(full,shortened)
    report=dict(passed=True,format_returns=checks,zero_choices_exact=True,mixed_heads_exact=True,
                single_token_exact=True,active_branch_changes=True,existing_matrices_preserved=True,
                no_long_hold_refinement_change=True,no_zero_gram_enable=True,gqa_joint_selection=True,
                nonfinite_rejected=True,both_folds_required=True,holdout_excluded_from_new_grams=True,
                calibration_immutable=True,global_rng_unchanged=True,key_subsampling_not_invariant=True,
                matched_frame_gqa=True,frame_holdout_excluded=True,frame_state_valid=True,
                query_subset_error=subset_error,unused_parent_call_removed=True)
    (Path(__file__).resolve().parents[1]/'docs/attention-gates-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
