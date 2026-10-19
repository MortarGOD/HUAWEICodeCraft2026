"""True paired layout preservation, metric permutation and isolated decoding."""
import copy
import json
import torch
from probe_v323_layout import configured,ROOT
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params
from test_v320_residual import same


def main():
    torch.set_num_threads(4);m=configured();other=configured();rng=torch.Generator().manual_seed(323)
    kh,qh,d=2,4,64;order=torch.stack([torch.randperm(d,generator=rng) for _ in range(kh)])
    u=torch.triu(torch.randn(kh,d,d,generator=rng)*.01)+torch.eye(d)
    state=dict(kvh=kh,gq=1,u=u,H=torch.eye(d).expand(kh,d,d).clone())
    new=m._sb_state(state,order,'q')
    inv=u.double().transpose(1,2)@u.double()
    expected=inv.gather(1,order[:,:,None].expand_as(inv)).gather(2,order[:,None,:].expand_as(inv))
    actual=new['u'].double().transpose(1,2)@new['u'].double()
    assert torch.allclose(expected,actual,atol=1e-6,rtol=1e-6)
    q=torch.randn(11,qh*d,generator=rng);k=torch.randn(17,kh*d,generator=rng)
    qt=m._kt_input(q,new,qh,d,'q').reshape(-1,kh,qh//kh,d)
    kt=m._kt_input(k,new,kh,d,'k').reshape(-1,kh,d)
    old=torch.einsum('thrd,shd->hrts',q.reshape(-1,kh,qh//kh,d),k.reshape(-1,kh,d))
    out=torch.einsum('thrd,shd->hrts',qt,kt)
    assert torch.allclose(old,out,atol=1e-5,rtol=2e-5)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    g=bank[4];qh,kh,d=[g[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
    parent=m._sb_parent_calibration(g['calib'],qh,kh,d);saved=copy.deepcopy(parent)
    order=m._sb_fit(g['calib'],parent,qh,kh,d);candidate=m._sb_guard(g['calib'],parent,qh,kh,d)
    assert same(saved,parent) and same(candidate['v_state'],parent['v_state']);checks=0
    for role in 'qk':
        base=parent[role+'_state'];nh=qh if role=='q' else kh
        st=dict(sb_new=m._sb_state(base,order,role),sb_fallback=base,sb_kvh=kh,sb_active=torch.arange(kh)%2==0)
        assert not validate_frozen_state(st,'state')
        for rows in (1,17,129,4097):
            pair=quantize_nvfp4(torch.randn(rows,nh*d,generator=rng)*.3);snapshot=copy.deepcopy((pair,st));random=torch.get_rng_state().clone()
            result=m._sb_dynamic(pair,nh,d,st,role)
            assert not validate_hif4_params(result,pair[0].shape,role)
            assert same(result,other._sb_dynamic(pair,nh,d,copy.deepcopy(st),role))
            assert same(snapshot,(pair,st)) and torch.equal(random,torch.get_rng_state())
            fn=m._sb_parent_q if role=='q' else m._sb_parent_k;baseline=fn(*pair,nh,d,base)
            inactive=(~st['sb_active']).repeat_interleave(nh//kh).repeat_interleave(d//64)
            for name,value in result.items():assert torch.equal(value[:,inactive],baseline[name][:,inactive])
            if rows in (1,4097):assert same(result,baseline)
            checks+=1
    report=dict(passed=True,checks=checks,bilinear_max_error=float((old-out).abs().max()),metric_congruence=True,
                independent_modules=True,inputs_rng_v_unchanged=True,inactive_boundary_exact=True)
    (ROOT/'docs/v323-layout-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
