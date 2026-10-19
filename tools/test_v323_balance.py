"""Reciprocal bilinear invariance and inverse-metric congruence tests."""
import copy
import json
import torch
from probe_v323_balance import configured,ROOT
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params
from test_v320_residual import same


def main():
    torch.set_num_threads(4);m=configured();other=configured();rng=torch.Generator().manual_seed(323)
    kh,qh,d=2,4,64;s=(torch.randn(kh,d,generator=rng)*.2).exp()
    u=torch.triu(torch.randn(kh,d,d,generator=rng)*.01)+torch.eye(d)
    state=dict(kvh=kh,gq=1,u=u,H=torch.eye(d).expand(kh,d,d).clone())
    qst=m._sb_state(state,s,'q');kst=m._sb_state(state,s,'k')
    q=torch.randn(11,qh*d,generator=rng);k=torch.randn(17,kh*d,generator=rng)
    qt=m._kt_input(q,qst,qh,d,'q').reshape(-1,kh,qh//kh,d)
    kt=m._kt_input(k,kst,kh,d,'k').reshape(-1,kh,d)
    original=torch.einsum('thrd,shd->hrts',q.reshape(-1,kh,qh//kh,d),k.reshape(-1,kh,d))
    modified=torch.einsum('thrd,shd->hrts',qt,kt)
    assert torch.allclose(original,modified,atol=1e-5,rtol=2e-5)
    for st,f in ((qst,1/s),(kst,s)):
        inv=u.double().transpose(1,2)@u.double()
        got=st['u'].double().transpose(1,2)@st['u'].double()
        target=inv*f.double()[:,:,None]*f.double()[:,None,:]
        assert torch.allclose(got,target,atol=1e-6,rtol=1e-6)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    g=bank[4];qh,kh,d=[g[x] for x in ('q_num_heads','kv_num_heads','head_dim')]
    parent=m._sb_parent_calibration(g['calib'],qh,kh,d);snapshot=copy.deepcopy(parent)
    scale=m._sb_fit(g['calib'],parent,qh,kh,d);candidate=m._sb_guard(g['calib'],parent,qh,kh,d)
    assert same(parent,snapshot) and same(parent['v_state'],candidate['v_state'])
    checks=0
    for role in 'qk':
        base=parent[role+'_state'];nh=qh if role=='q' else kh
        st=dict(sb_new=m._sb_state(base,scale,role),sb_fallback=base,sb_kvh=kh,sb_active=torch.arange(kh)%2==0)
        assert not validate_frozen_state(st,'state')
        for rows in (1,17,129,4097):
            pair=quantize_nvfp4(torch.randn(rows,nh*d,generator=rng)*.3);saved=copy.deepcopy((pair,st));random=torch.get_rng_state().clone()
            out=m._sb_dynamic(pair,nh,d,st,role)
            assert not validate_hif4_params(out,pair[0].shape,role)
            assert same(out,other._sb_dynamic(pair,nh,d,copy.deepcopy(st),role))
            assert same(saved,(pair,st)) and torch.equal(random,torch.get_rng_state())
            fn=m._sb_parent_q if role=='q' else m._sb_parent_k;old=fn(*pair,nh,d,base)
            inactive=(~st['sb_active']).repeat_interleave(nh//kh).repeat_interleave(d//64)
            for name,x in out.items():assert torch.equal(x[:,inactive],old[name][:,inactive])
            if rows in (1,4097):assert same(out,old)
            checks+=1
    report=dict(passed=True,checks=checks,bilinear_max_error=float((original-modified).abs().max()),
                inverse_metric_congruence=True,inputs_rng_v_unchanged=True,independent_modules=True,inactive_boundary_exact=True)
    (ROOT/'docs/v323-balance-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
