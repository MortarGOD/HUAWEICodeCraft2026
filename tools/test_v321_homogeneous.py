"""Independent-interface, unchanged-path and continuous-geometry checks."""
import copy
import json
import torch
from probe_v321_homogeneous import configured, ROOT, load_solution
from test_v320_residual import same
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state, validate_hif4_params


def main():
    torch.set_num_threads(4)
    m=configured();gen=torch.Generator().manual_seed(321)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    checks=0;geometry=[]
    for gi in (14,19):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._hc_parent_calibration(g['calib'],qh,kh,d)
        snap=copy.deepcopy(parent)
        for rank in (0,1,4):
            st=m._hc_fit(g['calib'],parent,qh,kh,d,rank)
            assert same(parent,snap) and not validate_frozen_state(st,'hc')
            s=g['test'][0]
            raw=[m.dequantize_nvfp4(*s[r]).float() for r in 'qk']
            q,k=[x.reshape(-1,h,d).transpose(0,1) for x,h in zip(raw,(qh,kh))]
            qn=m._hc_encode(raw[0],st['q_state'],qh,d,'q').reshape(-1,qh,d).transpose(0,1)
            kn=m._hc_encode(raw[1],st['k_state'],kh,d,'k').reshape(-1,kh,d).transpose(0,1)
            kc=(k-k.mean(1,keepdim=True)).repeat_interleave(qh//kh,0)
            before=q@kc.transpose(1,2);after=qn@kn.repeat_interleave(qh//kh,0).transpose(1,2)
            # Direct affine identity: all residual is exactly the dropped bilinear part
            # plus the represented mean offset; no hidden current-role exchange.
            tq=st['q_state']['hc_matrix'].repeat_interleave(qh//kh,0)
            tk=st['k_state']['hc_matrix'].repeat_interleave(qh//kh,0)
            bias=st['q_state']['hc_bias']
            pred=(q@(tq@tk.transpose(1,2)-torch.eye(d))+bias[:,None]@tk.transpose(1,2))@kc.transpose(1,2)
            rel=float(((after-before)-pred).norm()/before.norm().clamp_min(1e-20))
            loss=float((after-before).norm()/before.norm().clamp_min(1e-20))
            assert rel<1e-4,(gi,rank,rel)
            if rank==0:assert loss<1e-4,(gi,loss)
            geometry.append(dict(group=gi,rank=rank,identity_relative_error=rel,continuous_relative_change=loss))
            for role,nh in (('q',qh),('k',kh)):
                old=m._hc_parent_q if role=='q' else m._hc_parent_k
                raw0=m.dequantize_nvfp4(*s[role]).float()
                assert same(old(*s[role],nh,d,parent[role+'_state']),m._hc_quantize_raw(raw0,parent[role+'_state'],nh,d,role))
                checks+=1
        # Force mixed heads; active and inactive paths must both remain valid.
        state=m._hc_fit(g['calib'],parent,qh,kh,d,1)
        for role,nh in (('q',qh),('k',kh)):
            st=state[role+'_state'];st['hc_active'][1:]=False
            for key in ('u','H'):st[key][1:]=parent[role+'_state'][key][1:]
            independent=configured()
            for rows in (1,17,129,1025,4097):
                pair=quantize_nvfp4(torch.randn(rows,nh*d,generator=gen)*.3)
                snapshot=copy.deepcopy((pair,st));rng=torch.get_rng_state().clone()
                fn=getattr(m,'hif4_dynamic_quantize_'+role)
                out=fn(*pair,nh,d,st)
                assert not validate_hif4_params(out,pair[0].shape,role)
                assert same(snapshot,(pair,st)) and torch.equal(rng,torch.get_rng_state())
                assert same(out,getattr(independent,'hif4_dynamic_quantize_'+role)(*pair,nh,d,copy.deepcopy(st)))
                checks+=1
    report=dict(passed=True,interface_checks=checks,geometry=geometry,parent_state_unchanged=True,
                inputs_and_rng_unchanged=True,independent_module_equal=True)
    (ROOT/'docs/v321-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':main()
