"""Canonical coordinate covariance transport and independent-role compatibility."""
import copy,sys,tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import self_check,validate_hif4_params,validate_attention_calibration_result
from tools.test_dual_operator_isolation import equal
from tools.check_role_decoy_calibration import role_samples


def mapping(m,state,role,kh,d):
    st={k:v for k,v in state.items() if k!='temperature_gain'}
    return m._kt_input(torch.eye(d).repeat(1,kh),st,kh,d,role).reshape(d,kh,d).transpose(0,1)


def boundary_checks(m):
    gen=torch.Generator().manual_seed(418)
    for d in (64,96,128,256,512):
        for state in (None,{'kvh':2,'rot':0}):
            a=m._cc_state(state,'q',2,d);b=m._cc_state(state,'k',2,d)
            r=a['joint_transform'];s=b['joint_transform']
            assert torch.allclose(r@s.transpose(-1,-2),torch.eye(d),atol=2e-5,rtol=2e-5)
            x=torch.randn(3,8*d,generator=gen)
            y=m._kt_input(x,a,8,d,'q')
            expected=(x.reshape(3,8,d)@r[0]).reshape_as(x)
            assert torch.equal(y,expected)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    path=ROOT/'solution/solution_v418.py';m=load_solution(path)
    boundary_checks(m)
    assert path.read_text().startswith((ROOT/'solution/solution_v412.py').read_text())
    kh,d=2,64;gen=torch.Generator().manual_seed(417)
    z=torch.randn(kh,d,d,generator=gen)*.1
    h=z@z.transpose(1,2)+torch.eye(d)
    transform=torch.eye(d)[None].repeat(kh,1,1)+torch.randn(kh,d,d,generator=gen)*.02
    original=dict(kvh=kh,gq=1,u=m._upper_cholesky_inv(h),H=h,rf=1,rot=0,
                  joint_transform=transform,qs=torch.rand(kh,d,generator=gen)+.5)
    snapshot=copy.deepcopy(original);rng=torch.random.get_rng_state().clone()
    a=mapping(m,original,'q',kh,d);new=m._cc_state(original,'q',kh,d)
    rotation=m._make_R(d,0)
    c=rotation.T[None]@a
    expected=c@h@c.transpose(1,2)
    assert torch.allclose(new['H'],expected,atol=1e-5,rtol=1e-5)
    e=torch.randn(kh,3,d,generator=gen)
    assert torch.allclose((e@new['H']*e).sum(-1),((e@c)@h*(e@c)).sum(-1),atol=2e-5,rtol=1e-5)
    damp=expected.diagonal(dim1=-2,dim2=-1).mean(-1)*m.GPTQ_DAMP
    inverse=torch.linalg.inv(expected+damp[:,None,None]*torch.eye(d))
    assert torch.allclose(new['u'].transpose(1,2)@new['u'],inverse,atol=1e-4,rtol=1e-4)
    no_h=dict(original);no_h.pop('H')
    assert torch.allclose(m._cc_metric(no_h),h,atol=1e-4,rtol=1e-4)
    assert not m._cc_state(no_h,'q',kh,d)['canonical_metric_fallback']
    bad=dict(original,H=-torch.eye(d)[None].repeat(kh,1,1),joint_transform=torch.eye(d)[None].repeat(kh,1,1))
    fallback=m._cc_state(bad,'q',kh,d)
    assert fallback['canonical_metric_fallback'] and bool((fallback['H'].diagonal(dim1=-2,dim2=-1)>0).all())
    kbase=dict(original);kbase.pop('qs');kbase['ks']=torch.rand(kh,d,generator=gen)+.5
    nested=dict(kbase,kt_mode=4,kt_probe=torch.randn(kh,8,d,generator=gen),kt_active=torch.ones(kh,dtype=torch.bool),
                ts_variant=1,ts_active=torch.tensor([True,False]),ts_fallback=kbase)
    converted=m._cc_state(nested,'k',kh,d)
    assert equal(converted['kt_probe'],nested['kt_probe']) and equal(converted['ts_active'],nested['ts_active'])
    for st in (new,converted,converted['ts_fallback']):
        assert all(k not in st for k in ('qs','ks','rseed'))
        assert torch.equal(mapping(m,st,'k',kh,d),rotation[None].repeat(kh,1,1))
    assert equal(m._cc_state(new,'q',kh,d),new) and equal(original,snapshot)
    assert torch.equal(rng,torch.random.get_rng_state())
    group=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[10]
    qh,kh,d=(group[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
    capture=[];parent=m._vs_parent_calibration
    def cal(*a):
        result=parent(*a);capture.append(copy.deepcopy(result));return result
    m._vs_parent_calibration=cal
    states=m.hif4_calibration_attention(group['calib'],qh,kh,d)
    assert equal(states['v_state'],capture[0]['v_state'])
    assert not validate_attention_calibration_result(states,'canonical')
    separated={}
    for ri,role in enumerate('qk'):
        fresh=load_solution(path)
        st=fresh.hif4_calibration_attention(role_samples(group['calib'],role,'row_permute',20260906+ri),qh,kh,d)
        separated[role]=st[role+'_state']
    aa=mapping(m,separated['q'],'q',kh,d);bb=mapping(m,separated['k'],'k',kh,d)
    assert torch.allclose(aa@bb.transpose(1,2),torch.eye(d)[None].repeat(kh,1,1),atol=2e-5,rtol=2e-5)
    for sample in group['test']:
        for role,nh in (('q',qh),('k',kh)):
            fresh=load_solution(path);st=copy.deepcopy(separated[role]);before=copy.deepcopy(st)
            out=getattr(fresh,'hif4_dynamic_quantize_'+role)(*sample[role],nh,d,st)
            assert not validate_hif4_params(out,sample[role][0].shape,role) and equal(st,before)
    with tempfile.TemporaryDirectory(prefix='v418-check-') as td:
        (Path(td)/'solution.py').symlink_to(path);assert self_check(td)
    print('PASS v418: covariance identity, U recovery, diagonal fallback, nested K, independent AB=I, frozen state, parent V, fresh QK and official checks',flush=True)


if __name__=='__main__':main()
