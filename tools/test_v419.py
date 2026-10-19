"""Boundary-budget correction: exact algebra, finite format and frozen parent."""
import copy
import sys
import tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import self_check,validate_hif4_params,validate_attention_calibration_result
from tools.test_dual_operator_isolation import equal


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    path=ROOT/'solution/solution_v419.py';m=load_solution(path)
    assert path.read_text().startswith((ROOT/'solution/solution_v412.py').read_text())
    gen=torch.Generator().manual_seed(419)
    p=torch.softmax(torch.randn(2,4,12,generator=gen,dtype=torch.float64),-1)
    steps=torch.rand(2,12,5,generator=gen,dtype=torch.float64)+.1
    e=torch.randn(2,12,5,generator=gen,dtype=torch.float64)
    lift,chol=m._rb_factor(p,steps)
    actual=lift@torch.cholesky_solve(p@e,chol)
    diag=steps.square().mean(-1)
    gram=(p*diag[:,None])@p.transpose(1,2)
    ridge=gram.diagonal(dim1=-2,dim2=-1).mean(-1)*.01
    full=p.transpose(1,2)@p+torch.diag_embed(ridge[:,None]/diag)
    expected=torch.linalg.solve(full,p.transpose(1,2)@(p@e))
    assert torch.allclose(actual,expected,atol=1e-10,rtol=1e-10)
    changed=0
    for seed in range(16):
        g=torch.Generator().manual_seed(seed)
        t,d,kh=32+seed*8,64,2
        x=torch.randn(t,kh*d,generator=g)*.3
        initial=m._dyn_table(x,None,False)
        state=dict(center=torch.zeros(kh*d),projection=torch.randn(kh*d,8,generator=g)*.2,
                   kernel=torch.randn(kh,9,9,generator=g)*2,active=torch.tensor([True,False]),heads=kh)
        params=m._vk_refine(x,initial,state,kh,d)
        before=copy.deepcopy(state);rng=torch.random.get_rng_state().clone()
        probability=m._vk_probabilities(x,state,kh)
        y=m._deq_params(params).reshape(t,kh,d).transpose(0,1)
        error=x.reshape(t,kh,d).transpose(0,1)-y
        out=m._rb_refine(x,params,state,kh,d)
        assert not validate_hif4_params(out,x.shape,'boundary')
        yy=m._deq_params(out).reshape(t,kh,d).transpose(0,1)
        ee=x.reshape(t,kh,d).transpose(0,1)-yy
        old=(probability@error).square().mean(1)+error.square().mean(1)/t
        new=(probability@ee).square().mean(1)+ee.square().mean(1)/t
        assert bool((new<=old+1e-7).all())
        assert bool((ee.square().mean(1)<=error.square().mean(1)*1.05+1e-8).all())
        assert torch.equal(y[1],yy[1])
        assert torch.equal(m._params_unit_flat(params),m._params_unit_flat(out))
        assert equal(m._rb_refine(x,params,dict(state,active=torch.zeros(kh,dtype=torch.bool)),kh,d),params)
        changed+=int((y!=yy).sum())
        assert equal(state,before) and torch.equal(rng,torch.random.get_rng_state())
    assert changed>0
    group=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[10]
    qh,kh,d=(group[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
    capture=[];parent=m._vs_parent_calibration;dynamic=m._vs_parent_dynamic;calls=[0]
    def cal(*a):
        result=parent(*a);capture.append(copy.deepcopy(result));return result
    def dyn(*a):
        assert not isinstance(a[-1],dict) or 'residual_boundary_search' not in a[-1]
        calls[0]+=1;return dynamic(*a)
    m._vs_parent_calibration=cal;m._vs_parent_dynamic=dyn
    states=m.hif4_calibration_attention(group['calib'],qh,kh,d)
    assert calls[0]>0 and not validate_attention_calibration_result(states,'cal')
    assert all(equal(states[r+'_state'],capture[0][r+'_state']) for r in 'qk')
    fresh=load_solution(path)
    for sample in group['test']:
        st=copy.deepcopy(states['v_state']);snapshot=copy.deepcopy(st)
        out=fresh.hif4_dynamic_quantize_v(*sample['v'],kh,d,st)
        assert not validate_hif4_params(out,sample['v'][0].shape,'v') and equal(st,snapshot)
    with tempfile.TemporaryDirectory(prefix='v419-check-') as td:
        (Path(td)/'solution.py').symlink_to(path);assert self_check(td)
    print('PASS v419: explicit normal equation, fixed grid, monotonic risk/cap, inactive, frozen state/RNG, parent QK and official checks; changed',changed,flush=True)


if __name__=='__main__':main()
