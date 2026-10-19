"""Paired-code math and forced active VK tests, not online-distribution claims."""
import copy
import sys
import tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import self_check,validate_hif4_params,validate_attention_calibration_result
from tools.test_dual_operator_isolation import equal


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    path=ROOT/'solution/solution_v414.py'
    assert path.read_text().startswith((ROOT/'solution/solution_v412.py').read_text())
    m=load_solution(path)
    changed=0;scale_changed=0
    for seed in range(16):
        g=torch.Generator().manual_seed(seed)
        t,d,kh=32+seed*8,64,2
        x=torch.randn(t,kh*d,generator=g)*.3
        params=m._dyn_table(x,None,False)
        state=dict(center=torch.zeros(kh*d),projection=torch.randn(kh*d,8,generator=g)*.2,
                   kernel=torch.randn(kh,9,9,generator=g)*2,active=torch.tensor([True,False]),heads=kh)
        before=copy.deepcopy(state);rng=torch.random.get_rng_state().clone()
        p=m._vk_probabilities(x,state,kh)
        y=m._deq_params(params).reshape(t,kh,d).transpose(0,1)
        error=x.reshape(t,kh,d).transpose(0,1)-y
        delta=torch.zeros_like(error)
        delta[:,2]=.03;delta[:,7]=-.02
        h=p.transpose(1,2)@p/p.shape[1]+torch.eye(t)[None]/t**2
        grad=h@error
        increment=-2*(delta*grad).sum(1)+(delta*(h@delta)).sum(1)
        explicit=(p@(error-delta)).square().mean(1)+(error-delta).square().mean(1)/t
        initial=(p@error).square().mean(1)+error.square().mean(1)/t
        assert torch.allclose(increment,explicit-initial,atol=1e-7,rtol=1e-4)
        out=m._vs_scale_refine(x,params,state,kh,d)
        assert not validate_hif4_params(out,x.shape,'pair')
        yy=m._deq_params(out).reshape(t,kh,d).transpose(0,1)
        e=x.reshape(t,kh,d).transpose(0,1)-yy
        final=(p@e).square().mean(1)+e.square().mean(1)/t
        assert bool((final.sum(1)<=initial.sum(1)+1e-7).all())
        assert bool((e.square().mean(1)<=error.square().mean(1)*1.05+1e-8).all())
        assert torch.equal(y[1],yy[1])
        changed+=int((y!=yy).sum())
        scale_changed+=int((out['scale_factor']!=params['scale_factor']).sum())
        assert all(torch.equal(out[k][:,1],params[k][:,1]) for k in params)
        assert equal(state,before) and torch.equal(rng,torch.random.get_rng_state())
    assert changed>0 and scale_changed>0, (changed,scale_changed)
    # Real calibration observes unchanged parent dynamic execution (no early marker).
    group=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[10]
    qh,kh,d=(group[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
    capture=[];parent=m._vs_parent_calibration;dynamic=m._vs_parent_dynamic;calls=[0]
    def cal(*a):
        result=parent(*a);capture.append(copy.deepcopy(result));return result
    def dyn(*a):
        assert not isinstance(a[-1],dict) or 'vk_scale_search' not in a[-1]
        calls[0]+=1;return dynamic(*a)
    m._vs_parent_calibration=cal;m._vs_parent_dynamic=dyn
    states=m.hif4_calibration_attention(group['calib'],qh,kh,d)
    assert calls[0]>0 and not validate_attention_calibration_result(states,'cal')
    assert all(equal(states[r+'_state'],capture[0][r+'_state']) for r in 'qk')
    fresh=load_solution(path)
    for sample in group['test']:
        assert not validate_hif4_params(fresh.hif4_dynamic_quantize_v(*sample['v'],kh,d,states['v_state']),sample['v'][0].shape,'v')
    with tempfile.TemporaryDirectory(prefix='v414-check-') as td:
        (Path(td)/'solution.py').symlink_to(path)
        assert self_check(td)
    print('PASS v414: scale quadratic identity, monotonic objective, cap, inactive head, state/RNG, parent QK, official checks; changed',changed,'scale changes',scale_changed,flush=True)


if __name__=='__main__':main()
