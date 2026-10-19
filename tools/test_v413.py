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
    path=ROOT/'solution/solution_v413.py'
    assert path.read_text().startswith((ROOT/'solution/solution_v412.py').read_text())
    m=load_solution(path)
    # A concrete coupled escape: every nonzero one-code move raises risk,
    # but opposite moves at two tokens lower it (uniform P plus diagonal ridge).
    seed_params=m._dyn_table(torch.ones(32,64),None,False)
    zero_params=m._values_to_params(torch.zeros(32,64),seed_params)
    steps=m._params_unit_flat(zero_params)*.25
    escape_x=torch.zeros(32,64)
    escape_x[2,0]=.6*steps[2,0];escape_x[7,0]=-.6*steps[7,0]
    escape_state=dict(center=torch.zeros(64),projection=torch.zeros(64,1),
                      kernel=torch.zeros(1,2,2),active=torch.ones(1,dtype=torch.bool),heads=1)
    ep=m._vk_probabilities(escape_x,escape_state,1)
    eh=ep.transpose(1,2)@ep/ep.shape[1]+torch.eye(32)[None]/32**2
    eg=(eh@escape_x[None])[0]
    ed=eh[0].diagonal()[:,None]
    assert bool((ed*steps.square()-2*eg*steps>0).all())
    assert bool((ed*steps.square()+2*eg*steps>0).all())
    escape_out=m._vs_pair_refine(escape_x,zero_params,escape_state,1,64)
    escape_y=m._deq_params(escape_out)
    assert int((escape_y!=0).sum())==2
    old=(ep@escape_x[None]).square().mean(1)+escape_x.square().mean(0)/32
    new=(ep@(escape_x-escape_y)[None]).square().mean(1)+(escape_x-escape_y).square().mean(0)/32
    assert bool((new<=old+1e-10).all()) and new[0,0]<old[0,0]
    changed=0
    for seed in range(8):
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
        out=m._vs_pair_refine(x,params,state,kh,d)
        assert not validate_hif4_params(out,x.shape,'pair')
        yy=m._deq_params(out).reshape(t,kh,d).transpose(0,1)
        e=x.reshape(t,kh,d).transpose(0,1)-yy
        final=(p@e).square().mean(1)+e.square().mean(1)/t
        assert bool((final<=initial+1e-7).all())
        assert bool((e.square().mean(1)<=error.square().mean(1)*1.05+1e-8).all())
        assert torch.equal(y[1],yy[1])
        changed+=int((y!=yy).sum())
        assert equal(state,before) and torch.equal(rng,torch.random.get_rng_state())
    assert changed>0
    # Real calibration observes unchanged parent dynamic execution (no early marker).
    group=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[10]
    qh,kh,d=(group[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
    capture=[];parent=m._vs_parent_calibration;dynamic=m._vs_parent_dynamic;calls=[0]
    def cal(*a):
        result=parent(*a);capture.append(copy.deepcopy(result));return result
    def dyn(*a):
        assert not isinstance(a[-1],dict) or 'paired_code_search' not in a[-1]
        calls[0]+=1;return dynamic(*a)
    m._vs_parent_calibration=cal;m._vs_parent_dynamic=dyn
    states=m.hif4_calibration_attention(group['calib'],qh,kh,d)
    assert calls[0]>0 and not validate_attention_calibration_result(states,'cal')
    assert all(equal(states[r+'_state'],capture[0][r+'_state']) for r in 'qk')
    fresh=load_solution(path)
    for sample in group['test']:
        assert not validate_hif4_params(fresh.hif4_dynamic_quantize_v(*sample['v'],kh,d,states['v_state']),sample['v'][0].shape,'v')
    with tempfile.TemporaryDirectory(prefix='v413-check-') as td:
        (Path(td)/'solution.py').symlink_to(path)
        assert self_check(td)
    print('PASS v413: pair quadratic identity, monotonic objective, cap, inactive head, state/RNG, parent QK, official checks; changed',changed,flush=True)


if __name__=='__main__':main()
