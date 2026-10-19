"""Probability representatives and exact weighted-risk identity tests."""
import copy,sys,tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import self_check,validate_hif4_params,validate_attention_calibration_result
from tools.test_dual_operator_isolation import equal


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    path=ROOT/'solution/solution_v416.py';m=load_solution(path)
    assert path.read_text().startswith((ROOT/'solution/solution_v412.py').read_text())
    generator=torch.Generator().manual_seed(416)
    t,qh,kh,d=73,4,2,64
    x=torch.randn(t,kh*d,generator=generator)*.2
    state=dict(center=torch.zeros(kh*d),projection=torch.randn(kh*d,8,generator=generator),
               kernel=torch.randn(qh,9,9,generator=generator)*.4,
               query_sigma_modes=torch.randn(qh,2,9,generator=generator)*.1,
               active=torch.ones(kh,dtype=torch.bool),heads=qh,probability_representatives=True)
    snapshot=copy.deepcopy(state);rng=torch.random.get_rng_state().clone()
    f=m._vk_features(x,state);indices,weights=m._rs_select(f,state['kernel'])
    assert indices.shape==(qh,4) and bool(((indices>=0)&(indices<t)).all())
    assert all(len(torch.unique(row))==4 for row in indices)
    assert torch.allclose(weights.sum(1),torch.ones(qh),atol=1e-6) and bool((weights>=0).all())
    assert not torch.equal(indices,torch.linspace(0,t-1,4).long()[None].expand(qh,-1))
    logits=(f[indices]@state['kernel'])@f.T
    delta=state['query_sigma_modes']@f.T*2**.5
    offsets=torch.cat((delta,-delta),1)
    probs=torch.softmax(logits[:,:,None]+offsets[:,None],-1)
    factors=m._vk_probabilities(x,state,kh)
    error=torch.randn(kh,t,d,generator=generator)
    direct=torch.einsum('hijt,htd->hijd',probs,error.repeat_interleave(qh//kh,0))
    direct=(direct.square().mean(2)*weights[:,:,None]).sum(1).reshape(kh,qh//kh,d).mean(1)
    assert torch.allclose((factors@error).square().mean(1),direct,atol=1e-6,rtol=1e-5)
    # Identical probability signatures collapse their mass, not their risk.
    identical=dict(state,kernel=torch.zeros_like(state['kernel']),query_sigma_modes=torch.zeros_like(state['query_sigma_modes']))
    uniform=m._vk_probabilities(x,identical,kh)
    assert torch.allclose((uniform@error).square().mean(1),error.mean(1).square(),atol=1e-6,rtol=1e-5)
    unmarked=dict(state);unmarked.pop('probability_representatives')
    assert torch.equal(m._vk_probabilities(x,unmarked,kh),m._rs_parent_probabilities(x,unmarked,kh))
    assert torch.equal(m._vk_probabilities(x[:4],state,kh),m._rs_parent_probabilities(x[:4],state,kh))
    params=m._dyn_table(x,None,False)
    result=m._vk_refine(x,params,state,kh,d)
    assert not validate_hif4_params(result,x.shape,'representative')
    assert equal(state,snapshot) and torch.equal(rng,torch.random.get_rng_state())
    group=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[10]
    qh,kh,d=(group[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
    capture=[];parent=m._vs_parent_calibration;prob=m._rs_parent_probabilities;calls=[0]
    def cal(*a):
        out=parent(*a);capture.append(copy.deepcopy(out));return out
    def observed(*a):
        assert not a[1].get('probability_representatives')
        calls[0]+=1;return prob(*a)
    m._vs_parent_calibration=cal;m._rs_parent_probabilities=observed
    states=m.hif4_calibration_attention(group['calib'],qh,kh,d)
    assert calls[0]>0 and all(equal(states[r+'_state'],capture[0][r+'_state']) for r in 'qk')
    assert not validate_attention_calibration_result(states,'cal')
    fresh=load_solution(path)
    for sample in group['test']:
        assert not validate_hif4_params(fresh.hif4_dynamic_quantize_v(*sample['v'],kh,d,states['v_state']),sample['v'][0].shape,'v')
    with tempfile.TemporaryDirectory(prefix='v416-check-') as td:
        (Path(td)/'solution.py').symlink_to(path);assert self_check(td)
    print('PASS v416: unique representatives, masses, GQA weighted risk, identical signatures, parent fallback, fresh V, frozen state/RNG, QK and official checks',flush=True)


if __name__=='__main__':main()
