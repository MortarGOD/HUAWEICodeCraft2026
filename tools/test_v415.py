"""Joint kernel scenarios: GQA order, holdout boundary and parent isolation."""
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
    path=ROOT/'solution/solution_v415.py';m=load_solution(path)
    assert path.read_text().startswith((ROOT/'solution/solution_v412.py').read_text())
    group=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[10]
    samples=[]
    for sample in group['calib']:
        samples.append({role:(sample[role][0][:64,:channels].contiguous(),sample[role][1][:64,:channels//16].contiguous())
                        for role,channels in (('q',256),('k',128),('v',128))})
    qh,kh,d=4,2,64
    st=m._vk_fit(samples,qh,kh,d);before=copy.deepcopy(st)
    kernels=m._jk_fit(samples,st,qh,kh,d)
    assert kernels is not None and kernels.shape[0]==min(4,len(samples)-1)
    altered=copy.deepcopy(samples)
    altered[-1]={r:(torch.zeros_like(a),torch.ones_like(b)) for r,(a,b) in altered[-1].items()}
    assert torch.allclose(kernels.float(),m._jk_fit(altered,st,qh,kh,d).float(),atol=1e-6,rtol=1e-6)
    assert m._jk_fit(samples[:2],st,qh,kh,d) is None
    state=dict(st,joint_kernel_scenarios=kernels)
    assert not validate_attention_calibration_result(dict(q_state={},k_state={},v_state=state),'joint')
    x=m.dequantize_nvfp4(*samples[-1]['v']).float();f=m._vk_features(x,state)
    ix=torch.linspace(0,x.shape[0]-1,min(m._VK_QUERIES,x.shape[0])).long()
    actual=m._vk_probabilities(x,state,kh)
    per_head=[]
    for head in range(qh):
        per_head.append(torch.cat([torch.softmax((f[ix]@k[head].float())@f.T,-1) for k in kernels],0))
    expected=torch.stack(per_head).reshape(kh,-1,x.shape[0])
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-5)
    assert torch.allclose(actual.sum(-1),torch.ones_like(actual.sum(-1)),atol=1e-6)
    assert torch.equal(m._vk_probabilities(x,st,kh),m._jk_parent_probabilities(x,st,kh))
    identical=dict(st,joint_kernel_scenarios=st['kernel'][None].repeat(3,1,1,1))
    point=m._vu_parent_probabilities(x,st,kh).reshape(qh,-1,x.shape[0])
    repeated=point[:,None].expand(-1,3,-1,-1).reshape(kh,-1,x.shape[0])
    assert torch.allclose(m._vk_probabilities(x,identical,kh),repeated,atol=1e-6,rtol=1e-5)
    snapshot=copy.deepcopy(state);rng=torch.random.get_rng_state().clone()
    fresh=load_solution(path)
    result=fresh.hif4_dynamic_quantize_v(*samples[-1]['v'],kh,d,state)
    assert not validate_hif4_params(result,x.shape,'v')
    assert equal(state,snapshot) and equal(st,before) and torch.equal(rng,torch.random.get_rng_state())
    qh,kh,d=(group[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
    capture=[];parent=m._vs_parent_calibration;prob=m._jk_parent_probabilities;calls=[0]
    def cal(*a):
        result=parent(*a);capture.append(copy.deepcopy(result));return result
    def observed(*a):
        assert 'joint_kernel_scenarios' not in a[1]
        calls[0]+=1;return prob(*a)
    m._vs_parent_calibration=cal;m._jk_parent_probabilities=observed
    states=m.hif4_calibration_attention(group['calib'],qh,kh,d)
    assert calls[0]>0 and all(equal(states[r+'_state'],capture[0][r+'_state']) for r in 'qk')
    assert not validate_attention_calibration_result(states,'real')
    with tempfile.TemporaryDirectory(prefix='v415-check-') as td:
        (Path(td)/'solution.py').symlink_to(path);assert self_check(td)
    print('PASS v415: joint fitting, holdout boundary, GQA scenario order, normalized P, identical-kernel limit, fresh V, parent QK and official checks',flush=True)


if __name__=='__main__':main()
