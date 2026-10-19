"""Actual activated state: fresh role modules, parent Q/K identity and no state mutation."""
import copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_attention_calibration_result,validate_hif4_params


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(v,w) for v,w in zip(a,b))
    return a==b


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    path=ROOT/'solution/solution_v405.py'
    m=load_solution(path)
    g=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[10]
    qh,kh,d=(g[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
    parent=[];original=m._do_parent_calibration
    def capture(*a):
        result=original(*a);parent.append(copy.deepcopy(result));return result
    m._do_parent_calibration=capture
    states=m.hif4_calibration_attention(g['calib'],qh,kh,d)
    assert not validate_attention_calibration_result(states,'isolated')
    assert 'dual_operator' in states['v_state']
    assert all(equal(states[r+'_state'],parent[0][r+'_state']) for r in 'qk')
    snapshot=copy.deepcopy(states);rng=torch.random.get_rng_state().clone()
    cases=0
    for sample in g['test']:
        for r in 'qkv':
            nh=qh if r=='q' else kh
            role=load_solution(path)
            ans=getattr(role,'hif4_dynamic_quantize_'+r)(*sample[r],nh,d,copy.deepcopy(states[r+'_state']))
            same=getattr(m,'hif4_dynamic_quantize_'+r)(*sample[r],nh,d,states[r+'_state'])
            assert equal(ans,same)
            assert not validate_hif4_params(ans,sample[r][0].shape,r)
            cases+=1
    assert equal(snapshot,states) and torch.equal(rng,torch.random.get_rng_state())
    summary=dict(role_results=cases,parent_qk_identical=True,state_and_rng_unchanged=True,
                 active_heads=int(states['v_state']['dual_operator']['active'].sum()))
    (ROOT/'docs/v405-isolation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('PASS',summary)


if __name__=='__main__':main()
