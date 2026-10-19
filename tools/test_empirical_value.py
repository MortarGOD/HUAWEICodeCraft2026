"""State, fallback, identity premise, GQA layout and legal lattice checks."""
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from probe_empirical_value import configured
from build_fitted_dataset import quantize_nvfp4
from self_check import validate_frozen_state,validate_hif4_params


def main():
    torch.set_num_threads(4)
    m=configured(8);g=torch.Generator().manual_seed(31691)
    qh,kh,d=4,2,64
    samples=[]
    for rows in (64,96,129):
        kv=quantize_nvfp4(torch.randn(rows,kh*d,generator=g))
        samples.append(dict(q=quantize_nvfp4(torch.randn(rows,qh*d,generator=g)),k=kv,v=kv))
    state=m._empirical_fit(samples,qh,kh,d)
    assert state is not None and not validate_frozen_state(state,'empirical')
    changed=copy.deepcopy(samples)
    changed[-1]['q']=quantize_nvfp4(torch.randn(129,qh*d,generator=g)*3)
    again=m._empirical_fit(changed,qh,kh,d)
    assert torch.equal(state['empirical_queries'],again['empirical_queries'])
    changed=copy.deepcopy(samples)
    changed[0]['k']=[x.clone() for x in changed[0]['k']]
    changed[0]['k'][0][0,0]+=1
    assert m._empirical_fit(changed,qh,kh,d) is None
    state=dict(state,active=torch.tensor([True,False]),fallback=None)
    assert not validate_frozen_state(state,'masked')
    records=[]
    for rows in (1,129,1024,4097):
        pair=quantize_nvfp4(torch.randn(rows,kh*d,generator=g))
        parent=m._empirical_parent_v(*pair,kh,d,None)
        candidate=m.hif4_dynamic_quantize_v(*pair,kh,d,state)
        assert not validate_hif4_params(candidate,pair[0].shape,'candidate')
        for key in parent:
            assert torch.equal(parent[key][:,d//64:],candidate[key][:,d//64:])
            if rows==1 or rows>4096:assert torch.equal(parent[key],candidate[key])
        off=dict(state,active=torch.zeros(kh,dtype=torch.bool))
        disabled=m.hif4_dynamic_quantize_v(*pair,kh,d,off)
        assert all(torch.equal(parent[k],disabled[k]) for k in parent)
        x=m.dequantize_nvfp4(*pair).float()
        if rows<=4096:
            mse0=(x-m._deq_params(parent)).square().mean(0)
            mse1=(x-m._deq_params(candidate)).square().mean(0)
            assert bool((mse1<=mse0*1.05001+1e-18).all())
        if rows==129:
            p=m._vk_probabilities(x,state,kh)
            q=state['empirical_queries'].float()
            k=x.reshape(rows,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
            expected=torch.softmax(q@k.transpose(1,2)/d**.5,-1).reshape(kh,-1,rows)
            assert torch.equal(p,expected)
        records.append(dict(rows=rows,valid=True))
    result=dict(passed=True,checks=records,holdout_query_excluded=True,nonidentity_rejected=True,
                inactive_head_exact=True,query_layout=True,mse_cap=True,
                warning='Prototype invariants, not an online score guarantee.')
    (ROOT/'docs/empirical-value-invariants.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
