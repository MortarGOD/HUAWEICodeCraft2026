"""Validate nonlinear risk derivatives and actual fixed-metadata descent."""
import copy
import json
import torch
from probe_v321_query_kl import configured,ROOT
from test_v320_residual import same
from self_check import validate_frozen_state,validate_hif4_params


def main():
    torch.set_num_threads(4);g=torch.Generator().manual_seed(321);m=configured()
    x=torch.randn(7,9,generator=g,dtype=torch.float64,requires_grad=True)
    k=torch.randn(19,9,generator=g,dtype=torch.float64)
    v=torch.randn(19,5,generator=g,dtype=torch.float64)
    target=torch.softmax(torch.randn(7,19,generator=g,dtype=torch.float64),-1)
    logits=x@k.T;prob=torch.softmax(logits,-1)
    loss=(torch.logsumexp(logits,-1)-(target*logits).sum(-1)).sum()
    auto=torch.autograd.grad(loss,x,retain_graph=True)[0]
    kl_error=float((auto-(prob-target)@k).abs().max().detach());assert kl_error<1e-12
    goal=target@v;residual=prob@v-goal;weight=residual@v.T
    auto=torch.autograd.grad(residual.square().sum(),x)[0]
    expected=2*(prob*(weight-(prob*weight).sum(-1,keepdim=True)))@k
    pv_error=float((auto-expected).abs().max().detach());assert pv_error<1e-12
    checks=[];qh,kh,d=4,2,64
    for output in (False,True):
        raw=torch.randn(129,qh*d,generator=g)*.4;p=m._dyn_table(raw,None,has_scale=False)
        key=torch.randn(kh,32,d,generator=g)/d**.5
        bank=dict(raw=key,key=key.clone(),active=torch.tensor([True,False,True,False]),rounds=8,aligned=False)
        if output:
            val=torch.randn(kh,32,16,generator=g)
            bank.update(raw_value=val,value=val.clone())
        assert not validate_frozen_state(bank,'risk')
        snapshot=copy.deepcopy((raw,p,bank));rng=torch.get_rng_state().clone()
        out=m._nq_refine(raw,p,bank,qh,d)
        assert not validate_hif4_params(out,raw.shape,'risk')
        assert same(snapshot,(raw,p,bank)) and torch.equal(rng,torch.get_rng_state())
        rawheads=raw.reshape(-1,qh,d).transpose(0,1);kk=key.repeat_interleave(qh//kh,0)
        target=torch.softmax(rawheads@kk.transpose(1,2),-1)
        def risk(params):
            y=m._deq_params(params).reshape(-1,qh,d).transpose(0,1)
            z=y@kk.transpose(1,2)
            if output:
                vv=bank['value'].repeat_interleave(qh//kh,0)
                return ((torch.softmax(z,-1)-target)@vv).square().sum(-1)
            return torch.logsumexp(z,-1)-(target*z).sum(-1)
        delta=risk(out)-risk(p)
        assert float(delta.max())<1e-5 and float(delta.sum())<0
        assert torch.equal(m._deq_params(out).reshape(-1,qh,d)[:,~bank['active']],m._deq_params(p).reshape(-1,qh,d)[:,~bank['active']])
        checks.append(dict(output=output,sum_delta=float(delta.sum()),worst_delta=float(delta.max())))
    report=dict(passed=True,kl_gradient_max_error=kl_error,pv_gradient_max_error=pv_error,
                input_state_rng_unchanged=True,inactive_heads_equal=True,actual_decoded_descent=checks)
    (ROOT/'docs/v321-query-risk-invariants.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
