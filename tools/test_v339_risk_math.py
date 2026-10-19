"""Exact block output-risk labels and deployable candidate legality."""
import copy
import torch
from probe_v319_value import ROOT,load_solution
from probe_linear_deployed_joint import tree_equal
from self_check import validate_hif4_params


def main():
    torch.set_num_threads(4);rng=torch.Generator().manual_seed(339)
    p=torch.softmax(torch.randn(3,7,19,generator=rng,dtype=torch.float64),-1)
    b=torch.randn(3,19,128,generator=rng,dtype=torch.float64)
    target=torch.randn(3,7,128,generator=rng,dtype=torch.float64)
    delta=torch.randn(3,19,128,generator=rng,dtype=torch.float64)*.1
    e=p@b-target;change=p@delta
    label=(2*e*change+change.square()).reshape(3,7,2,64).mean((1,3))
    direct=((p@(b+delta)-target).square()-e.square()).reshape(3,7,2,64).mean((1,3))
    error=float((label-direct).abs().max());assert error<1e-12
    m=load_solution(ROOT/'solution/solution_v324.py')
    exec(compile((ROOT/'tools/attention_risk_rank_core.py').read_text(),'risk-test','exec'),m.__dict__)
    checks=1
    for d in (64,128):
        for t in (1,4,17,129):
            kh=2;v=torch.randn(t,kh*d,generator=rng)*.3
            if t==1:v.zero_()
            parent=m._dyn_table(v,None,has_scale=False)
            before=copy.deepcopy((v,parent));global_rng=torch.random.get_rng_state().clone()
            for context in (False,True):
                f=m._rk_features(v,m._deq_params(parent),kh,d,context)
                assert bool(torch.isfinite(f).all());checks+=1
                st=dict(scale=torch.ones(kh,f.shape[-1]),beta=torch.zeros(kh,f.shape[-1]),context=context)
                out,choice=m._rk_choose(v,parent,st,kh,d)
                assert tree_equal(out,parent) and not bool(choice.any());checks+=1
                st['beta']=torch.randn(kh,f.shape[-1],generator=rng)
                st['active']=torch.tensor([True,False])
                out,choice=m._rk_choose(v,parent,st,kh,d)
                assert not validate_hif4_params(out,v.shape,'risk')
                assert all(torch.equal(x[:,d//64:],parent[k][:,d//64:]) for k,x in out.items());checks+=2
            assert tree_equal(before,(v,parent)) and torch.equal(global_rng,torch.random.get_rng_state());checks+=1
    print('passed',checks,'exact_label_error',error,flush=True)


if __name__=='__main__':main()
