"""Continuous dual-frame invariants, not a generalization test."""
import torch
from probe_v319_value import ROOT,load_solution
from probe_v326_regroup import install,nonorthogonal_states,metric_states


def main():
    torch.set_num_threads(4)
    m=load_solution(ROOT/'solution/solution_v324.py');install(m)
    generator=torch.Generator().manual_seed(326)
    checks=0
    for d in (64,128):
        kh,qh=2,4
        a=torch.randn(kh,d,d,generator=generator)*.03
        r=torch.eye(d)+a
        states={}
        for role in 'qk':
            matrix=torch.randn(kh,d,d,generator=generator)*.03
            h=matrix@matrix.transpose(1,2)+torch.eye(d)
            u=torch.linalg.cholesky(torch.linalg.inv(h),upper=True)
            states[role+'_state']=dict(kvh=kh,u=u,H=h,joint_transform=torch.eye(d).repeat(kh,1,1))
        proposed,residual=nonorthogonal_states(m,states,r,qh,kh,d)
        assert residual<2e-6;checks+=1
        q=torch.randn(13,qh*d,generator=generator)
        k=torch.randn(17,kh*d,generator=generator)
        def logits(st):
            x=m._kt_input(q,st['q_state'],qh,d,'q').reshape(-1,qh,d).transpose(0,1)
            y=m._kt_input(k,st['k_state'],kh,d,'k').reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
            return x@y.transpose(1,2)
        assert torch.allclose(logits(states),logits(proposed),atol=3e-5,rtol=2e-5);checks+=1
        for role in 'qk':
            st=proposed[role+'_state'];inverse=st['u'].transpose(1,2)@st['u']
            assert torch.allclose(inverse@st['H'],torch.eye(d).expand_as(inverse),atol=3e-5,rtol=3e-5)
            checks+=1
        for name in ('q_flat','both_flat','q_sharp','both_shrink'):
            altered=metric_states(m,states,name,qh,kh,d)
            assert torch.equal(logits(states),logits(altered));checks+=1
            for role in 'qk':
                st=altered[role+'_state'];inverse=st['u'].transpose(1,2)@st['u']
                assert torch.allclose(inverse@st['H'],torch.eye(d).expand_as(inverse),atol=3e-5,rtol=3e-5)
                checks+=1
    print('passed',checks,flush=True)


if __name__=='__main__':main()
