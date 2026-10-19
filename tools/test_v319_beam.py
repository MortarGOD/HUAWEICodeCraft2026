import json
import torch
from probe_v319_beam import ROOT, configured
from self_check import validate_hif4_params, validate_frozen_state


def main():
    torch.set_num_threads(4)
    m=configured(); rng=torch.Generator().manual_seed(319)
    checks=[]
    for rows in (1,7,129):
        d=64;nh=2
        a=torch.randn(nh,d,d,generator=rng)
        h=a@a.transpose(-1,-2)/d+torch.eye(d)*.1
        st=m._beam_prepare(dict(gq=1,H=h,u=torch.linalg.cholesky(torch.linalg.inv(h),upper=True),kvh=nh),d)
        assert not validate_frozen_state(st,'beam')
        x=torch.randn(rows,nh*d,generator=rng)
        p=m._dyn_table(x,None,has_scale=False)
        before=m._deq_params(p)
        after=m._beam_refine(x,p,st,nh,d,max(2,rows))
        assert not validate_hif4_params(after,x.shape,'beam')
        perm=st['beam_perm'][:,None,:].expand(nh,rows,d)
        h=st['beam_h']
        def error(value):
            delta=(value-x).reshape(rows,nh,d).transpose(0,1).gather(-1,perm)
            return ((delta@h)*delta).sum(-1)
        assert bool((error(m._deq_params(after))<=error(before)+1e-5).all())
        assert torch.equal(before,m._deq_params(p))
        frozen=m._beam_refine(x,p,dict(st,beam_active=torch.zeros(nh,dtype=torch.bool)),nh,d,max(2,rows))
        assert all(torch.equal(p[k],frozen[k]) for k in p)
        checks.append(dict(rows=rows,reduction=float((error(before)-error(m._deq_params(after))).sum())))
    report=dict(passed=True,checks=checks)
    (ROOT/'docs/v319-beam-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':main()
