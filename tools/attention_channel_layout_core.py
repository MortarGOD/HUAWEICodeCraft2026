"""Paired channel layout: change legal micro-scale groups, not just GPTQ order."""
_SP_MODE=1


def _sb_fit(samples,parent,qh,kh,d):
    qenergy,kenergy,count=_sb_statistics(samples,parent,qh,kh,d)
    if not count:return torch.arange(d)[None].expand(kh,d)
    q=qenergy.clamp_min(1e-12).log();k=kenergy.clamp_min(1e-12).log()
    score=k if _SP_MODE==1 else (q-k if _SP_MODE==2 else q+k)
    return score.argsort(dim=-1,stable=True)


def _sb_state(st,order,role):
    kh,d=order.shape;matrix=st.get('joint_transform')
    if matrix is None:
        matrix=_make_R(d,int(st.get('rseed') or 0)) if st.get('rot') else None
        if matrix is None:matrix=torch.eye(d)
        matrix=matrix[None].expand(kh,d,d)
    result=dict(st,joint_transform=matrix.float().gather(2,order[:,None,:].expand(kh,d,d)).contiguous())
    if 'u' in st:
        u=st['u'].double();perm=order.repeat_interleave(u.shape[0]//kh,0)
        inv=u.transpose(1,2)@u
        inv=inv.gather(1,perm[:,:,None].expand_as(inv)).gather(2,perm[:,None,:].expand_as(inv))
        result['u']=torch.linalg.cholesky(inv,upper=True).float().contiguous()
    if 'H' in st:
        h=st['H'];perm=order.repeat_interleave(h.shape[0]//kh,0)
        result['H']=h.gather(1,perm[:,:,None].expand_as(h)).gather(2,perm[:,None,:].expand_as(h)).contiguous()
    if 'ts_fallback' in st:result['ts_fallback']=_sb_state(st['ts_fallback'],order,role)
    return result
