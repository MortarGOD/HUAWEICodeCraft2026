"""Independent coordinate interventions; only calibration fits new Grams."""
import torch


def refit(m,samples,states,qh,kh,d):
    hk,hq=m._gate_fit(samples,states,qh,kh,d)
    out=dict(states)
    for role,h in (('q',hk),('k',hq)):
        st=dict(states[role+'_state'])
        if st.get('gq'):
            u=m._upper_cholesky_inv(h)
            if u is None:raise ValueError('Cannot refit candidate inverse Gram')
            st['u']=u.contiguous()
        if 'H' in st:st['H']=h.bfloat16().contiguous()
        out[role+'_state']=st
    return out


def variants(m,samples,parent,qh,kh,d):
    answer=dict(parent=parent,refit=refit(m,samples,parent,qh,kh,d))
    if 'qs' in parent['q_state']:
        exact=dict(parent,k_state=dict(parent['k_state'],ks=parent['q_state']['qs'].float().reciprocal().contiguous()))
        answer['reciprocal_only']=exact
        answer['reciprocal_refit']=refit(m,samples,exact,qh,kh,d)
    for label,drop_scale,drop_dense in (('no_scale',True,False),('no_dense',False,True),('no_both',True,True)):
        st=dict(parent)
        for role in 'qk':
            field=dict(parent[role+'_state'])
            if drop_scale:field.pop('qs' if role=='q' else 'ks',None)
            if drop_dense:field.pop('joint_transform',None)
            st[role+'_state']=field
        answer[label]=refit(m,samples,st,qh,kh,d)
    return answer


def geometry(m,states,kh,d):
    transforms=[]
    for role in 'qk':
        st=states[role+'_state']
        if 'joint_transform' in st:t=st['joint_transform'].double()
        else:
            t=m._make_R(d,int(st.get('rseed') or 0)).double() if st.get('rot') else torch.eye(d,dtype=torch.float64)
            t=t[None].expand(kh,d,d)
        scale=st.get('qs' if role=='q' else 'ks')
        if scale is not None:t=scale.double()[:,:,None]*t
        transforms.append(t)
    error=transforms[0]@transforms[1].transpose(1,2)-torch.eye(d,dtype=torch.float64)[None]
    return dict(max_abs_identity_error=float(error.abs().max()),relative_frobenius_error=float(error.square().sum((1,2)).sqrt().mean()/d**.5))
