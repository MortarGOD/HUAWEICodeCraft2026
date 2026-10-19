"""Fixed-width affine QK embedding with a bounded rank sacrifice.

Low-rank calibration query means occupy homogeneous coordinates. The remaining
coordinates retain directions with high query/key bilinear energy. All tested
quantizers still see only their own current input and frozen calibration state.
"""
HC_DEBUG={}
_HC_RANKS=(0,1,4)
_hc_parent_calibration=hif4_calibration_attention
_hc_parent_q=hif4_dynamic_quantize_q
_hc_parent_k=hif4_dynamic_quantize_k


def _hc_psd_root(h):
    e,u=torch.linalg.eigh((h+h.transpose(-1,-2))*.5)
    e=e.clamp_min(e.amax(-1).clamp_min(1e-16)[:,None]*.001)
    return (u*e.sqrt()[:,None])@u.transpose(-1,-2),(u*e.rsqrt()[:,None])@u.transpose(-1,-2)


def _hc_inverse_factor(h):
    h=(h+h.transpose(-1,-2))*.5
    ridge=h.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-20)*GPTQ_DAMP
    eye=torch.eye(h.shape[-1])
    l=torch.linalg.cholesky(h+ridge[:,None,None]*eye)
    invl=torch.linalg.solve_triangular(l,eye.expand_as(l),upper=False)
    hi=invl.transpose(-1,-2)@invl
    return torch.linalg.cholesky((hi+hi.transpose(-1,-2))*.5,upper=True).contiguous()


def _hc_fit(samples,states,qh,kh,d,rank):
    fraction=rank/100. if rank>=10 else None
    if fraction is not None:rank=0
    qs=[];ks=[]
    for smp in samples[:-1]:
        rows=smp['k'][0].shape[0]
        if rows<4:continue
        ix=torch.linspace(0,rows-1,min(rows,256)).long()
        q=dequantize_nvfp4(*smp['q']).float().reshape(rows,qh,d)
        k=dequantize_nvfp4(*smp['k']).float().reshape(rows,kh,d)
        qs.append(q[ix]);ks.append((k-k.mean(0,keepdim=True))[ix])
    if len(qs)<2:return None
    q=torch.cat(qs);k=torch.cat(ks);n=q.shape[0];rep=qh//kh
    qb=q.transpose(0,1);kb=k.transpose(0,1)
    ck=kb.transpose(1,2)@kb/n
    root,invroot=_hc_psd_root(ck)
    rank=min(rank,rep,d-1)
    mean=q.mean(0).reshape(kh,rep,d)
    if rank:
        left,singular,right=torch.linalg.svd(mean@root,full_matrices=False)
        coefficient=left[:,:,:rank]*singular[:,:rank,None].transpose(1,2)
        basis=right[:,:rank]@invroot
        center=(coefficient@basis).reshape(qh,d)
    else:
        center=torch.zeros(qh,d)
    residual=qb-center[:,None]
    cq=(residual.transpose(1,2)@residual/n).reshape(kh,rep,d,d).mean(1)
    energy=root@cq@root
    eigen,axes=torch.linalg.eigh((energy+energy.transpose(-1,-2))*.5)
    keep=axes[:,:,rank:]
    power=eigen[:,rank:].clamp_min(eigen.amax(-1).clamp_min(1e-16)[:,None]*.001).pow(.25)
    tq=torch.zeros(kh,d,d);tk=torch.zeros_like(tq);bias=torch.zeros(qh,d)
    tq[:,:,:d-rank]=(root@keep)/power[:,None]
    tk[:,:,:d-rank]=(invroot@keep)*power[:,None]
    bias[:,:d-rank]=-(center[:,None,:]@tq[:,:,:d-rank].repeat_interleave(rep,0)).squeeze(1)
    if rank:
        a=coefficient.square().mean(1).sqrt().clamp_min(1e-12)
        b=(basis@ck@basis.transpose(1,2)).diagonal(dim1=-2,dim2=-1).clamp_min(1e-24).sqrt()
        scale=(b/a).sqrt().clamp(.01,100.)
        tk[:,:,d-rank:]=basis.transpose(1,2)/scale[:,None]
        bias[:,d-rank:]=(coefficient*scale[:,None]).reshape(qh,rank)
    if fraction is not None:
        retained=max(1,min(d,int(d*fraction)))
        selected=axes[:,:,-retained:]
        power=eigen[:,-retained:].clamp_min(eigen.amax(-1).clamp_min(1e-16)[:,None]*.001).pow(.25)
        generator=torch.Generator().manual_seed(321+d)
        orthogonal,_=torch.linalg.qr(torch.randn(d,d,generator=generator))
        frame=orthogonal[:retained]
        tq=((root@selected)/power[:,None])@frame
        tk=((invroot@selected)*power[:,None])@frame
    rot=_make_R(d,0)
    if rot is not None:tq,tk,bias=tq@rot,tk@rot,bias@rot
    qnew=qb@tq.repeat_interleave(rep,0)+bias[:,None]
    knew=kb@tk
    hq=(qnew.transpose(1,2)@qnew/n).reshape(kh,rep,d,d).sum(1)
    hk=knew.transpose(1,2)@knew/n
    result=dict(states)
    for role,mat,gram in (('q',tq,hk),('k',tk,hq)):
        old=states[role+'_state']
        st=dict(old,hc_matrix=mat.contiguous(),hc_active=torch.ones(kh,dtype=torch.bool),
                hc_bias=bias.contiguous() if role=='q' else torch.zeros(kh,d),
                H=gram.to(torch.bfloat16).contiguous(),u=_hc_inverse_factor(gram),kvh=kh)
        result[role+'_state']=st
    return result


def _hc_parent_input(raw,state,nh,d,role):
    x=raw
    sv=state.get('qs' if role=='q' else 'ks');kh=state['kvh']
    if isinstance(sv,torch.Tensor):
        if role=='q':x=_qks_apply_q(x,sv.float(),nh,kh,d,inv=False)
        else:x=(x.reshape(-1,nh,d)*sv.float()[None]).reshape(x.shape)
    if 'joint_transform' in state:
        mat=state['joint_transform'].float().repeat_interleave(nh//kh,0)
        x=(x.reshape(-1,nh,d).transpose(0,1)@mat).transpose(0,1).reshape(x.shape).contiguous()
    elif state.get('rot'):
        rot=_make_R(d,int(state.get('rseed') or 0))
        if rot is not None:x=(x.reshape(-1,nh,d)@rot).reshape(x.shape).contiguous()
    if role=='q' and 'temperature_gain' in state:
        x=(x.reshape(-1,nh,d)*state['temperature_gain'].float()[None,:,None]).reshape(x.shape).contiguous()
    return x


def _hc_encode(raw,state,nh,d,role):
    if 'hc_matrix' not in state:return _hc_parent_input(raw,state,nh,d,role)
    kh=state['kvh'];active=state['hc_active'].repeat_interleave(nh//kh)
    rb=raw.reshape(-1,nh,d).transpose(0,1)[active]
    if role=='k':rb=rb-rb.mean(1,keepdim=True)
    mat=state['hc_matrix'].float().repeat_interleave(nh//kh,0)[active]
    result=rb@mat+state['hc_bias'].float()[active,None]
    if bool(active.all()):return result.transpose(0,1).reshape(raw.shape).contiguous()
    x=_hc_parent_input(raw,state,nh,d,role)
    xb=x.reshape(-1,nh,d).transpose(0,1).clone();xb[active]=result
    return xb.transpose(0,1).reshape(raw.shape).contiguous()


def _hc_quantize_raw(raw,state,nh,d,role,total_rows=None):
    rows=raw.shape[0] if total_rows is None else total_rows
    if rows==1:return _dyn_table(raw,None,has_scale=False)
    x=_hc_encode(raw,state,nh,d,role)
    p=_dyn_table(x,None,has_scale=False);unit=_params_unit_flat(p)
    u=state['u'].float().repeat_interleave(nh//state['u'].shape[0],0)
    value=_gptq_quantize_batched(x.reshape(-1,nh,d).transpose(0,1).contiguous(),
        unit.reshape(-1,nh,d).transpose(0,1).contiguous(),u).transpose(0,1).reshape(x.shape).contiguous()
    rf=int(state.get('rf') or 0)
    if rf and rows<=ATTN_REFINE_MAX_T and 'H' in state:
        rounds=rf*(32 if rows<=128 else (4 if rows>1024 else _REF_ATTN_ROUNDS))
        value=_refine_attn_heads(x,value,unit,state['H'],nh,rf,total_rounds=rounds)
    return _values_to_params(value,p)


def _hc_guard(samples,parent,qh,kh,d):
    HC_DEBUG.clear();HC_DEBUG.update(accepted=0,error=False,ranks=[])
    if len(samples)<3 or d%64 or any(not parent[r+'_state'].get('gq') or 'H' not in parent[r+'_state'] for r in 'qk'):
        return parent
    hold=samples[-1];rows=hold['k'][0].shape[0]
    if not 4<=rows<=4096:return parent
    raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    ix=torch.linspace(0,rows-1,min(rows,64)).long()
    target=_joint_output(*raw,qh,kh,d,ix)
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
    q0=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k0=_deq_params(_hc_parent_k(*hold['k'],kh,d,parent['k_state']))
    def cost(q,k):
        e=(_joint_output(q,k,v,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    original=cost(q0,k0);best=original.clone();answer=dict(parent);chosen=torch.full((kh,),-1,dtype=torch.int32)
    for rank in _HC_RANKS:
        trial=_hc_fit(samples,parent,qh,kh,d,rank)
        if trial is None:continue
        q=_deq_params(_hc_quantize_raw(raw[0][ix],trial['q_state'],qh,d,'q',rows))
        k=_deq_params(_hc_quantize_raw(raw[1],trial['k_state'],kh,d,'k',rows))
        error=cost(q,k)
        take=(error<original*.995).all(0)&(error.mean(0)<best.mean(0))
        HC_DEBUG['ranks'].append(dict(rank=rank,ratios=(error.mean(0)/original.mean(0).clamp_min(1e-30)).tolist(),accepted=int(take.sum())))
        if not bool(take.any()):continue
        best=torch.where(take[None],error,best);chosen=torch.where(take,rank,chosen)
        for role in 'qk':
            old=answer[role+'_state'];ts=trial[role+'_state'];nh=qh if role=='q' else kh
            new=dict(old);new['hc_active']=old.get('hc_active',torch.zeros(kh,dtype=torch.bool))|take
            for key in ('u','H','hc_matrix'):
                base=old.get(key,torch.zeros_like(ts[key]))
                new[key]=torch.where(take[:,None,None],ts[key],base).contiguous()
            mask=take.repeat_interleave(nh//kh)
            new['hc_bias']=torch.where(mask[:,None],ts['hc_bias'],old.get('hc_bias',torch.zeros_like(ts['hc_bias']))).contiguous()
            answer[role+'_state']=new
    HC_DEBUG.update(accepted=int((chosen>=0).sum()),chosen=chosen.tolist())
    return answer


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_hc_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _hc_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        HC_DEBUG.update(error=True,error_type=type(exc).__name__)
        return parent


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    if not isinstance(q_state,dict) or 'hc_matrix' not in q_state:
        return _hc_parent_q(q_quant,q_scale,q_num_heads,head_dim,q_state)
    return _hc_quantize_raw(dequantize_nvfp4(q_quant,q_scale).float(),q_state,q_num_heads,head_dim,'q')


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    if not isinstance(k_state,dict) or 'hc_matrix' not in k_state:
        return _hc_parent_k(k_quant,k_scale,kv_num_heads,head_dim,k_state)
    return _hc_quantize_raw(dequantize_nvfp4(k_quant,k_scale).float(),k_state,kv_num_heads,head_dim,'k')
