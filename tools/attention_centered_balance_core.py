"""Gauge-aware reciprocal Q/K balancing, with exact metric congruence."""
SB_DEBUG={}
_SB_ALPHA=.5
_sb_parent_calibration=hif4_calibration_attention
_sb_parent_q=hif4_dynamic_quantize_q
_sb_parent_k=hif4_dynamic_quantize_k


def _sb_state(st,scale,role):
    kh,d=scale.shape
    matrix=st.get('joint_transform')
    if matrix is None:
        matrix=_make_R(d,int(st.get('rseed') or 0)) if st.get('rot') else None
        if matrix is None:matrix=torch.eye(d)
        matrix=matrix[None].expand(kh,d,d)
    factor=scale.reciprocal() if role=='q' else scale
    result=dict(st,joint_transform=(matrix.float()*factor[:,None,:]).contiguous())
    if 'u' in st:
        f=factor.repeat_interleave(st['u'].shape[0]//kh,0)
        result['u']=(st['u'].float()*f[:,None,:]).contiguous()
    if 'H' in st:
        f=factor.repeat_interleave(st['H'].shape[0]//kh,0)
        result['H']=(st['H'].float()/f[:,:,None]/f[:,None,:]).to(st['H'].dtype).contiguous()
    if 'ts_fallback' in st:result['ts_fallback']=_sb_state(st['ts_fallback'],scale,role)
    return result


def _sb_statistics(samples,parent,qh,kh,d):
    qenergy=torch.zeros(kh,d);kenergy=torch.zeros(kh,d);count=0
    for sample in samples[:-1][-2:]:
        rows=sample['k'][0].shape[0]
        if rows<2:continue
        qr=dequantize_nvfp4(*sample['q']).float()
        kr=dequantize_nvfp4(*sample['k']).float()
        ix=torch.linspace(0,qr.shape[0]-1,min(qr.shape[0],64)).long()
        q=_kt_input(qr[ix],parent['q_state'],qh,d,'q').reshape(-1,kh,qh//kh,d)
        k=_kt_input(kr,parent['k_state'],kh,d,'k').reshape(-1,kh,d)
        probe=qr[ix].reshape(-1,kh,qh//kh,d).permute(1,0,2,3).reshape(kh,-1,d)/(d**.5)
        p=torch.softmax(probe@kr.reshape(-1,kh,d).transpose(0,1).transpose(1,2),-1)
        weight=(.9*p.mean(1)+.1/rows).transpose(0,1)[:,:,None]
        center=(k*weight).sum(0,keepdim=True)
        qenergy+=q.square().mean((0,2));kenergy+=((k-center).square()*weight).sum(0);count+=1
    return qenergy,kenergy,count


def _sb_fit(samples,parent,qh,kh,d):
    qenergy,kenergy,count=_sb_statistics(samples,parent,qh,kh,d)
    if not count:return torch.ones(kh,d)
    logratio=.5*(qenergy.clamp_min(1e-12).log()-kenergy.clamp_min(1e-12).log())
    logratio=logratio-logratio.mean(-1,keepdim=True)
    return (_SB_ALPHA*logratio).exp().clamp(.5,2.)


def _sb_guard(samples,parent,qh,kh,d):
    SB_DEBUG.clear();SB_DEBUG.update(accepted=0,error=False)
    if any(not parent[r+'_state'].get('gq') or 'u' not in parent[r+'_state'] for r in 'qk'):return parent
    hold=samples[-1];t=hold['k'][0].shape[0]
    if not 4<=t<=4096:return parent
    scale=_sb_fit(samples,parent,qh,kh,d)
    proposed={r+'_state':_sb_state(parent[r+'_state'],scale,r) for r in 'qk'}
    ix=torch.linspace(0,t-1,min(t,32)).long()
    raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
    raw.extend(dequantize_nvfp4(*hold[r]).float() for r in 'kv')
    goal=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
    def cost(states):
        q=_joint_quant_queries(hold,states['q_state'],qh,kh,d,ix)
        k=_deq_params(_sb_parent_k(*hold['k'],kh,d,states['k_state']))
        e=(_joint_output(q,k,v,qh,kh,d,ix,query_subset=True)-goal).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    old=cost(parent);new=cost(proposed);take=(new<old*.995).all(0)
    SB_DEBUG.update(accepted=int(take.sum()),ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist(),scale_min=float(scale.min()),scale_max=float(scale.max()))
    if not bool(take.any()):return parent
    result=dict(parent)
    for role in 'qk':
        result[role+'_state']=dict(sb_new=proposed[role+'_state'],sb_fallback=parent[role+'_state'],sb_active=take,sb_kvh=kh)
    return result


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_sb_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _sb_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        SB_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def _sb_dynamic(pair,nh,d,state,role):
    fn=_sb_parent_q if role=='q' else _sb_parent_k
    if not isinstance(state,dict) or 'sb_new' not in state:return fn(*pair,nh,d,state)
    if not 4<=pair[0].shape[0]<=4096:return fn(*pair,nh,d,state['sb_fallback'])
    new=fn(*pair,nh,d,state['sb_new'])
    if bool(state['sb_active'].all()):return new
    old=fn(*pair,nh,d,state['sb_fallback'])
    mask=state['sb_active'].repeat_interleave(nh//state['sb_kvh']).repeat_interleave(d//64)
    return {name:torch.where(mask.reshape(1,-1,*([1]*(x.ndim-2))),x,old[name]) for name,x in new.items()}


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    return _sb_dynamic((q_quant,q_scale),q_num_heads,head_dim,q_state,'q')


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    return _sb_dynamic((k_quant,k_scale),kv_num_heads,head_dim,k_state,'k')
