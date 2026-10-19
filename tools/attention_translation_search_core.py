"""Task-metric Key translation research layered on the frozen v321 parent."""
TS_DEBUG={}
_TS_VARIANT=1
_TS_MATCH_GUARD=False
_ts_parent_calibration=hif4_calibration_attention
_ts_parent_k=hif4_dynamic_quantize_k


def _ts_quant(raw,st,kh,d,variant):
    if variant in (10,11):return _ts_kl_refine(raw,st,kh,d,variant)
    y=_kt_input(raw,st,kh,d,'k').reshape(-1,kh,d)
    weight=torch.full((raw.shape[0],kh,1),1/raw.shape[0])
    if 'kt_probe' in st:
        key=raw.reshape(-1,kh,d).transpose(0,1)
        prob=torch.softmax(st['kt_probe'].float()@key.transpose(1,2),-1)
        weight=(.9*prob.mean(1)+.1/raw.shape[0]).transpose(0,1)[:,:,None]
        if 'kt_weighted' in st:
            weight=torch.where(st['kt_weighted'][None,:,None],weight,torch.full_like(weight,1/raw.shape[0]))
    center=y.mean(0,keepdim=True)
    plain={k:v for k,v in st.items() if k not in ('joint_transform','ks','rot','hc_matrix')}
    def encode(c):
        if 'kt_active' in st:c=c*st['kt_active'][None,:,None]
        return _kt_fixed_grid((y-c).reshape(raw.shape).contiguous(),plain,kh,d,'k')
    if variant in (1,5,6,8,9):
        if variant!=9:center=(y*weight).sum(0,keepdim=True)
        for _ in range(8 if variant in (1,9) else (24 if variant==5 else (3 if variant==8 else 0))):
            z=y-center
            e=_deq_params(_dyn_table(z.reshape(raw.shape),None,has_scale=False)).reshape(y.shape)-z
            center=center-(e*weight).sum(0,keepdim=True)
        return encode(center)
    if variant==2:
        for _ in range(3):
            z=y-center
            e=_deq_params(_dyn_table(z.reshape(raw.shape),None,has_scale=False)).reshape(y.shape)-z
            center=center-(e*weight).sum(0,keepdim=True)
        first=encode(center)
        e=_deq_params(first).reshape(y.shape)-(y-center)
        center=center-(e*weight).sum(0,keepdim=True)
        return encode(center)
    # Remove the Softmax-null constant part before a channel-Hessian cost.
    h=st.get('H')
    if h is None:h=torch.eye(d).expand(kh,d,d)
    h=h.float()
    best=torch.full((kh,),float('inf'));best_center=center.clone()
    starts=(center,(y*weight).sum(0,keepdim=True),
            (y.amin(0,keepdim=True)+y.amax(0,keepdim=True))*.5)
    for initial in starts:
        c=initial.clone()
        for _ in range(4):
            z=y-c
            e=_deq_params(_dyn_table(z.reshape(raw.shape),None,has_scale=False)).reshape(y.shape)-z
            mean=(e*weight).sum(0,keepdim=True);res=e-mean
            metric=((res.transpose(0,1)@h).transpose(0,1)*res*weight).sum((0,2))
            take=metric<best;best=torch.minimum(best,metric)
            best_center=torch.where(take[None,:,None],c,best_center)
            c=c-mean
    return encode(best_center)


def _ts_guard(samples,parent,qh,kh,d):
    TS_DEBUG.clear();TS_DEBUG.update(accepted=0,error=False)
    cached=parent.get('_ts_cache')
    parent={k:v for k,v in parent.items() if k!='_ts_cache'}
    st=parent['k_state'];hold=samples[-1];t=hold['k'][0].shape[0]
    if not 4<=t<=4096 or not st.get('gq') or 'u' not in st:return parent
    candidate_state=dict(st,kt_active=torch.ones(kh,dtype=torch.bool),ts_fallback=st)
    # A different initial gauge may succeed where the mean-origin v321
    # candidate failed. Evaluate it against the actual parent, not raw GPTQ.
    if 'kt_probe' not in candidate_state:
        probes=[]
        for sample in samples[:-1][-2:]:
            rows=sample['q'][0].shape[0];ixq=torch.linspace(0,rows-1,min(rows,16)).long()
            qr=dequantize_nvfp4(sample['q'][0][ixq],sample['q'][1][ixq]).float().reshape(-1,kh,qh//kh,d)
            probes.append(qr.permute(1,0,2,3).reshape(kh,-1,d))
        candidate_state['kt_probe']=torch.cat(probes,1)/(d**.5)
    candidate_state.pop('kt_weighted',None)
    if _TS_VARIANT in (10,11):
        probes=[]
        for sample in samples[:-1][-2:]:
            rows=sample['q'][0].shape[0];ixq=torch.linspace(0,rows-1,min(rows,16)).long()
            qr=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,ixq).reshape(-1,kh,qh//kh,d)
            probes.append(qr.permute(1,0,2,3).reshape(kh,-1,d))
        candidate_state['ts_query']=torch.cat(probes,1)/(d**.5)
    if cached is not None:
        ix,raw,target,q,k,v=cached
    else:
        guard_rows=64 if _TS_MATCH_GUARD and not int(st.get('rf') or 0) else 32
        ix=torch.linspace(0,t-1,min(t,guard_rows)).long()
        raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
        raw.extend(dequantize_nvfp4(*hold[r]).float() for r in 'kv')
        target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
        q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
        k=_deq_params(_ts_parent_k(*hold['k'],kh,d,st))
        v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
    def cost(key):
        e=(_joint_output(q,key,v,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    old=cost(k);new=cost(_deq_params(_ts_quant(raw[1],candidate_state,kh,d,_TS_VARIANT)))
    take=(new<old*.995).all(0)
    TS_DEBUG.update(accepted=int(take.sum()),ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):return parent
    return dict(parent,k_state=dict(candidate_state,ts_variant=_TS_VARIANT,ts_active=take,ts_fallback=st))


def _ts_kl_refine(raw,st,kh,d,variant):
    origin=st['ts_fallback']
    p=(_kt_quant(raw,origin,kh,d,origin['kt_mode']) if 'kt_mode' in origin else
       _kt_fixed_grid(raw,origin,kh,d,'k'))
    query=st['ts_query'].float();teacher=st['kt_probe'].float()
    key=raw.reshape(-1,kh,d).transpose(0,1)
    value=_deq_params(p).reshape(-1,kh,d).transpose(0,1).contiguous()
    unit=_params_unit_flat(p).reshape(-1,kh,d).transpose(0,1).contiguous()/4
    if variant==11:
        teacher=query;key=_kt_input(raw,origin,kh,d,'k').reshape(-1,kh,d).transpose(0,1)
    target=torch.softmax(teacher@key.transpose(1,2),-1)
    logits=query@value.transpose(1,2)
    def loss(x):return (torch.logsumexp(x,-1)-(target*x).sum(-1)).mean(-1)
    best=loss(logits)
    levels=torch.round(value/unit)
    for _ in range(8):
        prob=torch.softmax(logits,-1)
        grad=(prob-target).transpose(1,2)@query
        curvature=(prob*(1-prob)).transpose(1,2)@query.square()
        direction=torch.where(grad<0,1.,-1.);delta=direction*unit
        estimate=grad*delta+.5*curvature*delta.square()
        estimate.masked_fill_((levels+direction).abs()>7,float('inf'))
        coord=estimate.argmin(-1,keepdim=True);change=delta.gather(-1,coord)
        improve=estimate.gather(-1,coord)<0;change=change*improve
        columns=query.gather(2,coord.squeeze(-1)[:,None,:].expand(-1,query.shape[1],-1))
        proposal=logits+columns*change.transpose(1,2)
        new=loss(proposal);take=new<best-1e-7
        if not bool(take.any()):break
        value.scatter_add_(-1,coord,change*take[:,None,None])
        levels.scatter_add_(-1,coord,direction.gather(-1,coord)*improve*take[:,None,None])
        logits=torch.where(take[:,None,None],proposal,logits);best=torch.where(take,new,best)
    return _values_to_params(value.transpose(0,1).reshape(raw.shape).contiguous(),p)


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_ts_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:answer=_ts_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        TS_DEBUG.update(error=True,error_type=type(exc).__name__);answer=parent
    return {k:v for k,v in answer.items() if k!='_ts_cache'}


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    if not isinstance(k_state,dict) or 'ts_variant' not in k_state or not 4<=k_quant.shape[0]<=4096:
        fallback=k_state.get('ts_fallback',k_state) if isinstance(k_state,dict) else k_state
        return _ts_parent_k(k_quant,k_scale,kv_num_heads,head_dim,fallback)
    raw=dequantize_nvfp4(k_quant,k_scale).float()
    new=_ts_quant(raw,k_state,kv_num_heads,head_dim,k_state['ts_variant'])
    if bool(k_state['ts_active'].all()):return new
    old=_ts_parent_k(k_quant,k_scale,kv_num_heads,head_dim,k_state.get('ts_fallback',k_state))
    # Merge complete 64-channel blocks without modifying metadata or codes.
    mask=k_state['ts_active'].repeat_interleave(head_dim//64)
    return {name:torch.where(mask.reshape(1,-1,*([1]*(a.ndim-2))),a,old[name])
            for name,a in new.items()}
