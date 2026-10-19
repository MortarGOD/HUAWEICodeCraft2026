"""Input-conditioned Softmax KL coordinate descent on frozen key prototypes."""
NQ_DEBUG={}
_NQ_ROUNDS=8
_NQ_OUTPUT=False
_NQ_ALIGNED=False
_nq_parent_calibration=hif4_calibration_attention
_nq_parent_q=hif4_dynamic_quantize_q


def _nq_input(raw,st,nh,kh,d,role):
    z=raw;sv=st.get('qs' if role=='q' else 'ks')
    if isinstance(sv,torch.Tensor):
        if role=='q':z=_qks_apply_q(z,sv.float(),nh,kh,d,inv=False)
        else:z=(z.reshape(-1,nh,d)*sv.float()[None]).reshape(raw.shape)
    if 'joint_transform' in st:
        z=(z.reshape(-1,nh,d).transpose(0,1)@st['joint_transform'].float().repeat_interleave(nh//kh,0)).transpose(0,1).reshape(raw.shape).contiguous()
    elif st.get('rot'):
        r=_make_R(d,int(st.get('rseed') or 0))
        if r is not None:z=(z.reshape(-1,nh,d)@r).reshape(raw.shape).contiguous()
    if role=='q' and 'temperature_gain' in st:z=(z.reshape(-1,nh,d)*st['temperature_gain'].float()[None,:,None]).reshape(raw.shape).contiguous()
    return z


def _nq_fit(samples,parent,qh,kh,d):
    original=[];quantized=[];values=[];qvalues=[]
    for s in samples[:-1][-2:]:
        rows=s['k'][0].shape[0]
        ix=torch.linspace(0,rows-1,min(rows,32)).long()
        original.append(dequantize_nvfp4(*s['k']).float()[ix].reshape(-1,kh,d))
        k=_deq_params(hif4_dynamic_quantize_k(*s['k'],kh,d,parent['k_state']))
        quantized.append(k[ix].reshape(-1,kh,d))
        if _NQ_ALIGNED:
            transformed=_nq_input(dequantize_nvfp4(*s['k']).float(),parent['k_state'],kh,kh,d,'k')
            original[-1]=transformed[ix].reshape(-1,kh,d)
            quantized[-1]=original[-1]
        if _NQ_OUTPUT:
            values.append(dequantize_nvfp4(*s['v']).float()[ix].reshape(-1,kh,d))
            v=_deq_params(hif4_dynamic_quantize_v(*s['v'],kh,d,parent['v_state']))
            qvalues.append(v[ix].reshape(-1,kh,d))
    raw=torch.cat(original).transpose(0,1).contiguous()/(d**.5)
    key=torch.cat(quantized).transpose(0,1).contiguous()/(d**.5)
    bank=dict(raw=raw,key=key,active=torch.ones(qh,dtype=torch.bool),rounds=_NQ_ROUNDS,aligned=_NQ_ALIGNED)
    if _NQ_OUTPUT:
        vr=torch.cat(values).transpose(0,1).contiguous();vq=torch.cat(qvalues).transpose(0,1).contiguous()
        center=vr.mean(1,keepdim=True);vr=vr-center;vq=vq-center
        _,axes=torch.linalg.eigh(vr.transpose(1,2)@vr)
        bank.update(value=vq@axes[:,:,-min(16,d):],raw_value=vr@axes[:,:,-min(16,d):])
    return bank


def _nq_refine(raw,p,bank,qh,d):
    rows=raw.shape[0]
    if rows<=1 or rows>4096:return p
    active=bank['active'];ids=torch.arange(qh)[active]
    if ids.numel()==0:return p
    k=bank['key'].float().repeat_interleave(qh//bank['key'].shape[0],0)[active]
    kr=bank['raw'].float().repeat_interleave(qh//bank['raw'].shape[0],0)[active]
    output='value' in bank
    if output:
        vv=bank['value'].float().repeat_interleave(qh//bank['value'].shape[0],0)[active]
        vr=bank['raw_value'].float().repeat_interleave(qh//bank['raw_value'].shape[0],0)[active]
    value=_deq_params(p).reshape(rows,qh,d).transpose(0,1).contiguous()
    unit=_params_unit_flat(p).reshape(rows,qh,d).transpose(0,1).contiguous()/4.
    x=raw.reshape(rows,qh,d).transpose(0,1)
    for start in range(0,rows,256):
        end=min(rows,start+256);v=value[active,start:end].clone();step=unit[active,start:end]
        level=torch.round(v/step)
        target=torch.softmax(x[active,start:end]@kr.transpose(1,2),-1)
        logits=v@k.transpose(1,2)
        cost=torch.logsumexp(logits,-1)-(target*logits).sum(-1)
        if output:
            goal=target@vr
            cost=(torch.softmax(logits,-1)@vv-goal).square().sum(-1)
        for _ in range(bank['rounds']):
            probability=torch.softmax(logits,-1)
            mean=probability@k
            gradient=(probability-target)@k
            curvature=(probability@k.square()-mean.square()).clamp_min(0)
            direction=torch.where(gradient<0,1.,-1.)
            delta=direction*step
            estimated=gradient*delta+.5*curvature*delta.square()
            if output:
                residual=probability@vv-goal
                weight=residual@vv.transpose(1,2)
                gradient=2*(probability*(weight-(probability*weight).sum(-1,keepdim=True)))@k
                direction=torch.where(gradient<0,1.,-1.)
                delta=direction*step
                estimated=gradient*delta
            estimated.masked_fill_((level+direction).abs()>7,float('inf'))
            coord=estimated.argmin(-1,keepdim=True)
            change=delta.gather(-1,coord)
            columns=k.transpose(1,2).gather(1,coord.expand(-1,-1,k.shape[1]))
            proposal=logits+change*columns
            newcost=torch.logsumexp(proposal,-1)-(target*proposal).sum(-1)
            if output:newcost=(torch.softmax(proposal,-1)@vv-goal).square().sum(-1)
            take=newcost<cost-1e-7
            if not bool(take.any()):break
            actual=change*take[:,:,None]
            v.scatter_add_(-1,coord,actual)
            level.scatter_add_(-1,coord,direction.gather(-1,coord)*take[:,:,None])
            logits=torch.where(take[:,:,None],proposal,logits)
            cost=torch.where(take,newcost,cost)
        value[ids[:,None],torch.arange(start,end)[None]]=v
    return _values_to_params(value.transpose(0,1).reshape(raw.shape).contiguous(),p)


def _nq_guard(samples,parent,qh,kh,d):
    NQ_DEBUG.clear();NQ_DEBUG.update(accepted=0,error=False)
    if len(samples)<3:return parent
    hold=samples[-1];rows=hold['k'][0].shape[0]
    if not 4<=rows<=4096:return parent
    bank=_nq_fit(samples,parent,qh,kh,d)
    raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    ix=torch.linspace(0,rows-1,min(rows,64)).long()
    target=_joint_output(*raw,qh,kh,d,ix)
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    # _joint_quant_queries returns decoded values; recover exactly its metadata
    # by quantizing the transformed original query with the same scale selection.
    x=raw[0][ix];st=parent['q_state'];z=x
    if isinstance(st.get('qs'),torch.Tensor):z=_qks_apply_q(z,st['qs'].float(),qh,kh,d,inv=False)
    if 'joint_transform' in st:
        z=(z.reshape(-1,qh,d).transpose(0,1)@st['joint_transform'].float().repeat_interleave(qh//kh,0)).transpose(0,1).reshape(x.shape).contiguous()
    elif st.get('rot'):
        r=_make_R(d,int(st.get('rseed') or 0))
        if r is not None:z=(z.reshape(-1,qh,d)@r).reshape(x.shape).contiguous()
    if 'temperature_gain' in st:z=(z.reshape(-1,qh,d)*st['temperature_gain'].float()[None,:,None]).reshape(x.shape).contiguous()
    params=_values_to_params(q,_dyn_table(z,None,has_scale=False))
    candidate=_deq_params(_nq_refine(z if bank['aligned'] else x,params,bank,qh,d))
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
    def cost(query):
        e=(_joint_output(query,k,v,qh,kh,d,ix,query_subset=True)-target).reshape(ix.numel(),qh,d).square().mean(-1)
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    old=cost(q);new=cost(candidate)
    active=(new<old*.995).all(0)
    NQ_DEBUG.update(accepted=int(active.sum()),ratios=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(active.any()):return parent
    bank['active']=active
    return dict(parent,q_state=dict(st,nq=bank))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_nq_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _nq_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        NQ_DEBUG.update(error=True,error_type=type(exc).__name__)
        return parent


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    p=_nq_parent_q(q_quant,q_scale,q_num_heads,head_dim,q_state)
    if not isinstance(q_state,dict) or 'nq' not in q_state:return p
    raw=dequantize_nvfp4(q_quant,q_scale).float()
    if q_state['nq']['aligned']:raw=_nq_input(raw,q_state,q_num_heads,q_state['kvh'],head_dim,'q')
    return _nq_refine(raw,p,q_state['nq'],q_num_heads,head_dim)
