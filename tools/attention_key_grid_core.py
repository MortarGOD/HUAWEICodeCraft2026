"""Key gauge + native scale-tree updates under the deployed channel metric."""
KG_DEBUG={}
_KG_MODE=1
_kg_parent_calibration=hif4_calibration_attention
_kg_parent_k=hif4_dynamic_quantize_k


def _kg_grid(x,st,kh,d,mode,total_rows=None):
    p=_dyn_table(x,None,has_scale=False)
    y=x.reshape(-1,kh,d)
    u=st['u'].float().repeat_interleave(kh//st['u'].shape[0],0)
    if mode>=3:
        initial=_gptq_quantize_batched(y.transpose(0,1).contiguous(),_params_unit_flat(p).reshape_as(y).transpose(0,1).contiguous(),u)
        p=_values_to_params(initial.transpose(0,1).reshape(x.shape).contiguous(),p)
    current=_deq_params(p).reshape_as(y)
    h=st['kg_h'].float()
    for block in range(d//64):
        a=block*64;b=a+64
        err=current-y
        gradient=torch.einsum('thd,hde->the',err,h)[:,:,a:b]
        old=current[:,:,a:b].clone()
        subh=h[:,a:b,a:b]
        proposal=y[:,:,a:b]
        if mode in (2,4):
            hh=subh.clone()
            hh.diagonal(dim1=-2,dim2=-1).add_(.01*subh.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None])
            proposal=old-torch.linalg.solve(hh,gradient.transpose(0,1).transpose(1,2)).transpose(1,2).transpose(0,1)
        cols=torch.arange(kh)*(d//64)+block
        selected={name:value[:,cols].clone() for name,value in p.items()}
        best=torch.zeros_like(gradient[:,:,0])
        for sf in _vm_neighbor_scales(selected['scale_factor']):
            trial=_vm_tree_at_sf(proposal.reshape(x.shape[0],-1),sf)
            value=_deq_params(trial).reshape(x.shape[0],kh,64)
            delta=value-old
            cost=2*(delta*gradient).sum(-1)+torch.einsum('thc,hce,the->th',delta,subh,delta)
            take=cost<best
            best=torch.minimum(best,cost)
            for name,tensor in selected.items():
                selected[name]=torch.where(take.reshape(*take.shape,*([1]*(tensor.ndim-2))),trial[name],tensor)
        current[:,:,a:b]=_deq_params(selected).reshape(x.shape[0],kh,64)
        for name in p:p[name][:,cols]=selected[name]
    unit=_params_unit_flat(p)
    value=(current.reshape(x.shape).contiguous() if mode>=3 else
        _gptq_quantize_batched(x.reshape(-1,kh,d).transpose(0,1).contiguous(),unit.reshape(-1,kh,d).transpose(0,1).contiguous(),u).transpose(0,1).reshape(x.shape).contiguous())
    rf=int(st.get('rf') or 0);t=x.shape[0] if total_rows is None else total_rows
    if rf and t<=ATTN_REFINE_MAX_T and 'H' in st:
        rounds=rf*(32 if t<=128 else (4 if t>1024 else _REF_ATTN_ROUNDS))
        value=_refine_attn_heads(x,value,unit,st['H'],kh,rf,total_rounds=rounds)
    return _values_to_params(value,p)


def _kg_quant(raw,st,kh,d,mode):
    if mode>=5:return _kg_center_search(raw,st,kh,d,mode)
    y=_kt_input(raw,st,kh,d,'k').reshape(-1,kh,d)
    key=raw.reshape(-1,kh,d).transpose(0,1)
    prob=torch.softmax(st['kt_probe'].float()@key.transpose(1,2),-1)
    weight=(.9*prob.mean(1)+.1/raw.shape[0]).transpose(0,1)[:,:,None]
    center=(y*weight).sum(0,keepdim=True)
    for _ in range(8):
        z=y-center
        error=_deq_params(_dyn_table(z.reshape(raw.shape),None,has_scale=False)).reshape(y.shape)-z
        center=center-(error*weight).sum(0,keepdim=True)
    return _kg_grid((y-center).reshape(raw.shape).contiguous(),st,kh,d,mode)


def _kg_center_search(raw,st,kh,d,mode):
    y=_kt_input(raw,st,kh,d,'k').reshape(-1,kh,d)
    key=raw.reshape(-1,kh,d).transpose(0,1)
    probability=torch.softmax(st['kt_probe'].float()@key.transpose(1,2),-1)
    weight=(.9*probability.mean(1)+.1/raw.shape[0]).transpose(0,1)[:,:,None]
    center=(y*weight).sum(0,keepdim=True)
    plain={k:v for k,v in st.items() if k not in ('joint_transform','ks','rot','hc_matrix')}
    for _ in range(8):
        z=y-center
        error=_deq_params(_dyn_table(z.reshape(raw.shape),None,has_scale=False)).reshape(y.shape)-z
        center=center-(error*weight).sum(0,keepdim=True)
    best=None;best_loss=None
    for step in range(3):
        z=y-center
        p=_kt_fixed_grid(z.reshape(raw.shape).contiguous(),plain,kh,d,'k')
        decoded=_deq_params(p).reshape(y.shape)
        logits=st['kg_query'].float()@decoded.transpose(0,1).transpose(1,2)
        loss=(torch.logsumexp(logits,-1)-(probability*logits).sum(-1)).mean(-1)
        if best is None:best=p;best_loss=loss
        else:
            take=loss<best_loss;best_loss=torch.minimum(best_loss,loss)
            mask=take.repeat_interleave(d//64)
            best={name:torch.where(mask.reshape(1,-1,*([1]*(v.ndim-2))),p[name],v) for name,v in best.items()}
        error=decoded-z
        center=center-(.5 if mode==5 else 1.)*(error*weight).sum(0,keepdim=True)
    return best


def _kg_guard(samples,parent,qh,kh,d):
    KG_DEBUG.clear();KG_DEBUG.update(accepted=0,error=False)
    st=parent['k_state'];hold=samples[-1];t=hold['k'][0].shape[0]
    if len(samples)<3 or d%64 or not 4<=t<=4096 or not st.get('gq') or 'u' not in st:return parent
    proposed=dict(st)
    if 'kt_probe' not in proposed:
        probes=[]
        for sample in samples[:-1][-2:]:
            count=sample['q'][0].shape[0];ixq=torch.linspace(0,count-1,min(count,16)).long()
            qr=dequantize_nvfp4(sample['q'][0][ixq],sample['q'][1][ixq]).float().reshape(-1,kh,qh//kh,d)
            probes.append(qr.permute(1,0,2,3).reshape(kh,-1,d))
        proposed['kt_probe']=torch.cat(probes,1)/(d**.5)
    if _KG_MODE>=5:
        probes=[];queries=[]
        for sample in samples[:-1][-2:]:
            count=sample['q'][0].shape[0];ixq=torch.linspace(0,count-1,min(count,16)).long()
            qr=dequantize_nvfp4(sample['q'][0][ixq],sample['q'][1][ixq]).float().reshape(-1,kh,qh//kh,d)
            probes.append(qr.permute(1,0,2,3).reshape(kh,-1,d))
            quant=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,ixq).reshape(-1,kh,qh//kh,d)
            queries.append(quant.permute(1,0,2,3).reshape(kh,-1,d))
        proposed['kt_probe']=torch.cat(probes,1)/(d**.5)
        proposed['kg_query']=torch.cat(queries,1)/(d**.5)
    inv=torch.linalg.inv(st['u'].float())
    h=inv@inv.transpose(-1,-2)
    proposed['kg_h']=h/h.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None].clamp_min(1e-30)
    ix=torch.linspace(0,t-1,min(t,32)).long()
    raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
    raw.extend(dequantize_nvfp4(*hold[r]).float() for r in 'kv')
    target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(_kg_parent_k(*hold['k'],kh,d,st))
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
    def cost(key):
        e=(_joint_output(q,key,v,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    trial=_deq_params(_kg_quant(raw[1],proposed,kh,d,_KG_MODE))
    old=cost(k);new=cost(trial)
    take=(new<old*.995).all(0)
    KG_DEBUG.update(accepted=int(take.sum()),ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):return parent
    return dict(parent,k_state=dict(proposed,kg_mode=_KG_MODE,kg_active=take,kg_fallback=st))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_kg_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _kg_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        KG_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    if not isinstance(k_state,dict) or 'kg_mode' not in k_state or not 4<=k_quant.shape[0]<=4096:
        fallback=k_state.get('kg_fallback',k_state) if isinstance(k_state,dict) else k_state
        return _kg_parent_k(k_quant,k_scale,kv_num_heads,head_dim,fallback)
    raw=dequantize_nvfp4(k_quant,k_scale).float()
    new=_kg_quant(raw,k_state,kv_num_heads,head_dim,k_state['kg_mode'])
    if bool(k_state['kg_active'].all()):return new
    old=_kg_parent_k(k_quant,k_scale,kv_num_heads,head_dim,k_state['kg_fallback'])
    mask=k_state['kg_active'].repeat_interleave(head_dim//64)
    return {name:torch.where(mask.reshape(1,-1,*([1]*(a.ndim-2))),a,old[name]) for name,a in new.items()}
