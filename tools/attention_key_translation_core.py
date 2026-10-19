"""Softmax-invariant current-key translation, including Lloyd offset fitting."""
KT_DEBUG={}
_KT_MODE=1
_KT_MODES=None
_KT_BUDGET_ROUTE=False
_KT_GUARD_ROWS=64
_KT_MIN_GAIN=.005
_kt_parent_calibration=_hc_parent_calibration
_kt_parent_k=_hc_parent_k
hif4_dynamic_quantize_q=_hc_parent_q


def _kt_quant(raw,st,kh,d,mode):
    if raw.shape[0]<=1:return _dyn_table(raw,None,has_scale=False)
    y=_hc_parent_input(raw,st,kh,d,'k').reshape(-1,kh,d)
    if mode==3:center=(y.amin(0,keepdim=True)+y.amax(0,keepdim=True))*.5
    else:center=y.mean(0,keepdim=True)
    weight=None
    if mode==4:
        key=raw.reshape(-1,kh,d).transpose(0,1)
        probability=torch.softmax(st['kt_probe'].float()@key.transpose(1,2),-1)
        weight=(probability.mean(1)*.9+.1/raw.shape[0]).transpose(0,1)[:,:,None]
        if 'kt_weighted' in st:weight=torch.where(st['kt_weighted'][None,:,None],weight,torch.full_like(weight,1/raw.shape[0]))
    if mode in (2,4):
        for _ in range(3):
            z=y-center
            quant=_deq_params(_dyn_table(z.reshape(raw.shape),None,has_scale=False)).reshape(y.shape)
            error=quant-z
            center=center-(error.mean(0,keepdim=True) if weight is None else (error*weight).sum(0,keepdim=True))
    if 'kt_active' in st:center=center*st['kt_active'][None,:,None]
    plain={k:v for k,v in st.items() if k not in ('joint_transform','ks','rot','hc_matrix')}
    return _hc_quantize_raw((y-center).reshape(raw.shape).contiguous(),plain,kh,d,'k')


def _kt_guard(samples,parent,qh,kh,d):
    KT_DEBUG.clear();KT_DEBUG.update(accepted=0,error=False)
    if len(samples)<3 or d%64 or not parent['k_state'].get('gq') or 'u' not in parent['k_state']:return parent
    hold=samples[-1];rows=hold['k'][0].shape[0]
    if not 4<=rows<=4096:return parent
    refined=int(parent['k_state'].get('rf') or 0)>0
    guard_rows=_KT_GUARD_ROWS if refined else max(64,_KT_GUARD_ROWS)
    minimum_gain=_KT_MIN_GAIN if refined else min(.005,_KT_MIN_GAIN)
    ix=torch.linspace(0,rows-1,min(rows,guard_rows)).long()
    KT_DEBUG.update(guard_rows=ix.numel(),minimum_gain=minimum_gain)
    raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
    raw.extend(dequantize_nvfp4(*hold[r]).float() for r in 'kv')
    target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(_hc_quantize_raw(raw[1],parent['k_state'],kh,d,'k'))
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
    candidate_state=parent['k_state']
    modes=_KT_MODES or (_KT_MODE,)
    if _KT_BUDGET_ROUTE and refined:modes=(4,)
    if 4 in modes:
        probes=[]
        for s in samples[:-1][-2:]:
            count=s['q'][0].shape[0];select=torch.linspace(0,count-1,min(count,16)).long()
            qr=dequantize_nvfp4(s['q'][0][select],s['q'][1][select]).float().reshape(-1,kh,qh//kh,d)
            probes.append(qr.permute(1,0,2,3).reshape(kh,-1,d))
        candidate_state=dict(candidate_state,kt_probe=torch.cat(probes,1)/(d**.5))
    def cost(key):
        e=(_joint_output(q,key,v,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    old=cost(k);best=old.clone();chosen=torch.zeros(kh,dtype=torch.int32);details=[]
    for mode in modes:
        candidate=_deq_params(_kt_quant(raw[1],candidate_state,kh,d,mode))
        new=cost(candidate)
        take=(new<old*(1.-minimum_gain)).all(0)&(new.mean(0)<best.mean(0))
        best=torch.where(take[None],new,best);chosen=torch.where(take,mode,chosen)
        details.append(dict(mode=mode,ratios=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist(),accepted=int(take.sum())))
    active=chosen>0
    KT_DEBUG.update(accepted=int(active.sum()),modes=details,chosen=chosen.tolist())
    if not bool(active.any()):return parent
    mode=4 if bool((chosen==4).any()) else int(chosen.max())
    candidate_state=dict(candidate_state,kt_mode=mode,kt_active=active)
    if mode==4:candidate_state['kt_weighted']=chosen==4
    else:candidate_state.pop('kt_probe',None)
    return dict(parent,k_state=candidate_state)


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_kt_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _kt_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        KT_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    if not isinstance(k_state,dict) or 'kt_mode' not in k_state:
        return _kt_parent_k(k_quant,k_scale,kv_num_heads,head_dim,k_state)
    raw=dequantize_nvfp4(k_quant,k_scale).float()
    if raw.shape[0]>4096:return _kt_parent_k(k_quant,k_scale,kv_num_heads,head_dim,k_state)
    return _kt_quant(raw,k_state,kv_num_heads,head_dim,k_state['kt_mode'])
