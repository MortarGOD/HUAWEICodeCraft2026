"""Post-deployment quantized-Query regression folded into Key encoding."""
KC_DEBUG={}
_KC_MODE=1
_kc_parent_calibration=hif4_calibration_attention
_kc_parent_k=hif4_dynamic_quantize_k


def _kc_fit(samples,parent,qh,kh,d,mode):
    features=[];targets=[]
    for sample in samples[:-1][-2:]:
        n=sample['q'][0].shape[0];ix=torch.linspace(0,n-1,min(n,128)).long()
        raw=dequantize_nvfp4(sample['q'][0][ix],sample['q'][1][ix]).float()
        quant=_joint_quant_queries(sample,parent['q_state'],qh,kh,d,ix)
        features.append(quant.reshape(-1,kh,qh//kh,d).permute(1,0,2,3).reshape(kh,-1,d))
        targets.append(raw.reshape(-1,kh,qh//kh,d).permute(1,0,2,3).reshape(kh,-1,d))
    x=torch.cat(features,1);y=torch.cat(targets,1)
    st=parent['k_state']
    identity=torch.eye(d).repeat(1,kh)
    anchor=_kt_input(identity,st,kh,d,'k').reshape(d,kh,d).transpose(0,1).transpose(1,2)
    gram=x.transpose(1,2)@x
    ridge=.01*gram.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1e-20)
    regularized=gram+ridge[:,None,None]*torch.eye(d)
    cross=x.transpose(1,2)@y+ridge[:,None,None]*anchor
    fit=torch.linalg.solve(regularized,cross)
    transform=(anchor+(.5 if mode==1 else 1.)*(fit-anchor)).transpose(1,2).contiguous()
    result=dict(st,kc_transform=transform)
    if mode==3:
        u=_upper_cholesky_inv(gram)
        if u is not None:result['u']=u.contiguous();result['H']=gram.to(torch.bfloat16).contiguous()
    return result


def _kc_quant(raw,st,kh,d):
    y=(raw.reshape(-1,kh,d).transpose(0,1)@st['kc_transform'].float()).transpose(0,1)
    key=raw.reshape(-1,kh,d).transpose(0,1)
    probability=torch.softmax(st['kt_probe'].float()@key.transpose(1,2),-1)
    weight=(.9*probability.mean(1)+.1/raw.shape[0]).transpose(0,1)[:,:,None]
    center=(y*weight).sum(0,keepdim=True)
    for _ in range(8):
        z=y-center
        error=_deq_params(_dyn_table(z.reshape(raw.shape),None,has_scale=False)).reshape(y.shape)-z
        center=center-(error*weight).sum(0,keepdim=True)
    plain={k:v for k,v in st.items() if k not in ('joint_transform','ks','rot','hc_matrix')}
    return _kt_fixed_grid((y-center).reshape(raw.shape).contiguous(),plain,kh,d,'k')


def _kc_guard(samples,parent,qh,kh,d):
    KC_DEBUG.clear();KC_DEBUG.update(accepted=0,error=False)
    st=parent['k_state'];hold=samples[-1];t=hold['k'][0].shape[0]
    if len(samples)<3 or d%64 or not 4<=t<=4096 or not st.get('gq') or 'u' not in st:return parent
    proposed=_kc_fit(samples,parent,qh,kh,d,_KC_MODE)
    if 'kt_probe' not in proposed:
        probes=[]
        for sample in samples[:-1][-2:]:
            count=sample['q'][0].shape[0];ixq=torch.linspace(0,count-1,min(count,16)).long()
            qr=dequantize_nvfp4(sample['q'][0][ixq],sample['q'][1][ixq]).float().reshape(-1,kh,qh//kh,d)
            probes.append(qr.permute(1,0,2,3).reshape(kh,-1,d))
        proposed['kt_probe']=torch.cat(probes,1)/(d**.5)
    ix=torch.linspace(0,t-1,min(t,64)).long()
    raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
    raw.extend(dequantize_nvfp4(*hold[r]).float() for r in 'kv')
    target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(_kc_parent_k(*hold['k'],kh,d,st))
    v=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
    def cost(key):
        e=(_joint_output(q,key,v,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        return torch.stack((e[::2].mean(0),e[1::2].mean(0)))
    trial=_deq_params(_kc_quant(raw[1],proposed,kh,d))
    old=cost(k);new=cost(trial)
    take=(new<old*.995).all(0)
    KC_DEBUG.update(accepted=int(take.sum()),ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(take.any()):return parent
    return dict(parent,k_state=dict(proposed,kc_mode=_KC_MODE,kc_active=take,kc_fallback=st))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_kc_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _kc_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        KC_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_k(k_quant,k_scale,kv_num_heads,head_dim,k_state):
    if not isinstance(k_state,dict) or 'kc_mode' not in k_state or not 4<=k_quant.shape[0]<=4096:
        fallback=k_state.get('kc_fallback',k_state) if isinstance(k_state,dict) else k_state
        return _kc_parent_k(k_quant,k_scale,kv_num_heads,head_dim,fallback)
    raw=dequantize_nvfp4(k_quant,k_scale).float()
    new=_kc_quant(raw,k_state,kv_num_heads,head_dim)
    if bool(k_state['kc_active'].all()):return new
    old=_kc_parent_k(k_quant,k_scale,kv_num_heads,head_dim,k_state['kc_fallback'])
    mask=k_state['kc_active'].repeat_interleave(head_dim//64)
    return {name:torch.where(mask.reshape(1,-1,*([1]*(a.ndim-2))),a,old[name]) for name,a in new.items()}
