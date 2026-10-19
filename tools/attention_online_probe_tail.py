"""Single selected calibration candidate, reusing unchanged public quantizers."""
LR_DEBUG={}
RG_DEBUG={}
_LR_HEAD_CAP=0
_rg_parent_calibration=hif4_calibration_attention


def _rg_transform(samples,states,qh,kh,d,name):
    if name.startswith('whiten_'):
        role=name[-1];power=.5 if role=='q' else -.5
        u=states[role+'_state']['u'].float()
        inverse=torch.linalg.solve_triangular(u,torch.eye(d).expand_as(u),upper=True)
        h=(inverse@inverse.transpose(1,2)).reshape(kh,-1,d,d).mean(1)
        e,v=torch.linalg.eigh((h+h.transpose(1,2))*.5)
        e=e.clamp_min(e.amax(-1)[:,None]*.001)
        gain=e.pow(power);gain=(gain/gain.log().mean(-1).exp()[:,None]).clamp(.25,4.)
        return (v*gain[:,None])@v.transpose(1,2)
    energies=[]
    for role,nh in (('q',qh),('k',kh)):
        values=[]
        for s in samples[:-1][-2:]:
            t=s[role][0].shape[0];ix=torch.linspace(0,t-1,min(128,t)).long()
            raw=dequantize_nvfp4(s[role][0][ix],s[role][1][ix]).float()
            x=_kt_input(raw,states[role+'_state'],nh,d,role).reshape(-1,kh,nh//kh,d)
            values.append(x.square().mean((0,2)))
        energies.append(torch.stack(values).mean(0).clamp_min(1e-20).sqrt())
    q,k=energies
    if name.startswith('balance_'):
        power=.5 if name=='balance_0.5' else (1. if name=='balance_1.0' else -1.)
        return torch.diag_embed((k/q).sqrt().clamp(.25,4.).pow(power))
    key=k if name=='key_sorted' else (q if name=='query_sorted' else q.log()+k.log())
    order=key.argsort(-1);eye=torch.eye(d).expand(kh,d,d)
    if name!='cross_pair_mix':return eye.gather(2,order[:,None].expand(kh,d,d)).contiguous()
    mix=torch.zeros_like(eye)
    for head in range(kh):
        lo=order[head,:d//2];hi=order[head,d//2:].flip(0);cols=torch.arange(d//2)*2
        mix[head,lo,cols]=2**-.5;mix[head,hi,cols]=2**-.5
        mix[head,lo,cols+1]=2**-.5;mix[head,hi,cols+1]=-2**-.5
    return mix


def _rg_apply(samples,parent,qh,kh,d):
    RG_DEBUG.clear();LR_DEBUG.clear();RG_DEBUG.update(mode=_RG_MODE,error=False)
    if len(samples)<3 or d%64 or not 4<=samples[-1]['q'][0].shape[0]<=4096:
        return parent
    if not all(parent[r+'_state'].get('gq') and 'u' in parent[r+'_state'] for r in 'qk'):
        return parent
    if _RG_MODE in ('q_sharp','q_flat','both_flat','both_shrink'):
        proposed=metric_states(parent,_RG_MODE,qh,kh,d)
    else:
        r=_rg_transform(samples,parent,qh,kh,d,_RG_MODE)
        if _RG_MODE.startswith(('balance_','whiten_')):
            proposed,error=nonorthogonal_states(parent,r,qh,kh,d)
            if not bool(torch.isfinite(r).all()) or error>=1e-4:return parent
        else:proposed=_lr_states(parent,r,qh,kh,d)
    result=_lr_guard(samples[-1],parent,proposed,qh,kh,d)
    RG_DEBUG.update(LR_DEBUG)
    return result


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_rg_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _rg_apply(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        RG_DEBUG.update(error=True,error_type=type(exc).__name__)
        return parent
