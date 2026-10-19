"""V-only extended sf grid with unchanged parent postprocessing."""
VG_DEBUG={}
_VG_GRID=tuple((off,sig) for off in (-1,0,1,2) for sig in (1.,1.25,1.5,1.75))
_vg_parent_calibration=hif4_calibration_attention
_vg_parent_v=hif4_dynamic_quantize_v


def _vg_table(raw,state,has_scale):
    return _quantize_weighted(raw,torch.ones(1,raw.shape[1]),grid=_VG_GRID)


def _vg_guard(samples,parent,qh,kh,d):
    VG_DEBUG.clear();VG_DEBUG.update(error=False,adopted=0)
    if len(samples)<3 or d%64 or not parent['q_state'].get('gq'):return parent
    hold=samples[-1];t=hold['v'][0].shape[0]
    if not 4<=t<=4096 or hold['v'][0].numel()>4194304:return parent
    ix=torch.linspace(0,t-1,min(32,t)).long()
    rawq=dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()
    rawk=dequantize_nvfp4(*hold['k']).float();rawv=dequantize_nvfp4(*hold['v']).float()
    q=_joint_quant_queries(hold,parent['q_state'],qh,kh,d,ix)
    k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
    p=_vr_probs(q,k,qh,kh,d);target=_vr_probs(rawq,rawk,qh,kh,d)@rawv.reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    base=_vg_parent_v(*hold['v'],kh,d,parent['v_state']);trial=_vg_v(*hold['v'],kh,d,parent['v_state'])
    def loss(params):
        values=_deq_params(params).reshape(t,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        e=(p@values-target).square().mean(-1).reshape(kh,qh//kh,-1).mean(1)
        return torch.stack((e[:,::2].mean(1),e[:,1::2].mean(1)))
    old=loss(base);new=loss(trial);active=(new<old*.995).all(0)&torch.isfinite(new).all(0)
    VG_DEBUG.update(adopted=int(active.sum()),hold_ratio=(new.mean(0)/old.mean(0).clamp_min(1e-30)).tolist())
    if not bool(active.any()):return parent
    return dict(parent,v_state=dict(wide_grid=dict(active=active,parent=parent['v_state'])))


def hif4_calibration_attention(calib_qkv_list,q_num_heads,kv_num_heads,head_dim):
    parent=_vg_parent_calibration(calib_qkv_list,q_num_heads,kv_num_heads,head_dim)
    try:return _vg_guard(calib_qkv_list,parent,q_num_heads,kv_num_heads,head_dim)
    except Exception as exc:
        VG_DEBUG.update(error=True,error_type=type(exc).__name__);return parent


def hif4_dynamic_quantize_v(v_quant,v_scale,kv_num_heads,head_dim,v_state):
    if not isinstance(v_state,dict) or 'wide_grid' not in v_state:
        return _vg_parent_v(v_quant,v_scale,kv_num_heads,head_dim,v_state)
    st=v_state['wide_grid']
    if not 4<=v_quant.shape[0]<=4096 or v_quant.numel()>4194304:
        return _vg_parent_v(v_quant,v_scale,kv_num_heads,head_dim,st['parent'])
    out=_vg_v(v_quant,v_scale,kv_num_heads,head_dim,st['parent'])
    if bool(st['active'].all()):return out
    base=_vg_parent_v(v_quant,v_scale,kv_num_heads,head_dim,st['parent'])
    active=st['active'].repeat_interleave(head_dim//64)
    return {k:torch.where(active.reshape(1,-1,*([1]*(v.ndim-2))),out[k],v) for k,v in base.items()}
