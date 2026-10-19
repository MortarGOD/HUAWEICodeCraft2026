"""Adapt the historical shared-rotation learner to v365's complete K state."""
_LR_STEPS=2
_LR_RATE=.03
_LR_QUERIES=16
_LR_MAX_ROWS=1024
_LR_BLOCK=0
_LR_HEAD_CAP=0


def _cov_transform(raw,st,nh,kh,d,role):
    return _kt_input(raw,st,nh,d,role).reshape(-1,nh,d).transpose(0,1).contiguous()


def _cr_role(old,rotation,kh,d):
    changed=(rotation!=torch.eye(d)).any((1,2))
    if not bool(changed.any()):return old
    st=dict(old)
    if 'joint_transform' in old:base=old['joint_transform'].float()
    else:
        base=_make_R(d,int(old.get('rseed') or 0)) if old.get('rot') else torch.eye(d)
        base=base[None].expand(kh,d,d)
    st['joint_transform']=torch.where(changed[:,None,None],base@rotation,base).contiguous()
    if 'u' in old:
        u=old['u'].double();r=rotation.double().repeat_interleave(len(u)//kh,0)
        inverse=r.transpose(1,2)@(u.transpose(1,2)@u)@r
        new=torch.linalg.cholesky((inverse+inverse.transpose(1,2))*.5,upper=True).to(old['u'].dtype)
        st['u']=torch.where(changed.repeat_interleave(len(u)//kh)[:,None,None],new,old['u']).contiguous()
    if 'H' in old:
        h=old['H'].float();r=rotation.repeat_interleave(len(h)//kh,0)
        new=(r.transpose(1,2)@h@r).to(old['H'].dtype)
        st['H']=torch.where(changed.repeat_interleave(len(h)//kh)[:,None,None],new,old['H']).contiguous()
    if isinstance(old.get('ts_fallback'),dict):st['ts_fallback']=_cr_role(old['ts_fallback'],rotation,kh,d)
    # kt_probe multiplies RAW K, so it must not be rotated with deployed K.
    return st


def _lr_states(states,rotation,qh,kh,d):
    if torch.equal(rotation,torch.eye(d).expand_as(rotation)):return states
    return dict(states,**{r+'_state':_cr_role(states[r+'_state'],rotation,kh,d) for r in 'qk'})


def _cr_guard(hold,states,rotation,qh,kh,d):
    proposed=_lr_states(states,rotation,qh,kh,d)
    if proposed is states:return states
    rows=len(hold['q'][0]);ix=torch.linspace(0,rows-1,min(rows,32)).long()
    raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
    raw += [dequantize_nvfp4(*hold[r]).float() for r in 'kv']
    target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
    value=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,states['v_state']))
    costs=[]
    for st in (states,proposed):
        q=_joint_quant_queries(hold,st['q_state'],qh,kh,d,ix)
        k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,st['k_state']))
        e=(_joint_output(q,k,value,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
        costs.append(torch.stack((e[::2].mean(0),e[1::2].mean(0))))
    take=(costs[1]<costs[0]*.995).all(0)&torch.isfinite(costs[1]).all(0)&torch.isfinite(costs[0]).all(0)
    LR_DEBUG.update(adopted=int(take.sum()),hold_fold_ratios=(costs[1]/costs[0].clamp_min(1.e-30)).tolist())
    selected=torch.where(take[:,None,None],rotation,torch.eye(d))
    return _lr_states(states,selected,qh,kh,d)


def hif4_calibration_attention(samples,qh,kh,d):
    states=_lr_parent_calibration(samples,qh,kh,d)
    LR_DEBUG.clear();LR_DEBUG.update(adopted=0,error=False)
    if (len(samples)<3 or not states['q_state'].get('gq') or not states['k_state'].get('gq')
            or d%64 or not 4<=len(samples[-1]['q'][0])<=4096):return states
    try:
        rotation=_lr_train(samples,states,qh,kh,d)
        return states if rotation is None else _cr_guard(samples[-1],states,rotation,qh,kh,d)
    except (RuntimeError,ValueError) as exc:
        LR_DEBUG.update(error=True,error_type=type(exc).__name__)
        return states
