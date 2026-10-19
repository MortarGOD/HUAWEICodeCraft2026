"""Calibration-only orthogonal search with true GPTQ forward and STE direction."""
_LR_STEPS=4
_LR_RATE=.05
_LR_QUERIES=16
_LR_MAX_ROWS=1024
_LR_OBJECTIVE='output'
_LR_BLOCK=0
_LR_HEAD_CAP=1.05
_LR_EARLY_STOP=True
LR_DEBUG={}
_lr_parent_calibration=hif4_calibration_attention


def _lr_cayley(rotation,gradient,rate):
    dtype=rotation.dtype
    rotation=rotation.double();gradient=gradient.double()
    skew=(rotation.transpose(1,2)@gradient-gradient.transpose(1,2)@rotation)*.5
    if _LR_BLOCK:
        groups=torch.arange(rotation.shape[-1])//_LR_BLOCK
        skew=skew*(groups[:,None]==groups[None,:])
    norm=skew.square().sum((1,2)).sqrt().clamp_min(1e-20)
    skew=skew*(rotation.shape[-1]**.5/norm)[:,None,None]
    eye=torch.eye(rotation.shape[-1],dtype=rotation.dtype).expand_as(rotation)
    update=torch.linalg.solve(eye+rate*.5*skew,eye-rate*.5*skew)
    return (rotation@update).to(dtype).detach()


def _lr_states(states,rotation,qh,kh,d):
    if torch.equal(rotation,torch.eye(d).expand_as(rotation)):
        return states
    result=dict(states)
    for role in 'qk':
        old=states[role+'_state'];st=dict(old)
        if 'joint_transform' in old:
            base=old['joint_transform'].float()
        else:
            base=_make_R(d,int(old.get('rseed') or 0)) if old.get('rot') else torch.eye(d)
            base=base[None].expand(kh,d,d)
        st['joint_transform']=(base@rotation).contiguous()
        u=old['u'].float();r=rotation.repeat_interleave(u.shape[0]//kh,0)
        inverse=r.transpose(1,2)@(u.transpose(1,2)@u)@r
        st['u']=torch.linalg.cholesky((inverse+inverse.transpose(1,2))*.5,upper=True).contiguous()
        if 'H' in old:
            h=old['H'].float();r=rotation.repeat_interleave(h.shape[0]//kh,0)
            st['H']=(r.transpose(1,2)@h@r).to(old['H'].dtype).contiguous()
        result[role+'_state']=st
    return result


def _lr_data(samples,states,qh,kh,d):
    data=[]
    eligible=[s for s in samples[:-1] if 2<=s['q'][0].shape[0]<=_LR_MAX_ROWS]
    for sample in eligible[-2:]:
        rows=sample['q'][0].shape[0]
        ix=torch.linspace(0,rows-1,min(rows,_LR_QUERIES)).long()
        raw=[dequantize_nvfp4(*sample[r]).float() for r in 'qkv']
        qb=_cov_transform(raw[0][ix],states['q_state'],qh,kh,d,'q')
        kb=_cov_transform(raw[1],states['k_state'],kh,kh,d,'k')
        value=_deq_params(hif4_dynamic_quantize_v(*sample['v'],kh,d,states['v_state']))
        value=value.reshape(rows,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        target=_joint_output(*raw,qh,kh,d,ix).reshape(ix.numel(),qh,d).transpose(0,1)
        rq=raw[0][ix].reshape(-1,qh,d).transpose(0,1)
        rk=raw[1].reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        target_logits=rq@rk.transpose(1,2)/d**.5
        data.append(dict(sample=sample,ix=ix,qbase=qb,kbase=kb,value=value,target=target,target_logits=target_logits))
    return data


def _lr_forward(item,states,qh,kh,d):
    q=_joint_quant_queries(item['sample'],states['q_state'],qh,kh,d,item['ix'])
    k=_deq_params(hif4_dynamic_quantize_k(*item['sample']['k'],kh,d,states['k_state']))
    q=q.reshape(-1,qh,d).transpose(0,1)
    k=k.reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    return q,k


def _lr_loss(q,k,item,qh,kh,d):
    logits=q@k.transpose(1,2)/d**.5
    if _LR_OBJECTIVE=='centered_logit':
        error=logits-item['target_logits']
        error=error-error.mean(-1,keepdim=True)
        return error.square().mean((1,2)).reshape(kh,qh//kh).mean(1)
    output=torch.softmax(logits,-1)@item['value']
    return (output-item['target']).square().mean((1,2)).reshape(kh,qh//kh).mean(1)


def _lr_evaluate(data,states,qh,kh,d):
    return torch.stack([_lr_loss(*_lr_forward(item,states,qh,kh,d),item,qh,kh,d) for item in data])


def _lr_train(samples,states,qh,kh,d):
    # Explicitly support callers using no_grad/inference_mode. Returned state
    # never retains an autograd graph, optimizer object or training input.
    with torch.inference_mode(False),torch.enable_grad():
        with torch.no_grad():
            data=_lr_data(samples,states,qh,kh,d)
            if not data:return None
            rotation=torch.eye(d)[None].repeat(kh,1,1)
            initial=_lr_evaluate(data,states,qh,kh,d)
            norm=initial.clamp_min(1e-20)
            best=(initial/norm).mean(0)
        history=[best.tolist()];accepted=torch.zeros(kh,dtype=torch.int64)
        for step in range(_LR_STEPS):
            rotation=rotation.detach().requires_grad_(True)
            with torch.no_grad():current=_lr_states(states,rotation.detach(),qh,kh,d)
            losses=[]
            for i,item in enumerate(data):
                with torch.no_grad():qquant,kquant=_lr_forward(item,current,qh,kh,d)
                r=rotation.repeat_interleave(qh//kh,0)
                q=item['qbase']@r
                k=(item['kbase']@rotation).repeat_interleave(qh//kh,0)
                q=qquant+(q-q.detach());k=kquant+(k-k.detach())
                losses.append(_lr_loss(q,k,item,qh,kh,d)/norm[i])
            objective=torch.stack(losses).mean()
            gradient=torch.autograd.grad(objective,rotation)[0]
            with torch.no_grad():
                chosen=rotation.detach()
                improved=False
                for fraction in (1.,.25):
                    trial=_lr_cayley(rotation.detach(),gradient,_LR_RATE*fraction)
                    proposed=_lr_states(states,trial,qh,kh,d)
                    scores=(_lr_evaluate(data,proposed,qh,kh,d)/norm).mean(0)
                    take=torch.isfinite(scores)&(scores<best*.9999)
                    chosen=torch.where(take[:,None,None],trial,chosen)
                    best=torch.where(take,scores,best)
                    accepted+=take.to(torch.int64)
                    improved=improved or bool(take.any())
                rotation=chosen.detach()
                history.append(best.tolist())
                if _LR_EARLY_STOP and not improved:
                    # No momentum, schedule or resampling changes next step;
                    # retrying the same states and rates would be redundant.
                    break
        LR_DEBUG.update(training_rows=[int(i['sample']['q'][0].shape[0]) for i in data],
                        training_history=history,accepted_trials=accepted.tolist(),
                        orthogonality_error=float((rotation.transpose(1,2)@rotation-torch.eye(d)).abs().max()))
        return rotation.detach()


def _lr_guard(hold,states,proposed,qh,kh,d):
    rows=hold['q'][0].shape[0]
    ix=torch.linspace(0,rows-1,min(rows,64)).long()
    raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    target=_joint_output(*raw,qh,kh,d,ix).reshape(-1,qh,d)
    value=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,states['v_state']))
    costs=[]
    for st in (states,proposed):
        q=_joint_quant_queries(hold,st['q_state'],qh,kh,d,ix)
        k=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,st['k_state']))
        out=_joint_output(q,k,value,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d)
        error=(out-target).square().mean(-1).reshape(-1,kh,qh//kh)
        costs.append(torch.stack((error[::2].mean(0),error[1::2].mean(0))))
    if not bool(torch.isfinite(torch.stack(costs)).all()):
        LR_DEBUG['nonfinite_guard']=True
        return states
    fold_accept=(costs[1].mean(2)<costs[0].mean(2)*.999).all(0)
    head_accept=(costs[1]<=costs[0]*_LR_HEAD_CAP).all((0,2)) if _LR_HEAD_CAP else torch.ones(kh,dtype=torch.bool)
    accept=fold_accept&head_accept
    LR_DEBUG.update(adopted=int(accept.sum()),fold_accept=fold_accept.tolist(),head_accept=head_accept.tolist(),
                    hold_fold_ratios=(costs[1].mean(2)/costs[0].mean(2).clamp_min(1e-30)).tolist(),
                    hold_ratios=(costs[1].mean((0,2))/costs[0].mean((0,2)).clamp_min(1e-30)).tolist())
    if not bool(accept.any()):return states
    result=dict(states)
    for role in 'qk':
        old,new=states[role+'_state'],proposed[role+'_state'];st=dict(new)
        if 'joint_transform' in old:base=old['joint_transform'].float()
        else:
            r=_make_R(d,int(old.get('rseed') or 0)) if old.get('rot') else torch.eye(d)
            base=r[None].expand(kh,d,d)
        st['joint_transform']=torch.where(accept[:,None,None],new['joint_transform'],base).contiguous()
        for field in ('u','H'):
            if field in old:
                mask=accept.repeat_interleave(old[field].shape[0]//kh)
                st[field]=torch.where(mask[:,None,None],new[field],old[field]).contiguous()
        result[role+'_state']=st
    return result


def hif4_calibration_attention(samples,qh,kh,d):
    states=_lr_parent_calibration(samples,qh,kh,d)
    LR_DEBUG.clear();LR_DEBUG.update(adopted=0,error=False)
    if (len(samples)<2 or not states['q_state'].get('gq') or not states['k_state'].get('gq')
            or not 4<=samples[-1]['q'][0].shape[0]<=4096):return states
    try:
        rotation=_lr_train(samples,states,qh,kh,d)
        if rotation is None:return states
        proposed=_lr_states(states,rotation,qh,kh,d)
        if proposed is states:return states
        return _lr_guard(samples[-1],states,proposed,qh,kh,d)
    except Exception as exc:
        LR_DEBUG.update(error=True,error_type=type(exc).__name__,error_message=str(exc)[:240])
        return states
