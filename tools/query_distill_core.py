"""Calibrate non-invertible Q correction against deployed softmax-K-V output.

Dynamic Q reads only current Q and CPU calibration parameters. This prototype
keeps a full parent call for exact fallback heads; it is not performance-tuned.
"""
_QD_STEPS=6
_QD_RATE=.02
_QD_QUERIES=32
_QD_MODE='both'
_QD_CAP=.2
QD_DEBUG={}
_qd_parent_calibration=hif4_calibration_attention
_qd_parent_q=hif4_dynamic_quantize_q


def _qd_quant(x,base,state,qh,d,rows,gain,bias):
    active=(gain!=0).any(-1)|(bias!=0).any(-1)
    if not bool(active.any()):return base
    shifted=x*(1+gain[:,None,:])+bias[:,None,:]
    flat=shifted.transpose(0,1).reshape(-1,qh*d).contiguous()
    p=_dyn_table(flat,None,has_scale=False)
    unit=_params_unit_flat(p)
    us=unit.reshape(-1,qh,d).transpose(0,1).contiguous()
    u=state['u'].float().repeat_interleave(qh//state['u'].shape[0],0)
    values=_gptq_quantize_batched(shifted.contiguous(),us,u)
    values=values.transpose(0,1).reshape(flat.shape).contiguous()
    rf=int(state.get('rf') or 0)
    if rf and rows<=ATTN_REFINE_MAX_T and 'H' in state:
        rounds=rf*(32 if rows<=128 else _REF_ATTN_ROUNDS)
        values=_refine_attn_heads(flat,values,unit,state['H'],qh,rf,total_rounds=rounds)
    candidate=_values_to_params(values,p)
    mask=active.repeat_interleave(d//64)[None,:,None,None,None]
    return {k:torch.where(mask,candidate[k],base[k]) for k in base}


def hif4_dynamic_quantize_q(quant,scale,qh,d,state):
    base=_qd_parent_q(quant,scale,qh,d,state)
    if not isinstance(state,dict) or 'qd_gain' not in state or quant.shape[0]<=1:
        return base
    x=dequantize_nvfp4(quant,scale).float()
    x=_cov_transform(x,state,qh,state['kvh'],d,'q')
    return _qd_quant(x,base,state,qh,d,quant.shape[0],state['qd_gain'],state['qd_bias'])


def _qd_data(samples,states,qh,kh,d,queries):
    data=[]
    for sample in samples:
        rows=sample['q'][0].shape[0]
        if not 2<=rows<=4096:continue
        ix=torch.linspace(0,rows-1,min(rows,queries)).long()
        raw={r:dequantize_nvfp4(*sample[r]).float() for r in 'qkv'}
        base=_qd_parent_q(*sample['q'],qh,d,states['q_state'])
        base={k:v[ix].clone() for k,v in base.items()}
        q=_cov_transform(raw['q'][ix],states['q_state'],qh,kh,d,'q')
        k=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,states['k_state']))
        v=_deq_params(hif4_dynamic_quantize_v(*sample['v'],kh,d,states['v_state']))
        k=k.reshape(rows,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        v=v.reshape(rows,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
        target=_joint_output(raw['q'],raw['k'],raw['v'],qh,kh,d,ix)
        target=target.reshape(-1,qh,d).transpose(0,1)
        data.append(dict(x=q,base=base,k=k,v=v,target=target,rows=rows))
    return data


def _qd_loss(q,item,d):
    output=torch.softmax(q@item['k'].transpose(1,2)/d**.5,-1)@item['v']
    return (output-item['target']).square().mean((1,2))


def _qd_forward(item,state,qh,d,gain,bias):
    p=_qd_quant(item['x'],item['base'],state,qh,d,item['rows'],gain,bias)
    return _deq_params(p).reshape(-1,qh,d).transpose(0,1)


def _qd_evaluate(data,state,qh,d,gain,bias):
    return torch.stack([_qd_loss(_qd_forward(item,state,qh,d,gain,bias),item,d) for item in data])


def _qd_train(samples,states,qh,kh,d):
    # No teacher/test tensor or optimizer is retained in the returned state.
    with torch.inference_mode(False),torch.enable_grad():
        with torch.no_grad():
            data=_qd_data(samples[:-1],states,qh,kh,d,_QD_QUERIES)
            if not data:return None
            rms=torch.stack([a['x'].square().mean(1) for a in data]).mean(0).sqrt().clamp_min(1e-8)
            gain=torch.zeros(qh,d);shift=torch.zeros_like(gain)
            initial=_qd_evaluate(data,states['q_state'],qh,d,gain,shift)
            norm=initial.clamp_min(1e-20)
            best=(initial/norm).mean(0)
        history=[best.tolist()];accepted=torch.zeros(qh,dtype=torch.int64)
        for _ in range(_QD_STEPS):
            gain=gain.detach().requires_grad_(True);shift=shift.detach().requires_grad_(True)
            losses=[]
            for i,item in enumerate(data):
                with torch.no_grad():
                    quant=_qd_forward(item,states['q_state'],qh,d,gain.detach(),shift.detach()*rms)
                smooth=item['x']*(1+gain[:,None,:])+shift[:,None,:]*rms[:,None,:]
                straight=quant+(smooth-smooth.detach())
                losses.append(_qd_loss(straight,item,d)/norm[i])
            gradient=torch.autograd.grad(torch.stack(losses).mean(),(gain,shift))
            with torch.no_grad():
                gg,gb=gradient
                if _QD_MODE=='bias':gg=torch.zeros_like(gg)
                if _QD_MODE=='gain':gb=torch.zeros_like(gb)
                magnitude=(gg.square().mean(-1)+gb.square().mean(-1)).sqrt().clamp_min(1e-20)
                gg=gg/magnitude[:,None];gb=gb/magnitude[:,None]
                chosen_gain=gain.detach();chosen_shift=shift.detach();improved=False
                for fraction in (1.,.25):
                    ga=(gain.detach()-_QD_RATE*fraction*gg).clamp(-_QD_CAP,_QD_CAP)
                    sh=(shift.detach()-_QD_RATE*fraction*gb).clamp(-_QD_CAP,_QD_CAP)
                    scores=(_qd_evaluate(data,states['q_state'],qh,d,ga,sh*rms)/norm).mean(0)
                    take=torch.isfinite(scores)&(scores<best*.9999)
                    chosen_gain=torch.where(take[:,None],ga,chosen_gain)
                    chosen_shift=torch.where(take[:,None],sh,chosen_shift)
                    best=torch.where(take,scores,best);accepted+=take.to(torch.int64)
                    improved=improved or bool(take.any())
                gain=chosen_gain.detach();shift=chosen_shift.detach();history.append(best.tolist())
                if not improved:break
        QD_DEBUG.update(training_history=history,accepted_trials=accepted.tolist(),
                        training_rows=[item['rows'] for item in data],max_gain=float(gain.abs().max()),
                        max_normalized_bias=float(shift.abs().max()))
        return gain.detach(),(shift*rms).detach()


def _qd_guard(hold,states,qh,kh,d,gain,bias):
    data=_qd_data([hold],states,qh,kh,d,64)
    if not data:return states
    item=data[0]
    parent=_deq_params(item['base']).reshape(-1,qh,d).transpose(0,1)
    candidate=_qd_forward(item,states['q_state'],qh,d,gain,bias)
    costs=[]
    for q in (parent,candidate):
        out=torch.softmax(q@item['k'].transpose(1,2)/d**.5,-1)@item['v']
        e=(out-item['target']).square().mean(-1)
        costs.append(torch.stack((e[:,::2].mean(1),e[:,1::2].mean(1))))
    take=torch.isfinite(costs[1]).all(0)&(costs[1]<costs[0]*.995).all(0)
    QD_DEBUG.update(adopted=int(take.sum()),hold_ratios=(costs[1]/costs[0].clamp_min(1e-30)).tolist())
    if not bool(take.any()):return states
    qs=dict(states['q_state'],qd_gain=torch.where(take[:,None],gain,0.).contiguous(),
            qd_bias=torch.where(take[:,None],bias,0.).contiguous())
    return dict(states,q_state=qs)


def hif4_calibration_attention(samples,qh,kh,d):
    states=_qd_parent_calibration(samples,qh,kh,d)
    QD_DEBUG.clear();QD_DEBUG.update(adopted=0,error=False)
    if len(samples)<2 or not states['q_state'].get('gq') or d%64:return states
    proposal=_qd_train(samples,states,qh,kh,d)
    if proposal is None:return states
    return _qd_guard(samples[-1],states,qh,kh,d,*proposal)
