"""Re-select Q/K algorithms with full keys and explicitly chosen V objective."""
_GATE_VALUE='deployed'
_GATE_MARGIN=.005
_GATE_HEAD_CAP=1.01
GATE_DEBUG={}
_gate_parent_calibration=hif4_calibration_attention
_gate_parent_q=hif4_dynamic_quantize_q
_gate_parent_k=hif4_dynamic_quantize_k


def _gate_fit(samples,states,qh,kh,d):
    hk=torch.zeros(kh,d,d);hq=torch.zeros(kh,d,d);included=[]
    for sample in samples[:-1]:
        rows=sample['q'][0].shape[0]
        if not 1<=rows<=1024:continue
        q=_cov_transform(dequantize_nvfp4(*sample['q']).float(),states['q_state'],qh,kh,d,'q')
        k=_cov_transform(dequantize_nvfp4(*sample['k']).float(),states['k_state'],kh,kh,d,'k')
        hk+=k.transpose(1,2)@k
        hq+=(q.transpose(1,2)@q).reshape(kh,qh//kh,d,d).sum(1)
        included.append(rows)
    GATE_DEBUG['fit_rows']=included
    return hk,hq


def _gate_bank(old,h,hold_rows):
    valid=bool(torch.isfinite(h).all() and (h.diagonal(dim1=-2,dim2=-1).sum(-1)>0).all())
    u=old.get('u')
    if u is None and valid:u=_upper_cholesky_inv(h)
    hh=old.get('H')
    if hh is None and valid:hh=h.bfloat16().contiguous()
    bank=[old];labels=['parent']
    parent_rf=int(old.get('rf') or 0)
    # No evidence for a different short-T refinement policy from a long hold.
    rfs=(0,1) if hold_rows<=ATTN_REFINE_MAX_T else (parent_rf,)
    for gq in (0,1):
        for rf in rfs:
            if gq and u is None:continue
            if rf and hh is None:continue
            st=dict(old,gq=gq,rf=rf)
            if gq:st['u']=u.contiguous()
            if rf:st['H']=hh.contiguous()
            if gq==int(old.get('gq') or 0) and rf==parent_rf:continue
            bank.append(st);labels.append('gq%d_rf%d'%(gq,rf))
    return bank,labels


def _gate_select(costs,kh):
    # costs[Qvariant,Kvariant,fold,Qhead]. Q heads are independent; K is shared.
    nq,nk,folds,qh=costs.shape;rep=qh//kh
    qi=torch.zeros(qh,dtype=torch.int64);ki=torch.zeros(kh,dtype=torch.int64)
    for hv in range(kh):
        sl=slice(hv*rep,(hv+1)*rep)
        best=costs[0,0,:,sl].clone()
        best_q=torch.zeros(rep,dtype=torch.int64);best_k=0
        for kk in range(nk):
            chosen=costs[0,kk,:,sl].clone();qids=torch.zeros(rep,dtype=torch.int64)
            for qq in range(1,nq):
                cand=costs[qq,kk,:,sl]
                take=torch.isfinite(cand).all(0)&(cand<chosen*(1-_GATE_MARGIN)).all(0)
                chosen=torch.where(take[None],cand,chosen);qids=torch.where(take,qq,qids)
            if (bool(torch.isfinite(chosen).all()) and
                bool((chosen.sum(-1)<best.sum(-1)*(1-_GATE_MARGIN)).all()) and
                (not _GATE_HEAD_CAP or bool((chosen<=best*_GATE_HEAD_CAP).all()))):
                best=chosen;best_q=qids;best_k=kk
        qi[sl]=best_q;ki[hv]=best_k
    return qi,ki


def _gate_frame_candidates(samples,states,qh,kh,d,rows):
    banks=[states];labels=['parent']
    fitted,_=_qks_fit(samples,qh,kh,d)
    scales=(None,fitted) if fitted is not None else (None,)
    for scaled,scale in enumerate(scales):
        for rotated in (0,1):
            if rotated and _make_R(d,0) is None:continue
            frame={}
            for role in 'qk':
                st=dict(kvh=kh,rot=rotated,rseed=0,gq=1,rf=int(states[role+'_state'].get('rf') or 0))
                if scale is not None:
                    st['qs' if role=='q' else 'ks']=(scale if role=='q' else 1/scale).bfloat16().contiguous()
                frame[role+'_state']=st
            hk,hq=_gate_fit(samples,frame,qh,kh,d)
            valid=True
            for role,h in (('q',hk),('k',hq)):
                if not bool(torch.isfinite(h).all() and (h.diagonal(dim1=-2,dim2=-1).sum(-1)>0).all()):
                    valid=False;break
                u=_upper_cholesky_inv(h)
                if u is None:valid=False;break
                frame[role+'_state'].update(u=u.contiguous(),H=h.bfloat16().contiguous())
            if valid:
                banks.append(frame);labels.append('scale%d_rot%d_gptq'%(scaled,rotated))
    return banks,labels


def _gate_select_frames(costs,kh):
    qh=costs.shape[-1];rep=qh//kh
    choice=torch.zeros(kh,dtype=torch.int64)
    for hv in range(kh):
        sl=slice(hv*rep,(hv+1)*rep);best=costs[0,:,sl]
        for i in range(1,costs.shape[0]):
            candidate=costs[i,:,sl]
            if (bool(torch.isfinite(candidate).all()) and
                bool((candidate.sum(-1)<best.sum(-1)*(1-_GATE_MARGIN)).all()) and
                (not _GATE_HEAD_CAP or bool((candidate<=best*_GATE_HEAD_CAP).all()))):
                choice[hv]=i;best=candidate
    return choice


def _gate_frames(samples,states,qh,kh,d):
    hold=samples[-1];rows=hold['q'][0].shape[0]
    frames,labels=_gate_frame_candidates(samples,states,qh,kh,d,rows)
    ix=torch.linspace(0,rows-1,min(rows,64)).long()
    raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    target=_joint_output(*raw,qh,kh,d,ix).reshape(-1,qh,d)
    value=raw[2] if _GATE_VALUE=='frames_raw' else _deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,states['v_state']))
    costs=[]
    for frame in frames:
        q=_gate_query_values(hold,frame['q_state'],qh,kh,d,ix)
        k=_deq_params(_gate_parent_k(*hold['k'],kh,d,frame['k_state']))
        out=_joint_output(q,k,value,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d)
        e=(out-target).square().mean(-1)
        costs.append(torch.stack((e[::2].mean(0),e[1::2].mean(0))))
    costs=torch.stack(costs);choice=_gate_select_frames(costs,kh)
    answer=dict(states)
    if bool(choice.any()):
        for role,rep in (('q',qh//kh),('k',1)):
            answer[role+'_state']=dict(gate_bank=[s[role+'_state'] for s in frames],gate_choice=choice.repeat_interleave(rep))
    GATE_DEBUG.update(q_changed=int((choice!=0).sum())*(qh//kh),k_changed=int((choice!=0).sum()),
                     frame_labels=labels,frame_choice=choice.tolist(),frame_costs=costs.tolist(),
                     hold_rows=rows,refinement_changed=False,
                     parent_gq=[int(states[r+'_state'].get('gq') or 0) for r in 'qk'])
    return answer


def _gate_dynamic(quant,scale,heads,d,state,parent):
    if not isinstance(state,dict) or 'gate_bank' not in state:
        return parent(quant,scale,heads,d,state)
    bank=state['gate_bank'];choice=state['gate_choice']
    if quant.shape[0]<=1:return parent(quant,scale,heads,d,bank[0])
    indices=choice.unique().tolist()
    out=parent(quant,scale,heads,d,bank[indices[0]])
    for index in indices[1:]:
        candidate=parent(quant,scale,heads,d,bank[index])
        mask=(choice==index).repeat_interleave(d//64)[None,:,None,None,None]
        out={key:torch.where(mask,candidate[key],out[key]) for key in out}
    return out


def _gate_query_values(hold,state,qh,kh,d,ix):
    if state.get('gq')==1:
        # Parent helper retains the original T refinement budget, full columns
        # and real GPTQ; only independent query rows are omitted.
        return _joint_quant_queries(hold,state,qh,kh,d,ix)
    return _deq_params(_gate_parent_q(*hold['q'],qh,d,state))[ix]


def hif4_dynamic_quantize_q(quant,scale,heads,d,state):
    return _gate_dynamic(quant,scale,heads,d,state,_gate_parent_q)


def hif4_dynamic_quantize_k(quant,scale,heads,d,state):
    return _gate_dynamic(quant,scale,heads,d,state,_gate_parent_k)


def hif4_calibration_attention(samples,qh,kh,d):
    states=_gate_parent_calibration(samples,qh,kh,d)
    GATE_DEBUG.clear();GATE_DEBUG.update(q_changed=0,k_changed=0)
    rows=samples[-1]['q'][0].shape[0]
    if len(samples)<2 or not 4<=rows<=4096 or d%64:return states
    if _GATE_VALUE.startswith('frames'):return _gate_frames(samples,states,qh,kh,d)
    hk,hq=_gate_fit(samples,states,qh,kh,d)
    qb,ql=_gate_bank(states['q_state'],hk,rows)
    kb,kl=_gate_bank(states['k_state'],hq,rows)
    hold=samples[-1];ix=torch.linspace(0,rows-1,min(rows,64)).long()
    raw=[dequantize_nvfp4(*hold[r]).float() for r in 'qkv']
    target=_joint_output(*raw,qh,kh,d,ix).reshape(-1,qh,d)
    deployed=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,states['v_state']))
    value=raw[2] if _GATE_VALUE=='raw' else deployed
    qs=[_gate_query_values(hold,st,qh,kh,d,ix) for st in qb]
    ks=[_deq_params(_gate_parent_k(*hold['k'],kh,d,st)) for st in kb]
    costs=torch.empty(len(qb),len(kb),2,qh)
    for qindex,q in enumerate(qs):
        for kindex,k in enumerate(ks):
            out=_joint_output(q,k,value,qh,kh,d,ix,query_subset=True).reshape(-1,qh,d)
            error=(out-target).square().mean(-1)
            costs[qindex,kindex]=torch.stack((error[::2].mean(0),error[1::2].mean(0)))
    qi,ki=_gate_select(costs,kh)
    answer=dict(states)
    for role,bank,choice in (('q',qb,qi),('k',kb,ki)):
        if bool((choice!=0).any()):
            answer[role+'_state']=dict(gate_bank=bank,gate_choice=choice)
    GATE_DEBUG.update(q_changed=int((qi!=0).sum()),k_changed=int((ki!=0).sum()),
                     q_labels=ql,k_labels=kl,q_choice=qi.tolist(),k_choice=ki.tolist(),
                     parent_gq=[int(states[r+'_state'].get('gq') or 0) for r in 'qk'],
                     parent_rf=[int(states[r+'_state'].get('rf') or 0) for r in 'qk'],
                     q_u_rebuilt='u' not in states['q_state'],k_u_rebuilt='u' not in states['k_state'],
                     hold_rows=rows,refinement_tested=rows<=ATTN_REFINE_MAX_T,
                     costs=costs.tolist())
    return answer
