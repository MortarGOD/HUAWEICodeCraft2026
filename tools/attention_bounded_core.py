"""Bounded-query full-key deployment guard; finite sampled long-calibration Grams."""
_CAL_QUERY_CAP = 128


def _cal_output(q,k,v,qh,kh,d):
    rows=q.shape[0]
    qb=q.reshape(rows,qh,d).transpose(0,1)
    kb=k.reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    vb=v.reshape(-1,kh,d).transpose(0,1).repeat_interleave(qh//kh,0)
    chunk=max(1,min(rows,4_000_000//max(1,qh*k.shape[0])))
    pieces=[]
    for start in range(0,rows,chunk):
        pieces.append(torch.softmax(qb[:,start:start+chunk]@kb.transpose(1,2)/(d**.5),-1)@vb)
    return torch.cat(pieces,1).transpose(0,1).reshape(rows,-1)


def _cal_qk_guard(q,k,v,rot,R,Hk,Hq,Uq,Uk,qh,kh,d):
    rows=q.shape[0];rep=qh//kh
    ix=torch.linspace(0,rows-1,min(rows,_CAL_QUERY_CAP)).long()
    qraw=q[ix]
    xq=(qraw.reshape(-1,qh,d)@R).reshape(qraw.shape) if rot and R is not None else qraw
    xk=(k.reshape(-1,kh,d)@R).reshape(k.shape) if rot and R is not None else k
    pq=_dyn_table(xq,None,False);pk=_dyn_table(xk,None,False)
    # V's public baseline is the real deployed mean/variance quantizer, not
    # unquantized V. Later joint calibration can still add the predictor.
    vp=_dyn_table(v,None,False)
    unitv=_params_unit_flat(vp);vq=_deq_params(vp)
    if rows>1:
        error=v-vq;penalty=1./rows
        bestcost=error.mean(0).square()+penalty*error.square().mean(0);shift=error.mean(0)
        for _ in range(8):
            proposal=torch.round((v+shift)/unitv*4.).clamp(-7.,7.)*(unitv*.25)
            error=v-proposal;mean=error.mean(0)
            cost=mean.square()+penalty*error.square().mean(0)
            accept=cost<bestcost
            vq=torch.where(accept[None],proposal,vq);bestcost=torch.minimum(cost,bestcost)
            shift=shift+mean*.75
    target=_cal_output(qraw,k,v,qh,kh,d)
    qbase,kbase=_deq_params(pq),_deq_params(pk)
    unitq,unitk=_params_unit_flat(pq),_params_unit_flat(pk)
    def loss(qv,kv):return (_cal_output(qv,kv,vq,qh,kh,d)-target).square().mean()
    qg=_gptq_quantize_batched(xq.reshape(-1,qh,d).transpose(0,1).contiguous(),
          unitq.reshape(-1,qh,d).transpose(0,1).contiguous(),Uq.repeat_interleave(rep,0))
    kg=_gptq_quantize_batched(xk.reshape(-1,kh,d).transpose(0,1).contiguous(),
          unitk.reshape(-1,kh,d).transpose(0,1).contiguous(),Uk)
    qg=qg.transpose(0,1).reshape(xq.shape);kg=kg.transpose(0,1).reshape(xk.shape)
    baseline=loss(qbase,kbase);candidate=loss(qg,kg)
    gq=int(ATTN_GPTQ_ENABLE and bool(torch.isfinite(candidate) and candidate<baseline))
    qcur,kcur=(qg,kg) if gq else (qbase,kbase)
    current=candidate if gq else baseline
    rfq=rfk=0
    if rows<=ATTN_REFINE_MAX_T:
        Hk,Hq=Hk.bfloat16().float(),Hq.bfloat16().float()
        rounds=32 if rows<=128 else (4 if rows>1024 else _REF_ATTN_ROUNDS)
        if ATTN_REFINE_Q_SWEEPS>0:
            newq=_refine_attn_heads(xq,qcur,unitq,Hk,qh,ATTN_REFINE_Q_SWEEPS,
                                    total_rounds=ATTN_REFINE_Q_SWEEPS*rounds)
            trial=loss(newq,kcur)
            if bool(torch.isfinite(trial) and (not ATTN_REFINE_GUARD or trial<current)):
                qcur=newq;current=trial;rfq=ATTN_REFINE_Q_SWEEPS
        if ATTN_REFINE_K_SWEEPS>0:
            newk=_refine_attn_heads(xk,kcur,unitk,Hq,kh,ATTN_REFINE_K_SWEEPS,
                                    total_rounds=ATTN_REFINE_K_SWEEPS*rounds)
            trial=loss(qcur,newk)
            if bool(torch.isfinite(trial) and (not ATTN_REFINE_GUARD or trial<current)):
                rfk=ATTN_REFINE_K_SWEEPS
    return gq,rfq,rfk
