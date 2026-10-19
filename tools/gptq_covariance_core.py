"""Fresh deployed-coordinate Grams, optional softmax-gauge removal for Q."""
_COV_MODE='centered'
_COV_H_FP32=False
_COV_MAX_ROWS=4096
_PVFULL_GUARD_QUERIES=64
COV_DEBUG={}
PVFULL_DEBUG=COV_DEBUG
_cov_parent_calibration=hif4_calibration_attention


def _cov_transform(x,state,nh,kh,d,role):
    if role=='q' and 'qs' in state:
        x=_qks_apply_q(x,state['qs'].float(),nh,kh,d,inv=False)
    elif role=='k' and 'ks' in state:
        x=(x.reshape(-1,nh,d)*state['ks'].float()[None]).reshape(x.shape)
    y=x.reshape(-1,nh,d).transpose(0,1)
    if 'joint_transform' in state:
        transform=state['joint_transform'].float().repeat_interleave(nh//kh,0)
        y=y@transform
    elif state.get('rot'):
        y=y@_make_R(d,int(state.get('rseed') or 0))
    return y.contiguous()


def _cov_fit(samples,states,qh,kh,d):
    hk=torch.zeros(kh,d,d);hq=torch.zeros(kh,d,d)
    included=[];skipped=[];mean_trace=0.;total_trace=0.
    for sample in samples[:-1]:
        rows=sample['q'][0].shape[0]
        if _COV_MODE=='legacy' and rows>1024:
            skipped.append(rows);continue
        ix=torch.linspace(0,rows-1,min(rows,_COV_MAX_ROWS)).long()
        q=dequantize_nvfp4(sample['q'][0][ix],sample['q'][1][ix]).float()
        k=dequantize_nvfp4(sample['k'][0][ix],sample['k'][1][ix]).float()
        q=_cov_transform(q,states['q_state'],qh,kh,d,'q')
        k=_cov_transform(k,states['k_state'],kh,kh,d,'k')
        total_trace+=float(k.square().sum())
        mean_trace+=float(k.mean(1).square().sum())*k.shape[1]
        if _COV_MODE=='centered':k=k-k.mean(1,keepdim=True)
        hk+=k.transpose(1,2)@k
        hq+=(q.transpose(1,2)@q).reshape(kh,qh//kh,d,d).sum(1)
        included.append(dict(rows=rows,used=int(ix.numel())))
    COV_DEBUG.update(included=included,skipped=skipped,key_mean_trace_fraction=mean_trace/max(total_trace,1e-30))
    return hk,hq


def _cov_proposal(state,h,heads):
    h=(h+h.transpose(1,2))*.5
    u=_upper_cholesky_inv(h)
    if u is None:return None
    proposed=dict(state,u=u.repeat_interleave(heads//u.shape[0],0).contiguous())
    if 'H' in state:
        carried=h if _COV_H_FP32 else h.bfloat16()
        proposed['H']=carried.repeat_interleave(heads//h.shape[0],0).contiguous()
    return proposed


def hif4_calibration_attention(samples,qh,kh,d):
    states=_cov_parent_calibration(samples,qh,kh,d)
    COV_DEBUG.clear();COV_DEBUG.update(q_heads=0,k_heads=0,error=False)
    if (len(samples)<2 or not states['q_state'].get('gq') or not states['k_state'].get('gq')
            or not 4<=samples[-1]['q'][0].shape[0]<=4096):
        COV_DEBUG['ineligible']=True
        return states
    try:
        hk,hq=_cov_fit(samples,states,qh,kh,d)
        proposed_q=_cov_proposal(states['q_state'],hk,qh)
        proposed_k=_cov_proposal(states['k_state'],hq,kh)
        if proposed_q is None or proposed_k is None:
            COV_DEBUG['factorization_failed']=True
            return states
        return _pvfull_guard(samples[-1],states,dict(states,q_state=proposed_q,k_state=proposed_k),qh,kh,d)
    except Exception as exc:
        COV_DEBUG.update(error=True,error_type=type(exc).__name__)
        return states
