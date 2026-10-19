"""Q-only calibration compensation for the deployed K quantization response.

No current-test K/V is read by Q. The correction is fused into its existing
channel transform; K, V, Linear, scale selection and search budgets are frozen.
"""
KR_DEBUG={}
_kr_parent_calibration=hif4_calibration_attention


def _kr_map(key,decoded,ridge=.1):
    # key, decoded: KV-head x token x channel. Centering removes the
    # softmax-invariant common key shift, including the K translation search.
    key=key-key.mean(1,keepdim=True)
    decoded=decoded-decoded.mean(1,keepdim=True)
    h=decoded.transpose(1,2)@decoded/decoded.shape[1]
    cross=(key-decoded).transpose(1,2)@decoded/decoded.shape[1]
    eye=torch.eye(h.shape[-1],dtype=h.dtype,device=h.device)
    damp=h.diagonal(dim1=-2,dim2=-1).mean(-1).clamp_min(1.e-12)*ridge
    delta=torch.linalg.solve(h+damp[:,None,None]*eye,cross.transpose(1,2)).transpose(1,2)
    return eye+delta


def _kr_transform(st,matrix,d,kh):
    if 'joint_transform' in st:base=st['joint_transform'].float()
    elif st.get('rot'):base=_make_R(d,int(st.get('rseed') or 0)).expand(kh,-1,-1)
    else:base=torch.eye(d).expand(kh,-1,-1)
    return dict(st,joint_transform=(base@matrix).contiguous())


@torch.inference_mode()
def hif4_calibration_attention(samples,q_num_heads,kv_num_heads,head_dim):
    parent=_kr_parent_calibration(samples,q_num_heads,kv_num_heads,head_dim)
    qh,kh,d=q_num_heads,kv_num_heads,head_dim
    KR_DEBUG.clear();KR_DEBUG.update(accepted=0,error=False)
    st=parent['q_state']
    if len(samples)<3 or not st.get('gq') or 'u' not in st or d%64:return parent
    hold=samples[-1];n=len(hold['q'][0])
    if not 4<=n<=4096:return parent
    try:
        keys=[];codes=[]
        for sample in samples[:-1][-2:]:
            if not 4<=len(sample['k'][0])<=4096:return parent
            raw=dequantize_nvfp4(*sample['k']).float()
            key=_kt_input(raw,parent['k_state'],kh,d,'k').reshape(-1,kh,d).transpose(0,1)
            decoded=_deq_params(hif4_dynamic_quantize_k(*sample['k'],kh,d,parent['k_state'])).reshape(-1,kh,d).transpose(0,1)
            # Each sequence has its own translation; center before concatenation.
            keys.append(key-key.mean(1,keepdim=True))
            codes.append(decoded-decoded.mean(1,keepdim=True))
        matrix=_kr_map(torch.cat(keys,1),torch.cat(codes,1))
        if not bool(torch.isfinite(matrix).all()):return parent
        ix=torch.linspace(0,n-1,min(n,32)).long()
        raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
        raw += [dequantize_nvfp4(*hold[r]).float() for r in 'kv']
        key=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
        value=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
        target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
        def cost(state):
            query=_joint_quant_queries(hold,state,qh,kh,d,ix)
            err=(_joint_output(query,key,value,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
            return torch.stack((err[::2].mean(0),err[1::2].mean(0)))
        before=cost(st);best=before.mean(0);eye=torch.eye(d)
        selected=eye.expand(kh,-1,-1).clone();take=torch.zeros(kh,dtype=torch.bool)
        ratios=[]
        for strength in (.5,1.):
            proposal=eye+strength*(matrix-eye)
            after=cost(_kr_transform(st,proposal,d,kh))
            eligible=torch.isfinite(after).all(0)&torch.isfinite(before).all(0)&(after<before*.995).all(0)&(after.mean(0)<best)
            selected=torch.where(eligible[:,None,None],proposal,selected)
            best=torch.where(eligible,after.mean(0),best);take|=eligible
            ratios.append((after.mean(0)/before.mean(0).clamp_min(1.e-30)).tolist())
        KR_DEBUG.update(accepted=int(take.sum()),ratios=ratios,max_correction=float((matrix-eye).abs().max()))
        if bool(take.any()):return dict(parent,q_state=_kr_transform(st,selected,d,kh))
    except (RuntimeError,ValueError) as exc:
        KR_DEBUG.update(error=True,error_type=type(exc).__name__)
    return parent
