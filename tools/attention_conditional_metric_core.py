"""Two query-direction conditioned output metrics; calibration K/V only."""
CM_DEBUG={}
_cm_parent_calibration=hif4_calibration_attention
_cm_parent_q=hif4_dynamic_quantize_q


def _cm_fit(samples,states,qh,kh,d):
    queries=[[] for _ in range(kh)];metrics=[[] for _ in range(kh)]
    for sample in samples[:-1][-2:]:
        n=sample['k'][0].shape[0]
        ix=torch.linspace(0,n-1,min(n,16)).long()
        rawq=dequantize_nvfp4(sample['q'][0][ix],sample['q'][1][ix]).float()
        rawk=dequantize_nvfp4(*sample['k']).float()
        rawv=dequantize_nvfp4(*sample['v']).float()
        q=_kt_input(rawq,states['q_state'],qh,d,'q').reshape(-1,kh,qh//kh,d)
        k=_kt_input(rawk,states['k_state'],kh,d,'k').reshape(n,kh,d)
        v=rawv.reshape(n,kh,d)
        for h in range(kh):
            x=q[:,h].reshape(-1,d);key=k[:,h];value=v[:,h]
            prob=torch.softmax(x@key.T/d**.5,-1);out=prob@value
            queries[h].append(x)
            for i in range(len(x)):
                jac=key.T@(prob[i,:,None]*(value-out[i]))/d**.5
                metrics[h].append(jac@jac.T)
    base=_joint_raw_metric(states['q_state']).float()
    centers=[];banks=[]
    for h in range(kh):
        x=torch.cat(queries[h]);x=x/x.norm(dim=-1,keepdim=True).clamp_min(1.e-20)
        # Deterministic farthest initialization, then four spherical updates.
        first=0;second=int((x@x[first]).argmin())
        c=x[[first,second]].clone()
        for _ in range(4):
            label=(x@c.T).argmax(-1)
            for j in range(2):
                if bool((label==j).any()):
                    mean=x[label==j].mean(0);c[j]=mean/mean.norm().clamp_min(1.e-20)
        label=(x@c.T).argmax(-1)
        allh=torch.stack(metrics[h]);row=[]
        old=base[h]/base[h].diagonal().mean().clamp_min(1.e-20)
        for j in range(2):
            selected=allh[label==j]
            metric=selected.mean(0) if len(selected) else allh.mean(0)
            metric=metric/metric.diagonal().mean().clamp_min(1.e-20)
            row.append(.5*old+.5*metric)
        centers.append(c);banks.append(torch.stack(row))
    return torch.stack(centers).contiguous(),torch.stack(banks).contiguous()


def _cm_trial(raw,st,qh,d,parent):
    x=_kt_input(raw,st,qh,d,'q');n=len(x)
    centers=st['cm_centers'].repeat_interleave(qh//st['cm_centers'].shape[0],0)
    direction=x.reshape(n,qh,d)
    label=torch.einsum('thd,hcd->thc',direction,centers).argmax(-1)
    bank=st['cm_metrics'];best=parent
    base=_deq_params(parent);unit=_params_unit_flat(parent)
    # Each proposal starts at the parent's actual code. Fixed metadata.
    values=base.reshape(n,qh,d).clone()
    for j in range(2):
        metric=bank[:,j].contiguous()
        proposal=_refine_attn_heads(x,base,unit,metric,qh,1,total_rounds=4)
        hf=metric.repeat_interleave(qh//metric.shape[0],0)
        def cost(v):
            err=(v.reshape(n,qh,d)-direction).transpose(0,1)
            return ((err@hf)*err).sum(-1).T
        take=(label==j)&(cost(proposal)<cost(base))
        if 'cm_active' in st:
            take &= st['cm_active'].repeat_interleave(qh//st['cm_active'].numel())[None]
        values=torch.where(take[:,:,None],proposal.reshape(n,qh,d),values)
    return _values_to_params(values.reshape_as(x),parent)


@torch.inference_mode()
def hif4_calibration_attention(samples,q_num_heads,kv_num_heads,head_dim):
    parent=_cm_parent_calibration(samples,q_num_heads,kv_num_heads,head_dim)
    qh,kh,d=q_num_heads,kv_num_heads,head_dim
    CM_DEBUG.clear();CM_DEBUG.update(accepted=0,error=False)
    st=parent['q_state']
    if len(samples)<3 or not st.get('gq') or 'u' not in st or d%64:return parent
    hold=samples[-1];n=hold['q'][0].shape[0]
    if not 4<=n<=4096:return parent
    try:
        centers,bank=_cm_fit(samples,parent,qh,kh,d)
        trial=dict(st,cm_centers=centers,cm_metrics=bank)
        ix=torch.linspace(0,n-1,min(n,32)).long()
        raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
        raw += [dequantize_nvfp4(*hold[r]).float() for r in 'kv']
        original=_joint_quant_queries(hold,st,qh,kh,d,ix)
        source=_kt_input(raw[0],st,qh,d,'q')
        pp=_values_to_params(original,_dyn_table(source,None,has_scale=False))
        candidate=_deq_params(_cm_trial(raw[0],trial,qh,d,pp))
        key=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
        value=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
        target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
        def cost(query):
            err=(_joint_output(query,key,value,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
            return torch.stack((err[::2].mean(0),err[1::2].mean(0)))
        before,after=cost(original),cost(candidate)
        take=(after<before*.995).all(0)&torch.isfinite(after).all(0)
        CM_DEBUG.update(accepted=int(take.sum()),ratios=(after.mean(0)/before.mean(0).clamp_min(1.e-30)).tolist())
        if bool(take.any()):return dict(parent,q_state=dict(trial,cm_active=take))
    except (RuntimeError,ValueError) as exc:
        CM_DEBUG.update(error=True,error_type=type(exc).__name__)
    return parent


def hif4_dynamic_quantize_q(q_quant,q_scale,q_num_heads,head_dim,q_state):
    parent=_cm_parent_q(q_quant,q_scale,q_num_heads,head_dim,q_state)
    if not isinstance(q_state,dict) or 'cm_active' not in q_state or not 4<=q_quant.shape[0]<=4096:return parent
    return _cm_trial(dequantize_nvfp4(q_quant,q_scale).float(),q_state,q_num_heads,head_dim,parent)
