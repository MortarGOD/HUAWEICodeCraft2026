"""Choose one four-round metric per KV head using calibration output loss."""
def _ms_choose(before, costs):
    # costs: metric x fold x KV-head. Both folds must beat the parent.
    eligible=(costs<before[None]*.995).all(1)&torch.isfinite(costs).all(1)&torch.isfinite(before).all(0)[None]
    risk=costs.mean(1).masked_fill(~eligible,float('inf'))
    choice=risk.argmin(0)
    return choice,eligible.any(0)


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
        old=_joint_raw_metric(st).float()
        old=old/old.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None].clamp_min(1.e-20)
        banks=[bank,old[:,None].expand_as(bank).contiguous()]
        ix=torch.linspace(0,n-1,min(n,32)).long()
        raw=[dequantize_nvfp4(hold['q'][0][ix],hold['q'][1][ix]).float()]
        raw += [dequantize_nvfp4(*hold[r]).float() for r in 'kv']
        original=_joint_quant_queries(hold,st,qh,kh,d,ix)
        source=_kt_input(raw[0],st,qh,d,'q')
        pp=_values_to_params(original,_dyn_table(source,None,has_scale=False))
        key=_deq_params(hif4_dynamic_quantize_k(*hold['k'],kh,d,parent['k_state']))
        value=_deq_params(hif4_dynamic_quantize_v(*hold['v'],kh,d,parent['v_state']))
        target=_joint_output(*raw,qh,kh,d,ix,query_subset=True)
        def cost(query):
            err=(_joint_output(query,key,value,qh,kh,d,ix,query_subset=True)-target).square().mean((-1,-2))
            return torch.stack((err[::2].mean(0),err[1::2].mean(0)))
        before=cost(original);costs=[]
        for metric in banks:
            trial=dict(st,cm_centers=centers,cm_metrics=metric)
            costs.append(cost(_deq_params(_cm_trial(raw[0],trial,qh,d,pp))))
        costs=torch.stack(costs)
        choice,take=_ms_choose(before,costs)
        CM_DEBUG.update(accepted=int(take.sum()),original_metric=int(((choice==1)&take).sum()),
                        output_metric=int(((choice==0)&take).sum()),choices=choice.tolist())
        if bool(take.any()):
            selected=torch.where((choice==1)[:,None,None,None],banks[1],banks[0])
            return dict(parent,q_state=dict(st,cm_centers=centers,cm_metrics=selected.contiguous(),cm_active=take))
    except (RuntimeError,ValueError) as exc:
        CM_DEBUG.update(error=True,error_type=type(exc).__name__)
    return parent
