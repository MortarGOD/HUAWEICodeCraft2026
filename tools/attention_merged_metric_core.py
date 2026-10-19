"""v374 ablation: merge its two fitted metrics, run one four-round refinement."""
_mm_fit_parent = _cm_fit


def _cm_fit(samples,states,qh,kh,d):
    centers,bank=_mm_fit_parent(samples,states,qh,kh,d)
    return centers,bank.mean(1,keepdim=True).expand_as(bank).contiguous()


def _cm_trial(raw,st,qh,d,parent):
    x=_kt_input(raw,st,qh,d,'q');n=len(x)
    metric=st['cm_metrics'][:,0].contiguous()
    base=_deq_params(parent);unit=_params_unit_flat(parent)
    proposal=_refine_attn_heads(x,base,unit,metric,qh,1,total_rounds=4)
    hf=metric.repeat_interleave(qh//metric.shape[0],0)
    def cost(value):
        err=(value-x).reshape(n,qh,d).transpose(0,1)
        return ((err@hf)*err).sum(-1).T
    take=cost(proposal)<cost(base)
    if 'cm_active' in st:
        take &= st['cm_active'].repeat_interleave(qh//st['cm_active'].numel())[None]
    values=torch.where(take[:,:,None],proposal.reshape(n,qh,d),base.reshape(n,qh,d))
    return _values_to_params(values.reshape_as(x),parent)
