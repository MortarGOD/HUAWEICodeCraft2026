"""Preserve any already accepted early-temperature dependency."""
SG_DEBUG={}
_sg_parent_calibration=hif4_calibration_attention


def _sg_protected(state):
    gain=state['q_state'].get('temperature_gain')
    return isinstance(gain,torch.Tensor) and bool((gain!=1).any())


@torch.inference_mode()
def hif4_calibration_attention(samples,qh,kh,d):
    parent=_sg_parent_calibration(samples,qh,kh,d)
    SG_DEBUG.clear();SG_DEBUG.update(protected=False,error=False)
    if _sg_protected(parent):
        SG_DEBUG['protected']=True
        return parent
    if len(samples)<2 or d%64 or not parent['q_state'].get('gq') or not 4<=len(samples[-1]['v'][0])<=4096:return parent
    TEMP_DEBUG.clear();TEMP_DEBUG.update(heads=0,error=False)
    try:
        delta,_records=_temp_fit(samples,parent,qh,kh,d)
        answer=_temp_guard(samples[-1],parent,delta,qh,kh,d)
    except (RuntimeError,ValueError) as exc:
        SG_DEBUG.update(error=True,error_type=type(exc).__name__)
        answer=parent
    SG_DEBUG.update(temperature=dict(TEMP_DEBUG))
    return {k:v for k,v in answer.items() if k not in ('_vr_cache','_ts_cache')}
