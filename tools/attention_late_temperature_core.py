"""Same temperature calibration, applied once after final Q/K/V selection."""
_lt_parent_calibration=hif4_calibration_attention
_vm_parent_calibration=_temp_parent_calibration
LT_DEBUG={}


@torch.inference_mode()
def hif4_calibration_attention(samples,qh,kh,d):
    parent=_lt_parent_calibration(samples,qh,kh,d)
    LT_DEBUG.clear();LT_DEBUG.update(order='V-OJ-KT-TS-TEMP',error=False)
    TEMP_DEBUG.clear();TEMP_DEBUG.update(heads=0,error=False)
    if len(samples)<2 or d%64 or not parent['q_state'].get('gq') or not 4<=len(samples[-1]['v'][0])<=4096:return parent
    try:
        delta,_records=_temp_fit(samples,parent,qh,kh,d)
        answer=_temp_guard(samples[-1],parent,delta,qh,kh,d)
    except (RuntimeError,ValueError) as exc:
        LT_DEBUG.update(error=True,error_type=type(exc).__name__)
        answer=parent
    LT_DEBUG.update(temperature=dict(TEMP_DEBUG))
    return {k:v for k,v in answer.items() if k not in ('_vr_cache','_ts_cache')}
