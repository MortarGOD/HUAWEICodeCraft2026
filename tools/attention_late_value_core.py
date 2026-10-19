"""Fit conditional V regression once, after final Q/K and temperature."""
_lv_parent_calibration=hif4_calibration_attention
# _vr_parent_calibration is the saved VM stage, before conditional VR.
# OJ resolves its parent dynamically; keep VM, defer only VR.
_oj_parent_calibration=_vr_parent_calibration
LV_DEBUG={}


@torch.inference_mode()
def hif4_calibration_attention(samples,qh,kh,d):
    parent=_lv_parent_calibration(samples,qh,kh,d)
    LV_DEBUG.clear();LV_DEBUG.update(order='VM-OJ-KT-TS-TEMP-VR',error=False)
    if len(samples)<2:return parent
    try:
        answer=_vr_guard(samples,parent,qh,kh,d)
    except (RuntimeError,ValueError) as exc:
        LV_DEBUG.update(error=True,error_type=type(exc).__name__)
        answer=parent
    LV_DEBUG.update(value=dict(VR_DEBUG))
    return {k:v for k,v in answer.items() if k not in ('_vr_cache','_ts_cache')}
