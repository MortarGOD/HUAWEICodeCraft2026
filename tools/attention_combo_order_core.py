"""Same components, new calibration order: Q output metric before K shifts."""
_co_metric_calibration=hif4_calibration_attention
# The saved output-metric function resolves this global when it runs.
# Bypass KT/TS there, and execute each once after Q has been selected.
_oj_parent_calibration=_kt_parent_calibration
CO_DEBUG={}


@torch.inference_mode()
def hif4_calibration_attention(samples,qh,kh,d):
    parent=_co_metric_calibration(samples,qh,kh,d)
    CO_DEBUG.clear();CO_DEBUG.update(order='OJ-KT-TS',error=False,oj=dict(OJ_DEBUG))
    try:
        shifted=_kt_guard(samples,parent,qh,kh,d)
    except (RuntimeError,ValueError) as exc:
        CO_DEBUG.update(error=True,stage='KT',error_type=type(exc).__name__)
        shifted=parent
    try:
        answer=_ts_guard(samples,shifted,qh,kh,d)
    except (RuntimeError,ValueError) as exc:
        CO_DEBUG.update(error=True,stage='TS',error_type=type(exc).__name__)
        answer=shifted
    CO_DEBUG.update(kt=dict(KT_DEBUG),ts=dict(TS_DEBUG))
    return {k:v for k,v in answer.items() if k!='_ts_cache'}
