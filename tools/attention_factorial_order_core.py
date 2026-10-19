"""Complete the late-temperature / K-first calibration combination."""
_kf_early_calibration=_kt_parent_calibration
KF_DEBUG={}


def _kf_before_metric(samples,qh,kh,d):
    parent=_kf_early_calibration(samples,qh,kh,d)
    KF_DEBUG.clear();KF_DEBUG.update(order='KT-TS-OJ-TEMP',error=False)
    try:
        shifted=_kt_guard(samples,parent,qh,kh,d)
    except (RuntimeError,ValueError) as exc:
        KF_DEBUG.update(error=True,stage='KT',error_type=type(exc).__name__)
        shifted=parent
    try:
        answer=_ts_guard(samples,shifted,qh,kh,d)
    except (RuntimeError,ValueError) as exc:
        KF_DEBUG.update(error=True,stage='TS',error_type=type(exc).__name__)
        answer=shifted
    return {k:v for k,v in answer.items() if k not in ('_vr_cache','_ts_cache')}


# The existing final TEMP wrapper calls OJ; OJ calls the K-first wrapper.
# Do not call the v381 OJ-KT-TS wrapper, which would repeat KT/TS.
_oj_parent_calibration=_kf_before_metric
_lt_parent_calibration=_co_metric_calibration
