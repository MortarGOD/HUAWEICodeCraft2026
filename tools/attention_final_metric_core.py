"""Move Q output-metric selection AFTER K shifts and temperature.

Each existing stage executes once. No new dynamic algorithm or search budget.
The alternative order is not guaranteed to improve upon the original order.
"""
_fo_temperature_stage = hif4_calibration_attention
_fo_base_calibration = _kt_parent_calibration
FO_DEBUG = {}


def _fo_keys(samples, qh, kh, d):
    parent = _fo_base_calibration(samples, qh, kh, d)
    FO_DEBUG['order'].append('KT')
    try:
        shifted = _kt_guard(samples, parent, qh, kh, d)
    except (RuntimeError, ValueError) as exc:
        FO_DEBUG['errors'].append(('KT', type(exc).__name__))
        shifted = parent
    FO_DEBUG['order'].append('TS')
    try:
        answer = _ts_guard(samples, shifted, qh, kh, d)
    except (RuntimeError, ValueError) as exc:
        FO_DEBUG['errors'].append(('TS', type(exc).__name__))
        answer = shifted
    return {key: value for key, value in answer.items() if key not in ('_ts_cache', '_vr_cache')}


def _fo_prepare(samples, qh, kh, d):
    # The saved v387 temperature wrapper calls _lt_parent_calibration below.
    answer = _fo_temperature_stage(samples, qh, kh, d)
    FO_DEBUG['order'].append('TEMP')
    FO_DEBUG['temperature'] = dict(TEMP_DEBUG)
    FO_DEBUG['temperature_stage'] = dict(LT_DEBUG)
    return {key: value for key, value in answer.items() if key not in ('_ts_cache', '_vr_cache')}


_lt_parent_calibration = _fo_keys
_oj_parent_calibration = _fo_prepare


@torch.inference_mode()
def hif4_calibration_attention(samples, qh, kh, d):
    FO_DEBUG.clear(); FO_DEBUG.update(order=[], errors=[])
    # Saved OJ function: prepare K/TEMP first, then fit and guard Q once.
    answer = _co_metric_calibration(samples, qh, kh, d)
    FO_DEBUG['order'].append('OJ')
    FO_DEBUG['oj'] = dict(OJ_DEBUG)
    FO_DEBUG['kt'] = dict(KT_DEBUG)
    FO_DEBUG['ts'] = dict(TS_DEBUG)
    return {key: value for key, value in answer.items() if key not in ('_ts_cache', '_vr_cache')}
